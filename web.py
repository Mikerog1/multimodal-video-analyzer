import os
import sys
import uuid
import time
import asyncio
import json as py_json
import concurrent.futures
from fastapi import FastAPI, Request, UploadFile, File, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse

# Add core directory to PATH on Windows to allow OpenCV to locate the openh264 DLL
if os.name == 'nt':
    core_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'core'))
    os.environ['PATH'] = core_dir + os.pathsep + os.environ.get('PATH', '')
    if hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(core_dir)
        except Exception:
            pass

import torch
from core.video_processor import VideoProcessor

app = FastAPI(title="Multimodal Video Analyzer Web API")

# Ensure necessary directories exist using paths relative to THIS script file,
# not the current working directory, so the server works regardless of launch location.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(BASE_DIR, "input")
output_dir = os.path.join(BASE_DIR, "output")
static_dir = os.path.join(BASE_DIR, "static")
models_dir = os.path.join(BASE_DIR, "models")

os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(static_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Mount static files using absolute paths
app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.mount("/output", StaticFiles(directory=output_dir), name="output")

# In-memory dictionary to track task status
tasks = {}

executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def generate_captions_with_qwen(video_path: str) -> str:
    """Loads Qwen2-VL-2B-Instruct locally and generates a caption description for the video."""
    try:
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "Qwen/Qwen2-VL-2B-Instruct"
        
        # Load model and processor
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 360,
                        "fps": 0.5,
                    },
                    {"type": "text", "text": "Describe the content and main actions of this video briefly in 1-2 sentences."},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_kwargs=video_kwargs,
            padding=True,
            return_tensors="pt"
        ).to(device)

        generated_ids = model.generate(**inputs, max_new_tokens=100)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0].strip()
    except Exception as e:
        print(f"[-] Error in Qwen2-VL caption generation: {e}")
        return f"Error generating caption: {str(e)}"

class CustomCloudDetector:
    def __init__(self, api_url: str, api_key: str, confidence: float = 0.5):
        self.api_url = api_url
        self.api_key = api_key
        self.confidence = confidence
        self.model_id = "custom_cloud_detector"
        self.device = "cloud"

    def detect(self, pil_image) -> list[dict]:
        import io
        import requests
        try:
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG")
            img_bytes = buffer.getvalue()
            
            files = {"file": ("frame.jpg", img_bytes, "image/jpeg")}
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            
            response = requests.post(self.api_url, files=files, headers=headers, timeout=10)
            if response.status_code == 200:
                results = response.json()
                filtered = []
                for det in results:
                    score = det.get("score", 1.0)
                    if score >= self.confidence:
                        filtered.append({
                            "label": det.get("label", "unknown"),
                            "score": score,
                            "box": det.get("box", [0, 0, 0, 0])
                        })
                return filtered
            else:
                print(f"[-] Cloud API returned code {response.status_code}: {response.text}")
        except Exception as e:
            print(f"[-] Error calling custom cloud detector: {e}")
        return []

def extract_keyframes(video_path: str, max_keyframes: int = 6) -> list[str]:
    """Extracts base64 encoded JPEGs uniformly distributed across the video."""
    import cv2
    import base64
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []
        
    step = max(1, total_frames // max_keyframes)
    keyframes = []
    
    for i in range(max_keyframes):
        frame_idx = min(total_frames - 1, i * step)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        b64 = base64.b64encode(buffer).decode("utf-8")
        keyframes.append(b64)
        
    cap.release()
    return keyframes

def generate_captions_with_gemini(video_path: str, api_key: str) -> str:
    import requests
    keyframes = extract_keyframes(video_path, max_keyframes=6)
    if not keyframes:
        return "Could not extract video keyframes."
        
    parts = [{"text": "These are chronological keyframes from a video. Describe the video content, main actions, and environment briefly in 2-3 sentences."}]
    for b64 in keyframes:
        parts.append({
            "inlineData": {
                "mimeType": "image/jpeg",
                "data": b64
            }
        })
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {"contents": [{"parts": parts}]}
    try:
        res = requests.post(url, json=payload, timeout=20)
        res.raise_for_status()
        data = res.json()
        return data["contents"][0]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"[-] Gemini caption generation failed: {e}")
        return f"Gemini caption generation error: {str(e)}"

def generate_captions_with_custom_vlm(video_path: str, api_url: str, api_key: str, model_id: str) -> str:
    import requests
    keyframes = extract_keyframes(video_path, max_keyframes=6)
    if not keyframes:
        return "Could not extract video keyframes."
        
    content = [{"type": "text", "text": "These are chronological keyframes from a video. Describe the video content, main actions, and environment briefly in 2-3 sentences."}]
    for b64 in keyframes:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}"
            }
        })
        
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        
    url = api_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = f"{url}/chat/completions"
        
    payload = {
        "model": model_id or "gpt-4o",
        "messages": [{"role": "user", "content": content}]
    }
    try:
        res = requests.post(url, json=payload, headers=headers, timeout=30)
        res.raise_for_status()
        data = res.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[-] Custom VLM caption generation failed: {e}")
        return f"Custom VLM caption generation error: {str(e)}"

def run_analysis_single(
    task_id: str,
    video_path: str,
    output_dir: str,
    model_type: str,
    model_id: str,
    device: str,
    codec: str,
    confidence: float,
    fps_sample: float,
    resize_factor: float,
    save_sampled_only: bool,
    generate_qa: bool,
    qa_categories: str,
    mask_persons: bool = False,
    generate_json: bool = True,
    generate_video: bool = True,
    progress_callback=None,
    captions: str = None,
    example_questions: str = None,
    custom_detector_id: str = None,
    detector_api_url: str = None,
    detector_api_key: str = None,
    vlm_model: str = "none",
    gemini_api_key: str = None,
    vlm_api_url: str = None,
    vlm_api_key: str = None,
    vlm_model_id: str = None,
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the appropriate model
    if model_type == "yolo":
        from models.yolo_detector import YoloDetector
        if not model_id:
            model_id = os.path.join("models", "yolov8n.pt")
        elif not os.path.dirname(model_id):
            model_id = os.path.join("models", model_id)
        detector = YoloDetector(model_id, device, confidence)
    elif model_type == "custom_local":
        from models.yolo_detector import YoloDetector
        from models.detr_detector import DetrDetector
        det_id = custom_detector_id or model_id or "yolov8n.pt"
        if "detr" in det_id.lower() or "facebook" in det_id.lower():
            detector = DetrDetector(det_id, device, confidence)
        else:
            if not os.path.dirname(det_id) and not det_id.endswith(".pt") and '/' not in det_id:
                det_id = os.path.join("models", det_id)
            detector = YoloDetector(det_id, device, confidence)
    elif model_type == "custom_api":
        detector = CustomCloudDetector(api_url=detector_api_url, api_key=detector_api_key, confidence=confidence)
    else:
        from models.detr_detector import DetrDetector
        if not model_id:
            model_id = "facebook/detr-resnet-50"
        detector = DetrDetector(model_id, device, confidence)
        
    processor = VideoProcessor(detector)
    
    tasks[task_id]["model_info"] = {
        "model_type": "YOLO" if model_type == "yolo" else ("DETR" if model_type == "detr" else "CUSTOM"),
        "model_name": detector.model_id.split(os.sep)[-1].split('/')[-1] if hasattr(detector, "model_id") else str(model_id or "custom"),
        "device": str(getattr(detector, "device", device)).upper()
    }
    
    existing_folders = set(os.listdir(output_dir)) if os.path.exists(output_dir) else set()
    
    def update_progress(current, total):
        if progress_callback:
            progress_callback(current, total)
        else:
            tasks[task_id]["progress"] = round((current / total) * 100) if total > 0 else 0

    qa_cats = [c.strip() for c in qa_categories.split(',')] if qa_categories else []

    processor.process_video(
        video_path=video_path,
        output_dir=output_dir,
        fps_sample=fps_sample,
        codec=codec,
        resize_factor=resize_factor,
        save_sampled_only=save_sampled_only,
        generate_qa=generate_qa,
        qa_categories=qa_cats,
        progress_callback=update_progress,
        mask_persons=mask_persons,
        write_json=generate_json,
        generate_video=generate_video,
        captions=captions,
        example_questions=example_questions,
        gemini_api_key=gemini_api_key,
        custom_vlm_url=vlm_api_url,
        custom_vlm_key=vlm_api_key,
        custom_vlm_model_id=vlm_model_id,
    )
    
    current_folders = set(os.listdir(output_dir))
    new_folders = current_folders - existing_folders
    
    if new_folders:
        run_folder = sorted(list(new_folders))[-1]
        run_path = os.path.join(output_dir, run_folder)
        
        # Find the output files
        files = os.listdir(run_path)
        video_exts = ('.mp4', '.avi', '.mkv', '.webm', '.mov')
        analyzed_video = next((f for f in files if f.endswith(video_exts) and '_analyzed' in f), None)
        original_video = next((f for f in files if f.endswith(video_exts) and '_original' in f), None)
        any_video = next((f for f in files if f.endswith(video_exts)), None) if not analyzed_video and not original_video else None
        video_file = analyzed_video or original_video or any_video
        csv_file = next((f for f in files if f.endswith('.csv') and f != 'total_report.csv'), None)
        json_file = next((f for f in files if f.endswith('.json') and '_qa_' not in f), None)
        qa_json_files = sorted([f for f in files if f.endswith('.json') and '_qa_' in f])
        
        tasks[task_id]["results"] = {
            "folder": f"/output/{run_folder}",
            "video": f"/output/{run_folder}/{video_file}" if video_file else None,
            "is_original_video": analyzed_video is None and video_file is not None,
            "csv": f"/output/{run_folder}/{csv_file}" if csv_file else None,
            "json": f"/output/{run_folder}/{json_file}" if json_file else None,
            "qa_json_files": [f"/output/{run_folder}/{f}" for f in qa_json_files],
        }

        # Read analysis settings from the generated JSON
        analysis_settings = None
        object_counts = {}
        if json_file:
            try:
                with open(os.path.join(run_path, json_file), "r", encoding="utf-8") as jf:
                    report_data = py_json.load(jf)
                    meta = report_data.get("metadata", {})
                    analysis_settings = _extract_analysis_settings(meta)
                    for obj in report_data.get("objects", []):
                        obj_type = obj.get("object_type", "unknown")
                        object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
            except Exception:
                pass
        tasks[task_id]["analysis_settings"] = analysis_settings
        tasks[task_id]["object_counts"] = object_counts
    else:
        raise RuntimeError("No output directory was created.")

def run_analysis(
    task_id: str,
    video_path: str,
    output_dir: str,
    model_type: str,
    model_id: str,
    device: str,
    codec: str,
    confidence: float,
    fps_sample: float,
    resize_factor: float,
    save_sampled_only: bool,
    generate_qa: bool,
    qa_categories: str,
    mask_persons: bool = False,
    generate_json: bool = True,
    generate_video: bool = True,
    hf_repo_id: str = None,
    hf_file_path: str = None,
    hf_token: str = None,
    captions: str = None,
    example_questions: str = None,
    auto_generate_captions: bool = False,
    custom_detector_id: str = None,
    detector_api_url: str = None,
    detector_api_key: str = None,
    vlm_model: str = "none",
    gemini_api_key: str = None,
    vlm_api_url: str = None,
    vlm_api_key: str = None,
    vlm_model_id: str = None,
):
    try:
        if hf_repo_id and hf_file_path:
            tasks[task_id]["status"] = "downloading_dataset"
            tasks[task_id]["progress"] = 0
            
            upload_dir = os.path.join(input_dir, task_id)
            os.makedirs(upload_dir, exist_ok=True)
            
            from huggingface_hub import hf_hub_download
            video_path = hf_hub_download(
                repo_id=hf_repo_id,
                filename=hf_file_path,
                repo_type="dataset",
                local_dir=upload_dir,
                token=hf_token if hf_token else None
            )
            tasks[task_id]["filename"] = os.path.basename(video_path)

        # Generate captions using selected VLM if requested or auto-generate is enabled
        if not captions or not captions.strip():
            if auto_generate_captions or vlm_model in ("qwen", "gemini", "custom_vlm"):
                if vlm_model == "gemini" and gemini_api_key:
                    tasks[task_id]["status"] = "generating_captions"
                    tasks[task_id]["progress"] = 0
                    captions = generate_captions_with_gemini(video_path, gemini_api_key)
                elif vlm_model == "custom_vlm" and vlm_api_url:
                    tasks[task_id]["status"] = "generating_captions"
                    tasks[task_id]["progress"] = 0
                    captions = generate_captions_with_custom_vlm(video_path, vlm_api_url, vlm_api_key, vlm_model_id)
                else:
                    tasks[task_id]["status"] = "generating_captions"
                    tasks[task_id]["progress"] = 0
                    captions = generate_captions_with_qwen(video_path)

        if model_type == "all":
            models_to_run = [
                ("yolo", "yolo26n.pt"),
                ("yolo", "yolov8n.pt"),
                ("detr", "facebook/detr-resnet-50")
            ]
            
            last_results = None
            for idx, (m_type, m_id) in enumerate(models_to_run):
                tasks[task_id]["status"] = f"Running model {idx+1}/{len(models_to_run)} ({m_id})..."
                tasks[task_id]["progress"] = round((idx / len(models_to_run)) * 100)
                
                def sub_progress(current, total):
                    fraction = current / total if total > 0 else 0
                    tasks[task_id]["progress"] = round(((idx + fraction) / len(models_to_run)) * 100)
                
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Input video file not found at: {video_path}")
                
                run_analysis_single(
                    task_id=task_id,
                    video_path=video_path,
                    output_dir=output_dir,
                    model_type=m_type,
                    model_id=m_id,
                    device=device,
                    codec=codec,
                    confidence=confidence,
                    fps_sample=fps_sample,
                    resize_factor=resize_factor,
                    save_sampled_only=save_sampled_only,
                    generate_qa=generate_qa,
                    qa_categories=qa_categories,
                    mask_persons=mask_persons,
                    generate_json=generate_json,
                    generate_video=generate_video,
                    progress_callback=sub_progress,
                    captions=captions,
                    example_questions=example_questions,
                    gemini_api_key=gemini_api_key,
                    vlm_api_url=vlm_api_url,
                    vlm_api_key=vlm_api_key,
                    vlm_model_id=vlm_model_id,
                )
                last_results = tasks[task_id]["results"]
                
            tasks[task_id]["results"] = last_results
            tasks[task_id]["status"] = "completed"
        else:
            tasks[task_id]["status"] = "loading_model"
            tasks[task_id]["progress"] = 0
            
            run_analysis_single(
                task_id=task_id,
                video_path=video_path,
                output_dir=output_dir,
                model_type=model_type,
                model_id=model_id,
                device=device,
                codec=codec,
                confidence=confidence,
                fps_sample=fps_sample,
                resize_factor=resize_factor,
                save_sampled_only=save_sampled_only,
                generate_qa=generate_qa,
                qa_categories=qa_categories,
                mask_persons=mask_persons,
                generate_json=generate_json,
                generate_video=generate_video,
                captions=captions,
                example_questions=example_questions,
                custom_detector_id=custom_detector_id,
                detector_api_url=detector_api_url,
                detector_api_key=detector_api_key,
                vlm_model=vlm_model,
                gemini_api_key=gemini_api_key,
                vlm_api_url=vlm_api_url,
                vlm_api_key=vlm_api_key,
                vlm_model_id=vlm_model_id,
            )
            tasks[task_id]["status"] = "completed"
            
    except Exception as e:
        tasks[task_id]["status"] = "error"
        tasks[task_id]["error"] = str(e)
    finally:
        # Clean up temporary uploaded/downloaded input video file to avoid folder bloating
        try:
            if video_path and os.path.exists(video_path):
                if os.path.isfile(video_path):
                    os.remove(video_path)
                upload_dir = os.path.dirname(video_path)
                while upload_dir and upload_dir != input_dir and os.path.exists(upload_dir) and not os.listdir(upload_dir):
                    os.rmdir(upload_dir)
                    upload_dir = os.path.dirname(upload_dir)
        except Exception as cleanup_err:
            print(f"[-] Error cleaning up temporary file: {cleanup_err}")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html", "r") as f:
        return f.read()

def parse_hf_dataset_url(url: str) -> tuple[str, str | None]:
    """Parses a Hugging Face dataset URL into (repo_id, file_path).
    
    Supports formats:
    - https://huggingface.co/datasets/username/repo-name/resolve/main/video.mp4 -> (username/repo-name, video.mp4)
    - https://huggingface.co/datasets/username/repo-name/blob/main/video.mp4 -> (username/repo-name, video.mp4)
    - https://huggingface.co/datasets/username/repo-name -> (username/repo-name, None)
    - username/repo-name -> (username/repo-name, None)
    """
    url = url.strip()
    if "huggingface.co/datasets/" in url:
        path = url.split("huggingface.co/datasets/")[-1]
    else:
        path = url
        
    for sep in ("/resolve/", "/blob/"):
        if sep in path:
            repo_part, file_part = path.split(sep, 1)
            file_parts = file_part.split("/", 1)
            file_path = file_parts[1] if len(file_parts) > 1 else file_part
            return repo_part, file_path
            
    return path, None

@app.get("/api/hf/list-videos")
async def list_hf_videos(repo_id: str, token: str = None):
    """List all video files available in a Hugging Face dataset repository."""
    parsed_repo_id, file_path = parse_hf_dataset_url(repo_id)
    
    if not parsed_repo_id:
        return JSONResponse(status_code=400, content={"error": "Invalid Hugging Face Dataset repository link or ID"})
        
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token if token else None)
        files = api.list_repo_files(repo_id=parsed_repo_id, repo_type="dataset")
        
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        video_files = [f for f in files if f.lower().endswith(video_extensions)]
        
        return {
            "repo_id": parsed_repo_id,
            "videos": video_files,
            "auto_selected_file": file_path
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

def str_to_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes", "on")
    return bool(val)

@app.post("/api/analyze")
async def analyze_video(
    file: UploadFile = File(None),
    hf_repo_id: str = Form(None),
    hf_file_path: str = Form(None),
    hf_token: str = Form(None),
    model_type: str = Form("detr"),
    model_id: str = Form(""),
    device: str = Form("auto"),
    codec: str = Form("avc1"),
    confidence: float = Form(0.4),
    fps_sample: float = Form(1.0),
    resize_factor: float = Form(1.0),
    save_sampled_only: bool = Form(False),
    generate_qa: bool = Form(True),
    generate_json: bool = Form(True),
    qa_categories: str = Form(""),
    remove_audio: bool = Form(False),
    mask_persons: bool = Form(False),
    generate_video: bool = Form(True),
    captions: str = Form(None),
    example_questions: str = Form(None),
    auto_generate_captions: bool = Form(False),
    custom_detector_id: str = Form(None),
    detector_api_url: str = Form(None),
    detector_api_key: str = Form(None),
    vlm_model: str = Form("none"),
    gemini_api_key: str = Form(None),
    vlm_api_url: str = Form(None),
    vlm_api_key: str = Form(None),
    vlm_model_id: str = Form(None),
):
    save_sampled_only = str_to_bool(save_sampled_only)
    generate_qa = str_to_bool(generate_qa)
    generate_json = str_to_bool(generate_json)
    remove_audio = str_to_bool(remove_audio)
    mask_persons = str_to_bool(mask_persons)
    generate_video = str_to_bool(generate_video)
    auto_generate_captions = str_to_bool(auto_generate_captions)

    task_id = str(uuid.uuid4())
    
    filename = ""
    file_path = ""
    
    if file is not None:
        # Save the uploaded file in a unique folder to prevent name collisions
        # while preserving the original filename for cleaner output results.
        upload_dir = os.path.join(input_dir, task_id)
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        filename = file.filename
    elif hf_repo_id and hf_file_path:
        filename = os.path.basename(hf_file_path)
    else:
        return JSONResponse(status_code=400, content={"error": "Either a video file upload or a Hugging Face dataset file is required."})
    
    tasks[task_id] = {
        "status": "pending",
        "progress": 0,
        "filename": filename,
        "results": None,
        "model_info": None,
        "analysis_settings": None,
        "object_counts": {},
        "error": None
    }
    
    # Run analysis in a thread to avoid blocking the event loop
    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        executor, 
        run_analysis, 
        task_id, 
        file_path, 
        output_dir, 
        model_type,
        model_id,
        device,
        codec,
        confidence, 
        fps_sample, 
        resize_factor, 
        save_sampled_only,
        generate_qa,
        qa_categories,
        mask_persons,
        generate_json,
        generate_video,
        hf_repo_id,
        hf_file_path,
        hf_token,
        captions,
        example_questions,
        auto_generate_captions,
        custom_detector_id,
        detector_api_url,
        detector_api_key,
        vlm_model,
        gemini_api_key,
        vlm_api_url,
        vlm_api_key,
        vlm_model_id,
    )
    
    return {"task_id": task_id, "status": "pending"}

@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks:
        return JSONResponse(status_code=404, content={"error": "Task not found"})
    return tasks[task_id]

@app.get("/api/debug/tasks")
async def debug_tasks():
    return tasks

@app.get("/api/download")
async def download_file(path: str):
    """Serve a file from the output directory as a proper download attachment.
    
    The 'path' parameter must be a relative URL path like '/output/folder/file.csv'.
    This endpoint adds Content-Disposition: attachment so browsers download the file.
    """
    # Strip leading /output/ prefix and resolve to the actual filesystem path
    relative = path.lstrip("/")
    if not relative.startswith("output/"):
        return JSONResponse(status_code=400, content={"error": "Invalid path"})
    
    # Remove 'output/' prefix to get path within output_dir
    file_subpath = relative[len("output/"):]
    file_path = os.path.join(output_dir, file_subpath)
    file_path = os.path.normpath(file_path)
    
    # Security: ensure the resolved path is still inside output_dir
    if not file_path.startswith(os.path.normpath(output_dir)):
        return JSONResponse(status_code=403, content={"error": "Access denied"})
    
    if not os.path.isfile(file_path):
        return JSONResponse(status_code=404, content={"error": "File not found"})
    
    filename = os.path.basename(file_path)
    return FileResponse(
        path=file_path,
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

@app.get("/api/results")
async def get_results(video: str):
    # Use the module-level output_dir (script-relative, not CWD-relative)
    if not os.path.exists(output_dir):
        return JSONResponse(status_code=404, content={"error": "Output directory not found"})
        
    base_name = os.path.splitext(os.path.basename(video))[0]
    
    # Find all results folders for this base name
    folders = []
    for f in os.listdir(output_dir):
        if f.startswith(f"results_{base_name}_") and os.path.isdir(os.path.join(output_dir, f)):
            folders.append(f)
            
    if not folders:
        return JSONResponse(status_code=404, content={"error": "No existing results found"})
        
    # Pick the latest results folder by sorting (since name contains timestamp)
    folders.sort()
    latest_folder = folders[-1]
    run_path = os.path.join(output_dir, latest_folder)
    
    files = os.listdir(run_path)
    video_exts = ('.mp4', '.avi', '.mkv', '.webm', '.mov')
    analyzed_video = next((f for f in files if f.endswith(video_exts) and '_analyzed' in f), None)
    original_video = next((f for f in files if f.endswith(video_exts) and '_original' in f), None)
    any_video = next((f for f in files if f.endswith(video_exts)), None) if not analyzed_video and not original_video else None
    video_file = analyzed_video or original_video or any_video
    csv_file = next((f for f in files if f.endswith('.csv')), None)
    json_file = next((f for f in files if f.endswith('.json') and '_qa_' not in f), None)
    qa_json_files = sorted([f for f in files if f.endswith('.json') and '_qa_' in f])
    
    results = {
        "folder": f"/output/{latest_folder}",
        "video": f"/output/{latest_folder}/{video_file}" if video_file else None,
        "original_video": f"/output/{latest_folder}/{original_video}" if original_video else None,
        "is_original_video": analyzed_video is None and video_file is not None,
        "csv": f"/output/{latest_folder}/{csv_file}" if csv_file else None,
        "json": f"/output/{latest_folder}/{json_file}" if json_file else None,
        "qa_json_files": [f"/output/{latest_folder}/{f}" for f in qa_json_files],
    }
    
    model_info = None
    analysis_settings = None
    object_counts = {}
    captions = None
    if json_file:
        try:
            with open(os.path.join(run_path, json_file), "r", encoding="utf-8") as jf:
                report_data = py_json.load(jf)
                meta = report_data.get("metadata", {})
                m_id = meta.get("model", "unknown")
                dev = meta.get("device", "cpu")
                m_type = "YOLO" if ("yolo" in m_id.lower() or m_id.endswith(".pt")) else "DETR"
                model_info = {
                    "model_type": m_type,
                    "model_name": m_id.split(os.sep)[-1].split('/')[-1],
                    "device": str(dev).upper()
                }
                analysis_settings = _extract_analysis_settings(meta)
                captions = meta.get("captions")
                for obj in report_data.get("objects", []):
                    obj_type = obj.get("object_type", "unknown")
                    object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
        except Exception as e:
            print(f"[-] Error reading metadata from json: {e}")
            
    return {
        "status": "completed",
        "results": results,
        "model_info": model_info,
        "analysis_settings": analysis_settings,
        "object_counts": object_counts,
        "captions": captions,
    }


def _extract_analysis_settings(meta: dict) -> dict:
    """Extract analysis-relevant settings from a metadata dict."""
    return {
        "resolution": meta.get("resolution"),
        "fps": meta.get("fps"),
        "total_frames": meta.get("total_frames"),
        "duration_seconds": meta.get("duration_seconds"),
        "confidence_threshold": meta.get("confidence_threshold"),
        "fps_sample": meta.get("fps_sample"),
    }


def _parse_run_folder(folder_name: str) -> dict | None:
    """Extract metadata from a single results folder and return a summary dict."""
    run_path = os.path.join(output_dir, folder_name)
    if not os.path.isdir(run_path):
        return None

    files = os.listdir(run_path)
    video_exts = ('.mp4', '.avi', '.mkv', '.webm', '.mov')
    analyzed_video = next((f for f in files if f.endswith(video_exts) and '_analyzed' in f), None)
    original_video = next((f for f in files if f.endswith(video_exts) and '_original' in f), None)
    any_video = next((f for f in files if f.endswith(video_exts)), None) if not analyzed_video and not original_video else None
    video_file = analyzed_video or original_video or any_video
    csv_file = next((f for f in files if f.endswith('.csv') and f != 'total_report.csv'), None)
    json_file = next((f for f in files if f.endswith('.json') and '_qa_' not in f), None)
    qa_json_files = sorted([f for f in files if f.endswith('.json') and '_qa_' in f])

    # Extract video name and timestamp from folder name pattern: results_<name>_YYYYMMDD_HHMM
    # May also contain a UUID prefix: results_<uuid>_<name>_YYYYMMDD_HHMM
    parts = folder_name.split('_')
    # Last two parts are date and time
    run_date = None
    if len(parts) >= 3:
        date_part = parts[-2]  # e.g. "20260611"
        time_part = parts[-1]  # e.g. "2007"
        if len(date_part) == 8 and date_part.isdigit() and len(time_part) == 4 and time_part.isdigit():
            run_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]} {time_part[:2]}:{time_part[2:]}"

    # Video name: everything between "results_" prefix and the last _YYYYMMDD_HHMM
    prefix = "results_"
    name_part = folder_name[len(prefix):] if folder_name.startswith(prefix) else folder_name
    # Remove the trailing _YYYYMMDD_HHMM
    if run_date and len(parts) >= 3:
        name_part = '_'.join(name_part.split('_')[:-2])

    # Try to read model info + analysis settings from analysis JSON
    model_info = None
    analysis_settings = None
    object_counts = {}
    captions = None
    if json_file:
        try:
            with open(os.path.join(run_path, json_file), "r", encoding="utf-8") as jf:
                report_data = py_json.load(jf)
                meta = report_data.get("metadata", {})
                m_id = meta.get("model", "unknown")
                dev = meta.get("device", "cpu")
                m_type = "YOLO" if ("yolo" in m_id.lower() or m_id.endswith(".pt")) else "DETR"
                model_info = {
                    "model_type": m_type,
                    "model_name": m_id.split(os.sep)[-1].split('/')[-1],
                    "device": str(dev).upper()
                }
                analysis_settings = _extract_analysis_settings(meta)
                captions = meta.get("captions")
                for obj in report_data.get("objects", []):
                    obj_type = obj.get("object_type", "unknown")
                    object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
        except Exception:
            pass

    return {
        "folder": folder_name,
        "video_name": name_part,
        "run_date": run_date,
        "model_info": model_info,
        "analysis_settings": analysis_settings,
        "object_counts": object_counts,
        "captions": captions,
        "files": {
            "video": f"/output/{folder_name}/{video_file}" if video_file else None,
            "original_video": f"/output/{folder_name}/{original_video}" if original_video else None,
            "is_original_video": analyzed_video is None and video_file is not None,
            "csv": f"/output/{folder_name}/{csv_file}" if csv_file else None,
            "json": f"/output/{folder_name}/{json_file}" if json_file else None,
            "qa_json_files": [f"/output/{folder_name}/{f}" for f in qa_json_files],
        },
    }


@app.get("/api/history")
async def get_history():
    """Return all analysis runs found in the output directory grouped by video_name, newest first."""
    if not os.path.exists(output_dir):
        return []

    folders = [f for f in os.listdir(output_dir)
               if f.startswith("results_") and os.path.isdir(os.path.join(output_dir, f))]

    runs = []
    for folder_name in folders:
        entry = _parse_run_folder(folder_name)
        if entry:
            runs.append(entry)

    import datetime
    def get_sort_key(entry):
        if entry.get("run_date"):
            return entry["run_date"]
        try:
            mtime = os.path.getmtime(os.path.join(output_dir, entry["folder"]))
            return datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return ""

    runs.sort(key=get_sort_key, reverse=True)

    # Group by video_name
    grouped = {}
    for entry in runs:
        vname = entry.get("video_name") or entry.get("folder")
        if vname not in grouped:
            grouped[vname] = {
                "video_name": vname,
                "latest_run_date": entry.get("run_date"),
                "runs": []
            }
        grouped[vname]["runs"].append(entry)
        rdate = entry.get("run_date")
        if rdate:
            ldate = grouped[vname]["latest_run_date"]
            if not ldate or rdate > ldate:
                grouped[vname]["latest_run_date"] = rdate

    # Convert to list and sort by latest run date
    history_list = list(grouped.values())
    history_list.sort(key=lambda x: x["latest_run_date"] or "", reverse=True)
    return history_list



@app.get("/api/history/{folder_name}")
async def get_history_detail(folder_name: str):
    """Return the full detail (files + model info) for a single analysis run."""
    if not folder_name.startswith("results_"):
        return JSONResponse(status_code=400, content={"error": "Invalid folder name"})

    entry = _parse_run_folder(folder_name)
    if not entry:
        return JSONResponse(status_code=404, content={"error": "Run not found"})
    return entry

@app.get("/api/analysis-timeline")
async def get_analysis_timeline(folder: str):
    """Return the per-second detection timeline from the analysis JSON.

    Returns the timeline array plus basic video metadata for charting.
    """
    if not folder.startswith("results_"):
        return JSONResponse(status_code=400, content={"error": "Invalid folder name"})

    run_path = os.path.normpath(os.path.join(output_dir, folder))
    if not run_path.startswith(os.path.normpath(output_dir)):
        return JSONResponse(status_code=403, content={"error": "Access denied"})
    if not os.path.isdir(run_path):
        return JSONResponse(status_code=404, content={"error": "Folder not found"})

    # Find the analysis JSON file (not QA JSONs)
    json_file = None
    for fname in os.listdir(run_path):
        if fname.endswith(".json") and "_qa_" not in fname:
            json_file = os.path.join(run_path, fname)
            break

    if not json_file:
        return JSONResponse(status_code=404, content={"error": "No analysis JSON found"})

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            report_data = py_json.load(f)

        timeline = report_data.get("timeline", [])
        metadata = report_data.get("metadata", {})

        return {
            "timeline": timeline,
            "duration_seconds": metadata.get("duration_seconds"),
            "fps": metadata.get("fps"),
            "total_frames": metadata.get("total_frames"),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/qa-data")
async def get_qa_data(folder: str):
    """Load all QA JSON files for a given results folder.

    Returns a dict keyed by category name, e.g.
    {"counting": [...], "negative": [...], "ambiguity": [...], "day_night": [...]}
    """
    if not folder.startswith("results_"):
        return JSONResponse(status_code=400, content={"error": "Invalid folder name"})

    run_path = os.path.normpath(os.path.join(output_dir, folder))
    if not run_path.startswith(os.path.normpath(output_dir)):
        return JSONResponse(status_code=403, content={"error": "Access denied"})
    if not os.path.isdir(run_path):
        return JSONResponse(status_code=404, content={"error": "Folder not found"})

    qa_data = {}
    for fname in sorted(os.listdir(run_path)):
        if fname.endswith(".json") and "_qa_" in fname:
            match = fname.rsplit("_qa_", 1)
            if len(match) == 2:
                category = match[1].replace(".json", "")
                try:
                    with open(os.path.join(run_path, fname), "r", encoding="utf-8") as f:
                        qa_data[category] = py_json.load(f)
                except Exception as e:
                    qa_data[category] = {"error": str(e)}
    return qa_data


@app.put("/api/qa-data")
async def save_qa_data(request: Request, folder: str, category: str):
    """Save edited QA pairs back to the corresponding JSON file.

    Expects a JSON array body with the full list of QA pairs for the category.
    """
    if not folder.startswith("results_"):
        return JSONResponse(status_code=400, content={"error": "Invalid folder name"})
    # Sanitise category to prevent path traversal
    safe_category = category.replace("/", "").replace("\\", "").replace("..", "")
    if not safe_category:
        return JSONResponse(status_code=400, content={"error": "Invalid category"})

    run_path = os.path.normpath(os.path.join(output_dir, folder))
    if not run_path.startswith(os.path.normpath(output_dir)):
        return JSONResponse(status_code=403, content={"error": "Access denied"})
    if not os.path.isdir(run_path):
        return JSONResponse(status_code=404, content={"error": "Folder not found"})

    # Find the matching QA file
    target_file = None
    for fname in os.listdir(run_path):
        if fname.endswith(".json") and f"_qa_{safe_category}.json" in fname:
            target_file = os.path.join(run_path, fname)
            break

    if not target_file:
        return JSONResponse(status_code=404, content={"error": f"QA file for category '{safe_category}' not found"})

    try:
        body = await request.json()
        if not isinstance(body, list):
            return JSONResponse(status_code=400, content={"error": "Body must be a JSON array"})
        with open(target_file, "w", encoding="utf-8") as f:
            py_json.dump(body, f, indent=2, ensure_ascii=False)
        return {"status": "saved", "category": safe_category, "count": len(body)}
    except py_json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/qa-regenerate")
async def regenerate_qa(request: Request):
    """Regenerate QA JSON files for a given results folder using updated captions and custom questions."""
    try:
        body = await request.json()
        folder = body.get("folder", "")
        captions = body.get("captions")
        example_questions = body.get("example_questions")
        qa_categories = body.get("qa_categories", "")
        
        # Parse VLM configuration details from request body
        vlm_model = body.get("vlm_model", "none")
        gemini_api_key = body.get("gemini_api_key")
        vlm_api_url = body.get("vlm_api_url")
        vlm_api_key = body.get("vlm_api_key")
        vlm_model_id = body.get("vlm_model_id")
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    if not folder:
        return JSONResponse(status_code=400, content={"error": "Folder parameter is required"})

    if not folder.startswith("results_"):
        return JSONResponse(status_code=400, content={"error": "Invalid folder name"})

    # Sanitise folder name to prevent path traversal
    safe_folder = os.path.basename(folder)
    run_path = os.path.normpath(os.path.join(output_dir, safe_folder))
    if not run_path.startswith(os.path.normpath(output_dir)):
        return JSONResponse(status_code=403, content={"error": "Access denied"})
    if not os.path.isdir(run_path):
        return JSONResponse(status_code=404, content={"error": "Folder not found"})

    # Find the analysis JSON file
    json_file = None
    for fname in os.listdir(run_path):
        if fname.endswith(".json") and "_qa_" not in fname:
            json_file = os.path.join(run_path, fname)
            break

    if not json_file:
        return JSONResponse(status_code=404, content={"error": "No analysis JSON found in output directory"})

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            report_data = py_json.load(f)

        meta = report_data.get("metadata", {})
        objects = report_data.get("objects", [])

        filename = meta.get("video_file", "video.mp4")
        duration = meta.get("duration_seconds", 0.0)

        # Build processed_frames from objects' bbox_observations
        from core.qa_generator import QAGenerator, parse_timestamp_to_seconds
        from utils.report_generator import save_qa_report

        frames_by_timestamp = {}
        for obj in objects:
            obj_type = obj.get("object_type", "unknown")
            for obs in obj.get("bbox_observations", []):
                ts_str = obs.get("timestamp", "00:00:00")
                ts_sec = parse_timestamp_to_seconds(ts_str)

                if ts_sec not in frames_by_timestamp:
                    frames_by_timestamp[ts_sec] = []

                frames_by_timestamp[ts_sec].append({
                    "label": obj_type,
                    "box": [obs.get("x1", 0.0), obs.get("y1", 0.0), obs.get("x2", 0.0), obs.get("y2", 0.0)],
                    "score": obs.get("confidence", 1.0)
                })

        # Convert to sorted processed_frames
        processed_frames = []
        for idx, ts in enumerate(sorted(frames_by_timestamp.keys())):
            processed_frames.append({
                "frame_idx": idx,
                "timestamp": ts,
                "detections": frames_by_timestamp[ts],
                "blur_var": 200.0,  # default
                "brightness": 128.0  # default
            })

        # Reconstruct tracked_objects from report_data
        from core.tracking import TrackedObject, BBoxObservation
        from utils.time_utils import timestamp_to_seconds
        tracked_objects = {}
        for obj in objects:
            try:
                obj_id = int(obj.get("object_id", 0))
            except ValueError:
                continue
            obj_type = obj.get("object_type", "unknown")
            try:
                first_seen = timestamp_to_seconds(obj.get("first_time_seen", "00:00:00:000"))
            except Exception:
                first_seen = 0.0
            try:
                screen_time = timestamp_to_seconds(obj.get("total_screen_time", "00:00:00:000"))
            except Exception:
                screen_time = 0.0
                
            obs_list = []
            for obs in obj.get("bbox_observations", []):
                obs_list.append(
                    BBoxObservation(
                        timestamp=obs.get("timestamp", "00:00:00:000"),
                        x1=float(obs.get("x1", 0.0)),
                        y1=float(obs.get("y1", 0.0)),
                        x2=float(obs.get("x2", 0.0)),
                        y2=float(obs.get("y2", 0.0)),
                        confidence=float(obs.get("confidence", 1.0))
                    )
                )
            tracked_objects[obj_id] = TrackedObject(
                object_type=obj_type,
                first_time_seen_seconds=first_seen,
                screen_time_seconds=screen_time,
                bbox_observations=obs_list
            )
            
        # Find local video path in run_path
        files = os.listdir(run_path)
        video_exts = ('.mp4', '.avi', '.mkv', '.webm', '.mov', '.MOV')
        analyzed_video = next((f for f in files if f.endswith(video_exts) and '_analyzed' in f), None)
        original_video = next((f for f in files if f.endswith(video_exts) and '_original' in f), None)
        any_video = next((f for f in files if f.endswith(video_exts)), None) if not analyzed_video and not original_video else None
        video_file = analyzed_video or original_video or any_video
        video_path = os.path.join(run_path, video_file) if video_file else None

        qa_cats = [c.strip() for c in qa_categories.split(',')] if qa_categories else ["counting", "negative", "ambiguity", "day_night"]
        if captions or example_questions:
            if "user_queries" not in qa_cats:
                qa_cats.append("user_queries")

        from core.verifier import verify_all_qa
        qa_generator = QAGenerator(
            filename,
            processed_frames,
            duration,
            tracked_objects=tracked_objects,
            video_path=video_path,
            qa_categories=qa_cats,
            captions=captions,
            example_questions=example_questions,
            gemini_api_key=gemini_api_key,
            custom_vlm_url=vlm_api_url,
            custom_vlm_key=vlm_api_key,
            custom_vlm_model_id=vlm_model_id,
        )
        qa_by_category = qa_generator.generate_qa_pairs()
        qa_by_category = verify_all_qa(qa_by_category, tracked_objects)

        base_name = os.path.splitext(filename)[0]

        # Clean existing QA files
        for fname in os.listdir(run_path):
            if fname.endswith(".json") and "_qa_" in fname:
                try:
                    os.remove(os.path.join(run_path, fname))
                except Exception:
                    pass

        out_qa_paths = save_qa_report(run_path, base_name, qa_by_category)

        return {
            "status": "success",
            "qa_json_files": [f"/output/{safe_folder}/{os.path.basename(p)}" for p in out_qa_paths]
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"QA regeneration failed: {str(e)}"})


@app.get("/api/stream-video")
async def stream_video(request: Request, path: str):
    """Stream a video file with Range header support for browser playback.

    Browsers need Range requests to seek within videos.  The default
    StaticFiles mount does not always handle this reliably for large files
    produced by OpenCV, so this dedicated endpoint ensures correct
    Content-Type and partial-content responses.

    Supports paths under both /output/ and /input/ directories.
    """
    relative = path.lstrip("/")

    # Resolve the file path — support both output/ and input/ directories
    if relative.startswith("output/"):
        file_subpath = relative[len("output/"):]
        file_path = os.path.normpath(os.path.join(output_dir, file_subpath))
        if not file_path.startswith(os.path.normpath(output_dir)):
            return JSONResponse(status_code=403, content={"error": "Access denied"})
    elif relative.startswith("input/"):
        file_subpath = relative[len("input/"):]
        file_path = os.path.normpath(os.path.join(input_dir, file_subpath))
        if not file_path.startswith(os.path.normpath(input_dir)):
            return JSONResponse(status_code=403, content={"error": "Access denied"})
    else:
        return JSONResponse(status_code=400, content={"error": "Invalid path"})

    if not os.path.isfile(file_path):
        return JSONResponse(status_code=404, content={"error": "File not found"})

    # Auto-detect MIME type based on extension
    ext = os.path.splitext(file_path)[1].lower()
    mime_types = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
    }
    media_type = mime_types.get(ext, "video/mp4")

    file_size = os.path.getsize(file_path)
    range_header = request.headers.get("range")

    if range_header:
        # Parse "bytes=start-end" or "bytes=-num" or "bytes=start-"
        range_spec = range_header.replace("bytes=", "").strip()
        if range_spec.startswith("-"):
            num_bytes = int(range_spec[1:])
            start = max(0, file_size - num_bytes)
            end = file_size - 1
        else:
            parts = range_spec.split("-")
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if (len(parts) > 1 and parts[1]) else file_size - 1
            
        end = min(end, file_size - 1)
        start = min(start, end)
        length = end - start + 1

        def iter_file():
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return StreamingResponse(
            iter_file(),
            status_code=206,
            media_type=media_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length),
            },
        )
    else:
        def iter_full():
            with open(file_path, "rb") as f:
                while chunk := f.read(65536):
                    yield chunk

        return StreamingResponse(
            iter_full(),
            media_type=media_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
            },
        )



def perform_track_fusion(runs, cls):
    intervals = []
    for run in runs:
        json_file_path = run.get("files", {}).get("json")
        if not json_file_path:
            continue
        relative = json_file_path.lstrip("/")
        if relative.startswith("output/"):
            file_subpath = relative[len("output/"):]
            abs_path = os.path.join(output_dir, file_subpath)
            if os.path.isfile(abs_path):
                try:
                    with open(abs_path, "r", encoding="utf-8") as f:
                        data = py_json.load(f)
                        for obj in data.get("objects", []):
                            if obj.get("object_type") == cls:
                                from utils.time_utils import timestamp_to_seconds
                                start_t = timestamp_to_seconds(obj.get("first_time_seen", "00:00:00"))
                                duration_t = timestamp_to_seconds(obj.get("total_screen_time", "00:00:00"))
                                intervals.append((start_t, start_t + duration_t))
                except Exception:
                    pass
    if not intervals:
        return 0
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        prev = merged[-1]
        if current[0] <= prev[1] + 1.0:
            merged[-1] = (prev[0], max(prev[1], current[1]))
        else:
            merged.append(current)
    return len(merged)

@app.get("/api/video-comparison")
async def video_comparison(video_name: str, consensus_method: str = "average"):
    if not os.path.exists(output_dir):
        return JSONResponse(status_code=404, content={"error": "Output directory not found"})
        
    folders = [f for f in os.listdir(output_dir)
               if f.startswith(f"results_{video_name}_") and os.path.isdir(os.path.join(output_dir, f))]
               
    if not folders:
        return JSONResponse(status_code=404, content={"error": f"No analysis runs found for video: {video_name}"})
        
    runs_data = []
    for f in folders:
        entry = _parse_run_folder(f)
        if entry:
            runs_data.append(entry)
            
    runs_data.sort(key=lambda x: x.get("run_date") or "", reverse=True)
    
    all_classes = set()
    for run in runs_data:
        all_classes.update(run.get("object_counts", {}).keys())
        
    consensus_counts = {}
    for cls in all_classes:
        counts = []
        for run in runs_data:
            counts.append(run.get("object_counts", {}).get(cls, 0))
            
        if consensus_method == "average":
            consensus_counts[cls] = round(sum(counts) / len(runs_data)) if runs_data else 0
        elif consensus_method == "median":
            sorted_c = sorted(counts)
            mid = len(sorted_c) // 2
            if len(sorted_c) % 2 == 0:
                consensus_counts[cls] = round((sorted_c[mid - 1] + sorted_c[mid]) / 2)
            else:
                consensus_counts[cls] = sorted_c[mid]
        elif consensus_method == "maximum":
            consensus_counts[cls] = max(counts)
        elif consensus_method == "minimum":
            consensus_counts[cls] = min(counts)
        elif consensus_method == "track_fusion":
            consensus_counts[cls] = perform_track_fusion(runs_data, cls)
            
    verified_file = os.path.join(output_dir, f"verified_{video_name}", "verified_report.json")
    verified_data = None
    if os.path.isfile(verified_file):
        try:
            with open(verified_file, "r", encoding="utf-8") as vf:
                verified_data = py_json.load(vf)
        except Exception:
            pass
            
    return {
        "video_name": video_name,
        "runs": runs_data,
        "consensus_counts": consensus_counts,
        "verified_data": verified_data
    }

@app.post("/api/save-verified")
async def save_verified(request: Request):
    try:
        body = await request.json()
        video_name = body.get("video_name")
        if not video_name:
            return JSONResponse(status_code=400, content={"error": "video_name is required"})
            
        verified_dir = os.path.join(output_dir, f"verified_{video_name}")
        os.makedirs(verified_dir, exist_ok=True)
        
        verified_file = os.path.join(verified_dir, "verified_report.json")
        with open(verified_file, "w", encoding="utf-8") as f:
            py_json.dump({
                "video_name": video_name,
                "verified_counts": body.get("verified_counts", {}),
                "verified_qa": body.get("verified_qa", []),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())
            }, f, indent=2, ensure_ascii=False)
            
        return {"status": "success", "message": f"Verified report saved for {video_name}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web:app", host="0.0.0.0", port=8000, reload=True)
