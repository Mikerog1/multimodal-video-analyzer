import os
import sys
import uuid
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
):
    try:
        tasks[task_id]["status"] = "loading_model"
        tasks[task_id]["progress"] = 0
        
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
        else:
            from models.detr_detector import DetrDetector
            if not model_id:
                model_id = "facebook/detr-resnet-50"
            detector = DetrDetector(model_id, device, confidence)
            
        processor = VideoProcessor(detector)
        
        # Store model info in task dict
        tasks[task_id]["model_info"] = {
            "model_type": "YOLO" if model_type == "yolo" else "DETR",
            "model_name": detector.model_id.split(os.sep)[-1].split('/')[-1] if hasattr(detector, "model_id") else str(model_id),
            "device": str(detector.device).upper() if hasattr(detector, "device") else str(device).upper()
        }
        
        tasks[task_id]["status"] = "analyzing"
        
        # Override print to capture some output if needed, but for now we just run it
        # Note: We need a way to know the exact output directory since it's timestamped.
        # Actually, VideoProcessor creates a timestamped folder inside output_dir.
        # Let's inspect the output directory before and after to find the new folder,
        # or we could patch the processor to return the paths.
        
        existing_folders = set(os.listdir(output_dir)) if os.path.exists(output_dir) else set()
        
        def update_progress(current, total):
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
        )
        
        current_folders = set(os.listdir(output_dir))
        new_folders = current_folders - existing_folders
        
        if new_folders:
            run_folder = list(new_folders)[0]
            run_path = os.path.join(output_dir, run_folder)
            
            # Find the output files
            files = os.listdir(run_path)
            video_exts = ('.mp4', '.avi', '.mkv', '.webm', '.mov')
            analyzed_video = next((f for f in files if f.endswith(video_exts) and '_analyzed' in f), None)
            original_video = next((f for f in files if f.endswith(video_exts) and '_original' in f), None)
            # Fallback: any video file
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
            if json_file:
                try:
                    with open(os.path.join(run_path, json_file), "r", encoding="utf-8") as jf:
                        report_data = py_json.load(jf)
                        meta = report_data.get("metadata", {})
                        analysis_settings = _extract_analysis_settings(meta)
                except Exception:
                    pass
            tasks[task_id]["analysis_settings"] = analysis_settings

            tasks[task_id]["status"] = "completed"
        else:
            tasks[task_id]["status"] = "error"
            tasks[task_id]["error"] = "No output directory was created."
            
    except Exception as e:
        tasks[task_id]["status"] = "error"
        tasks[task_id]["error"] = str(e)
    finally:
        # Clean up temporary uploaded input video file to avoid folder bloating
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
            upload_dir = os.path.dirname(video_path)
            if os.path.exists(upload_dir) and not os.listdir(upload_dir):
                os.rmdir(upload_dir)
        except Exception as cleanup_err:
            print(f"[-] Error cleaning up temporary file: {cleanup_err}")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/api/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    model_type: str = Form("detr"),
    model_id: str = Form(""),
    device: str = Form("auto"),
    codec: str = Form("avc1"),
    confidence: float = Form(0.7),
    fps_sample: float = Form(1.0),
    resize_factor: float = Form(1.0),
    save_sampled_only: bool = Form(False),
    generate_qa: bool = Form(True),
    generate_json: bool = Form(True),
    qa_categories: str = Form(""),
    remove_audio: bool = Form(False),
    mask_persons: bool = Form(False),
    generate_video: bool = Form(True),
):
    task_id = str(uuid.uuid4())
    
    # Save the uploaded file in a unique folder to prevent name collisions
    # while preserving the original filename for cleaner output results.
    upload_dir = os.path.join(input_dir, task_id)
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    tasks[task_id] = {
        "status": "pending",
        "progress": 0,
        "filename": file.filename,
        "results": None,
        "model_info": None,
        "analysis_settings": None,
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
        "is_original_video": analyzed_video is None and video_file is not None,
        "csv": f"/output/{latest_folder}/{csv_file}" if csv_file else None,
        "json": f"/output/{latest_folder}/{json_file}" if json_file else None,
        "qa_json_files": [f"/output/{latest_folder}/{f}" for f in qa_json_files],
    }
    
    model_info = None
    analysis_settings = None
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
        except Exception as e:
            print(f"[-] Error reading metadata from json: {e}")
            
    return {"status": "completed", "results": results, "model_info": model_info, "analysis_settings": analysis_settings}


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
        except Exception:
            pass

    return {
        "folder": folder_name,
        "video_name": name_part,
        "run_date": run_date,
        "model_info": model_info,
        "analysis_settings": analysis_settings,
        "files": {
            "video": f"/output/{folder_name}/{video_file}" if video_file else None,
            "is_original_video": analyzed_video is None and video_file is not None,
            "csv": f"/output/{folder_name}/{csv_file}" if csv_file else None,
            "json": f"/output/{folder_name}/{json_file}" if json_file else None,
            "qa_json_files": [f"/output/{folder_name}/{f}" for f in qa_json_files],
        },
    }


@app.get("/api/history")
async def get_history():
    """Return a list of all analysis runs found in the output directory, newest first."""
    if not os.path.exists(output_dir):
        return []

    folders = sorted(
        (f for f in os.listdir(output_dir)
         if f.startswith("results_") and os.path.isdir(os.path.join(output_dir, f))),
        reverse=True,
    )

    runs = []
    for folder_name in folders:
        entry = _parse_run_folder(folder_name)
        if entry:
            runs.append(entry)
    return runs


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web:app", host="0.0.0.0", port=8000, reload=True)
