import os
import sys
import uuid
import asyncio
import concurrent.futures
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

# Add core directory to PATH on Windows to allow OpenCV to locate the openh264 DLL
if os.name == 'nt':
    core_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'core'))
    os.environ['PATH'] = core_dir + os.pathsep + os.environ.get('PATH', '')

import torch
from core.video_processor import VideoProcessor

app = FastAPI(title="Multimodal Video Analyzer Web API")

# Ensure necessary directories exist using absolute paths to prevent resolution discrepancies
input_dir = os.path.abspath("input")
output_dir = os.path.abspath("output")
static_dir = os.path.abspath("static")
models_dir = os.path.abspath("models")

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
    generate_video: bool,
    generate_csv: bool,
    generate_json: bool,
    generate_qa: bool,
    qa_categories: str
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
            generate_video=generate_video,
            generate_csv=generate_csv,
            generate_json=generate_json,
            generate_qa=generate_qa,
            qa_categories=qa_cats,
            progress_callback=update_progress
        )
        
        current_folders = set(os.listdir(output_dir))
        new_folders = current_folders - existing_folders
        
        if new_folders:
            run_folder = list(new_folders)[0]
            run_path = os.path.join(output_dir, run_folder)
            
            # Find the output files
            files = os.listdir(run_path)
            video_file = next((f for f in files if f.endswith('.mp4')), None)
            csv_file = next((f for f in files if f.endswith('.csv')), None)
            json_file = next((f for f in files if f.endswith('_analysis.json')), None)
            qa_json_file = next((f for f in files if f.endswith('_qa_pairs.json')), None)
            
            tasks[task_id]["results"] = {
                "folder": f"/output/{run_folder}",
                "video": f"/output/{run_folder}/{video_file}" if video_file else None,
                "csv": f"/output/{run_folder}/{csv_file}" if csv_file else None,
                "json": f"/output/{run_folder}/{json_file}" if json_file else None,
                "qa_json": f"/output/{run_folder}/{qa_json_file}" if qa_json_file else None,
            }
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
    codec: str = Form("mp4v"),
    confidence: float = Form(0.7),
    fps_sample: float = Form(1.0),
    resize_factor: float = Form(1.0),
    save_sampled_only: bool = Form(False),
    generate_video: bool = Form(True),
    generate_csv: bool = Form(True),
    generate_json: bool = Form(True),
    generate_qa: bool = Form(True),
    qa_categories: str = Form("")
):
    task_id = str(uuid.uuid4())
    
    # Save the uploaded file in a unique folder to prevent name collisions
    # while preserving the original filename for cleaner output results.
    upload_dir = os.path.join(os.path.abspath("input"), task_id)
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    tasks[task_id] = {
        "status": "pending",
        "progress": 0,
        "filename": file.filename,
        "results": None,
        "error": None
    }
    
    # Run analysis in a thread to avoid blocking the event loop
    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        executor, 
        run_analysis, 
        task_id, 
        file_path, 
        os.path.abspath("output"), 
        model_type,
        model_id,
        device,
        codec,
        confidence, 
        fps_sample, 
        resize_factor, 
        save_sampled_only,
        generate_video,
        generate_csv,
        generate_json,
        generate_qa,
        qa_categories
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

@app.get("/api/results")
async def get_results(video: str):
    output_dir = os.path.abspath("output")
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
    video_file = next((f for f in files if f.endswith('.mp4')), None)
    csv_file = next((f for f in files if f.endswith('.csv')), None)
    json_file = next((f for f in files if f.endswith('_analysis.json')), None)
    qa_json_file = next((f for f in files if f.endswith('_qa_pairs.json')), None)
    
    results = {
        "folder": f"/output/{latest_folder}",
        "video": f"/output/{latest_folder}/{video_file}" if video_file else None,
        "csv": f"/output/{latest_folder}/{csv_file}" if csv_file else None,
        "json": f"/output/{latest_folder}/{json_file}" if json_file else None,
        "qa_json": f"/output/{latest_folder}/{qa_json_file}" if qa_json_file else None,
    }
    
    return {"status": "completed", "results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web:app", host="0.0.0.0", port=8000, reload=True)
