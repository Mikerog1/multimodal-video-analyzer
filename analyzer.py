import os
import sys
import time
import argparse
import csv
import json
from datetime import timedelta
import numpy as np
import cv2
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection

def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Video Analyzer using DETR")
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to the input video file or a folder of videos"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="output", 
        help="Directory to save analyzed video and reports"
    )
    parser.add_argument(
        "--confidence", 
        type=float, 
        default=0.7, 
        help="Minimum confidence threshold for detections (0.0 to 1.0)"
    )
    parser.add_argument(
        "--fps-sample", 
        type=float, 
        default=1.0, 
        help="Frames per second to sample and run AI model inference (e.g. 1.0 = once per second, 0 = analyze all frames)"
    )
    parser.add_argument(
        "--model-id", 
        type=str, 
        default="facebook/detr-resnet-50", 
        help="Hugging Face model ID for object detection"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto", 
        help="Device to run inference on ('cuda', 'cpu', or 'auto')"
    )
    return parser.parse_args()

def draw_futuristic_box(img, x1, y1, x2, y2, color, label, score, line_thickness=2):
    """Draws a premium styled bounding box with corner brackets and a label tag."""
    # Ensure coordinates are within image boundaries
    h, w, _ = img.shape
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    
    # Draw a thin bounding box line
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)
    
    # Draw thicker corner brackets for a futuristic/HUD look
    corner_len = min(20, int((x2 - x1) * 0.2), int((y2 - y1) * 0.2))
    # Top-Left corner
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, line_thickness, lineType=cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, line_thickness, lineType=cv2.LINE_AA)
    # Top-Right corner
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, line_thickness, lineType=cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, line_thickness, lineType=cv2.LINE_AA)
    # Bottom-Left corner
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, line_thickness, lineType=cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, line_thickness, lineType=cv2.LINE_AA)
    # Bottom-Right corner
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, line_thickness, lineType=cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, line_thickness, lineType=cv2.LINE_AA)
    
    # Draw a tag background above or inside the box
    tag_text = f"{label} {int(score * 100)}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    
    (tw, th), baseline = cv2.getTextSize(tag_text, font, font_scale, font_thickness)
    tag_y = y1 - 5 if y1 - 5 - th > 0 else y1 + th + 5
    tag_x = x1
    
    # Tag background
    cv2.rectangle(img, (tag_x, tag_y - th - 3), (tag_x + tw + 6, tag_y + baseline), color, -1)
    # Tag text (white or dark text depending on color contrast, using black for maximum contrast on bright tags)
    cv2.putText(img, tag_text, (tag_x + 3, tag_y - 2), font, font_scale, (255, 255, 255) if color == (255, 0, 0) else (0, 0, 0), font_thickness, lineType=cv2.LINE_AA)

def draw_hud(img, counts, elapsed_time, total_time, model_name, device, frame_idx, total_frames):
    """Draws a premium top-bar HUD containing current counts, time, and system details."""
    h, w, _ = img.shape
    
    # HUD Bar dimensions
    hud_h = 55
    hud_bg = img[0:hud_h, 0:w].copy()
    
    # Apply semi-transparent black overlay
    overlay = np.zeros_like(hud_bg)
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (15, 20, 30), -1)
    hud_bg = cv2.addWeighted(hud_bg, 0.4, overlay, 0.6, 0)
    
    # Write overlay back
    img[0:hud_h, 0:w] = hud_bg
    
    # Draw separation line
    cv2.line(img, (0, hud_h), (w, hud_h), (255, 255, 255), 1, lineType=cv2.LINE_AA)
    
    # Text styles
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # 1. Live Detections on the Left
    text_y = 33
    start_x = 20
    
    # People (Cyan)
    cv2.circle(img, (start_x, text_y - 6), 5, (255, 255, 0), -1, lineType=cv2.LINE_AA)
    cv2.putText(img, f"People: {counts['person']}", (start_x + 12, text_y), font, 0.5, (220, 240, 255), 1, lineType=cv2.LINE_AA)
    
    # Cars (Green)
    start_x += 130
    cv2.circle(img, (start_x, text_y - 6), 5, (75, 220, 75), -1, lineType=cv2.LINE_AA)
    cv2.putText(img, f"Cars: {counts['car']}", (start_x + 12, text_y), font, 0.5, (220, 240, 255), 1, lineType=cv2.LINE_AA)
    
    # Dogs (Orange)
    start_x += 110
    cv2.circle(img, (start_x, text_y - 6), 5, (0, 165, 255), -1, lineType=cv2.LINE_AA)
    cv2.putText(img, f"Dogs: {counts['dog']}", (start_x + 12, text_y), font, 0.5, (220, 240, 255), 1, lineType=cv2.LINE_AA)
    
    # 2. Time & Progress in the Center/Right
    time_str = f"{str(timedelta(seconds=int(elapsed_time)))} / {str(timedelta(seconds=int(total_time)))}"
    (tw, _), _ = cv2.getTextSize(time_str, font, 0.5, 1)
    time_x = (w - tw) // 2
    cv2.putText(img, time_str, (time_x, text_y), font, 0.5, (200, 200, 200), 1, lineType=cv2.LINE_AA)
    
    # 3. Model & Device info on the Right
    info_str = f"Model: {model_name} | Device: {device.upper()}"
    (iw, _), _ = cv2.getTextSize(info_str, font, 0.4, 1)
    cv2.putText(img, info_str, (w - iw - 20, text_y), font, 0.45, (140, 150, 160), 1, lineType=cv2.LINE_AA)

def analyze_video(video_path, output_dir, confidence_threshold, fps_sample, model_id, device_name):
    # Determine execution device
    if device_name == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_name
        
    print(f"\n[+] Target Execution Device: {device.upper()}")
    
    # Load Model and Preprocessor
    print(f"[+] Loading {model_id} from Hugging Face...")
    t0 = time.time()
    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForObjectDetection.from_pretrained(model_id)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"[-] Error loading model {model_id}: {e}")
        sys.exit(1)
    print(f"[+] Model loaded in {time.time() - t0:.2f} seconds.")
    
    # Open Video File
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[-] Error: Could not open video file: {video_path}")
        return
        
    # Get Video Properties
    filename = os.path.basename(video_path)
    base_name = os.path.splitext(filename)[0]
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / video_fps if video_fps > 0 else 0
    
    print(f"[+] Video Properties: {width}x{height} | {video_fps:.2f} FPS | {total_frames} frames | {duration:.2f} seconds")
    
    # Setup Output Paths
    os.makedirs(output_dir, exist_ok=True)
    out_video_path = os.path.join(output_dir, f"{base_name}_analyzed.mp4")
    out_csv_path = os.path.join(output_dir, f"{base_name}_analysis.csv")
    out_json_path = os.path.join(output_dir, f"{base_name}_analysis.json")
    
    # Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(out_video_path, fourcc, video_fps, (width, height))
    
    # Calculate sampling intervals
    # If fps_sample is 0, we process every frame. Otherwise we calculate frame steps.
    if fps_sample > 0:
        sample_step = int(max(1, round(video_fps / fps_sample)))
        print(f"[+] Sampling Mode: Running AI inference every {sample_step} frames (~{fps_sample} frame(s) per second)")
    else:
        sample_step = 1
        print("[+] Sampling Mode: Running AI inference on EVERY frame")
        
    # Tracking variables
    timeline_data = [] # Stores second-by-second analytics
    current_sec_counts = {"person": 0, "car": 0, "dog": 0}
    last_detected_boxes = [] # Persisted detections for interpolation
    
    # Color mappings for classes
    # BGR format: Cyan for person, Emerald for car, Orange for dog
    class_colors = {
        "person": (255, 255, 0), # Cyan
        "car": (75, 220, 75),    # Emerald Green
        "dog": (0, 165, 255)     # Orange
    }
    
    # Summary metrics
    max_counts = {"person": 0, "car": 0, "dog": 0}
    sum_counts = {"person": 0, "car": 0, "dog": 0}
    seconds_tracked = 0
    
    print(f"\n[+] Analyzing video frames...")
    
    # Wrap in tqdm progress bar
    for frame_idx in tqdm(range(total_frames), desc="Analyzing Frames"):
        ret, frame = cap.read()
        if not ret:
            break
            
        current_second = int(frame_idx / video_fps)
        
        # Check if we should run inference on this frame
        should_infer = (frame_idx % sample_step == 0)
        
        if should_infer:
            # Prepare image for PyTorch
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            inputs = processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Post-process bounding boxes
            results = processor.post_process_object_detection(
                outputs, 
                threshold=confidence_threshold, 
                target_sizes=[(height, width)]
            )[0]
            
            # Extract and filter target detections
            last_detected_boxes = []
            frame_counts = {"person": 0, "car": 0, "dog": 0}
            
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                label_name = model.config.id2label[label.item()].lower()
                
                # Check if detected object is in our targets
                if label_name in ["person", "car", "dog"]:
                    score_val = score.item()
                    box_coords = box.tolist() # [xmin, ymin, xmax, ymax]
                    
                    last_detected_boxes.append({
                        "label": label_name,
                        "score": score_val,
                        "box": box_coords
                    })
                    frame_counts[label_name] += 1
            
            # Aggregate counts for second-by-second analytics
            current_sec_counts = frame_counts
            
            # Update summary metrics
            for k in max_counts:
                max_counts[k] = max(max_counts[k], frame_counts[k])
                
        # Draw bounding boxes (persisted from last inference frame)
        annotated_frame = frame.copy()
        active_counts = {"person": 0, "car": 0, "dog": 0}
        
        for det in last_detected_boxes:
            lbl = det["label"]
            draw_futuristic_box(
                annotated_frame, 
                det["box"][0], 
                det["box"][1], 
                det["box"][2], 
                det["box"][3], 
                class_colors[lbl], 
                lbl.capitalize(), 
                det["score"]
            )
            active_counts[lbl] += 1
            
        # If we entered a new second or reached the end, log the second data
        # We record the counts at the boundaries of seconds
        elapsed_sec = frame_idx / video_fps
        if frame_idx % int(video_fps) == 0 or frame_idx == total_frames - 1:
            # Avoid duplicate logs for the same second
            sec_rounded = int(elapsed_sec)
            if not any(d["second"] == sec_rounded for d in timeline_data):
                timeline_data.append({
                    "second": sec_rounded,
                    "timestamp": str(timedelta(seconds=sec_rounded)),
                    "people": active_counts["person"],
                    "cars": active_counts["car"],
                    "dogs": active_counts["dog"]
                })
                # Add to cumulative sums for average calculation
                for k in sum_counts:
                    sum_counts[k] += active_counts[k]
                seconds_tracked += 1
                
        # Add HUD overlay
        draw_hud(
            annotated_frame, 
            active_counts, 
            elapsed_sec, 
            duration, 
            model_id.split('/')[-1], 
            device, 
            frame_idx, 
            total_frames
        )
        
        # Write to Output Video
        out_writer.write(annotated_frame)
        
    # Release resources
    cap.release()
    out_writer.release()
    
    # Save CSV Report
    with open(out_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Timestamp", "Second", "People Count", "Car Count", "Dog Count"])
        for entry in timeline_data:
            writer.writerow([
                entry["timestamp"],
                entry["second"],
                entry["people"],
                entry["cars"],
                entry["dogs"]
            ])
            
    # Calculate Averages
    avg_counts = {
        "person": round(sum_counts["person"] / seconds_tracked, 2) if seconds_tracked > 0 else 0,
        "car": round(sum_counts["car"] / seconds_tracked, 2) if seconds_tracked > 0 else 0,
        "dog": round(sum_counts["dog"] / seconds_tracked, 2) if seconds_tracked > 0 else 0,
    }
    
    # Save JSON Report
    report_summary = {
        "metadata": {
            "video_file": filename,
            "resolution": f"{width}x{height}",
            "fps": round(video_fps, 2),
            "total_frames": total_frames,
            "duration_seconds": round(duration, 2),
            "model": model_id,
            "device": device,
            "confidence_threshold": confidence_threshold,
            "fps_sample": fps_sample,
            "analysis_time_utc": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())
        },
        "summary": {
            "max_people": max_counts["person"],
            "avg_people": avg_counts["person"],
            "max_cars": max_counts["car"],
            "avg_cars": avg_counts["car"],
            "max_dogs": max_counts["dog"],
            "avg_dogs": avg_counts["dog"]
        },
        "timeline": timeline_data
    }
    
    with open(out_json_path, mode="w") as json_file:
        json.dump(report_summary, json_file, indent=2)
        
    # Draw Print Summary Table
    print_cli_summary(filename, duration, max_counts, avg_counts, out_video_path, out_csv_path)

def print_cli_summary(filename, duration, max_counts, avg_counts, out_video, out_csv):
    """Prints a beautiful summary table of detections to the console."""
    border = "=" * 60
    thin_line = "-" * 60
    print(f"\n{border}")
    print(f"  ANALYSIS COMPLETE: {filename}")
    print(f"  Duration: {str(timedelta(seconds=int(duration)))} ({duration:.1f}s)")
    print(f"{border}")
    print(f"  Class      | Max Count  | Average Count")
    print(f"  {thin_line[0:11]}+{thin_line[0:12]}+{thin_line[0:15]}")
    print(f"  People     | {max_counts['person']:<10} | {avg_counts['person']:<13}")
    print(f"  Cars       | {max_counts['car']:<10} | {avg_counts['car']:<13}")
    print(f"  Dogs       | {max_counts['dog']:<10} | {avg_counts['dog']:<13}")
    print(f"{border}")
    print(f"  Outputs Saved:")
    print(f"  - Annotated Video : {os.path.abspath(out_video)}")
    print(f"  - CSV Count Log   : {os.path.abspath(out_csv)}")
    print(f"{border}\n")

if __name__ == "__main__":
    args = parse_args()
    
    # Validate input exists
    if not os.path.exists(args.input):
        print(f"[-] Error: Input path does not exist: {args.input}")
        sys.exit(1)
        
    # If input is a folder, search for videos inside it
    if os.path.isdir(args.input):
        video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")
        video_files = [
            os.path.join(args.input, f) for f in os.listdir(args.input)
            if f.lower().endswith(video_extensions)
        ]
        
        if not video_files:
            print(f"[-] Error: No video files found in folder: {args.input}")
            sys.exit(1)
            
        print(f"[+] Found {len(video_files)} video(s) to process in folder: {args.input}")
        for idx, vid in enumerate(video_files):
            print(f"\n[+] Processing video {idx+1}/{len(video_files)}: {os.path.basename(vid)}")
            analyze_video(vid, args.output_dir, args.confidence, args.fps_sample, args.model_id, args.device)
    else:
        # Single file processing
        analyze_video(args.input, args.output_dir, args.confidence, args.fps_sample, args.model_id, args.device)
