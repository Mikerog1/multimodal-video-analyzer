import os
import time
import cv2
from PIL import Image
from tqdm import tqdm
from datetime import timedelta
from utils.overlay_renderer import draw_overlays
from utils.report_generator import generate_reports

class VideoProcessor:
    def __init__(self, detector):
        """Initializes the VideoProcessor with a detector.
        
        Args:
            detector: An instance of a detector implementing BaseDetector.
        """
        self.detector = detector

    def process_video(self, video_path: str, output_dir: str, fps_sample: float) -> None:
        """Processes the video, running object detection, drawing overlays, and writing reports.
        
        Args:
            video_path: Path to the input video file.
            output_dir: Directory to save results.
            fps_sample: Inference frequency in frames per second.
        """
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
        
        # Setup Output Paths (Create a unique results folder)
        timestamp_str = time.strftime("%Y%m%d_%H%M")
        run_output_dir = os.path.join(output_dir, f"results_{base_name}_{timestamp_str}")
        os.makedirs(run_output_dir, exist_ok=True)
        out_video_path = os.path.join(run_output_dir, f"{base_name}_analyzed.mp4")
        
        # Setup Video Writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_video_path, fourcc, video_fps, (width, height))
        
        # Calculate sampling intervals
        if fps_sample > 0:
            sample_step = int(max(1, round(video_fps / fps_sample)))
            print(f"[+] Sampling Mode: Running AI inference every {sample_step} frames (~{fps_sample} frame(s) per second)")
        else:
            sample_step = 1
            print("[+] Sampling Mode: Running AI inference on EVERY frame")
            
        # Tracking variables
        timeline_data = []  # Stores second-by-second analytics
        last_detected_boxes = []  # Persisted detections for interpolation
        
        # Summary metrics
        max_counts = {"person": 0, "car": 0, "dog": 0}
        sum_counts = {"person": 0, "car": 0, "dog": 0}
        seconds_tracked = 0
        
        model_name = getattr(self.detector, "model_id", "custom").split('/')[-1]
        device = getattr(self.detector, "device", "cpu")
        confidence_threshold = getattr(self.detector, "confidence_threshold", 0.0)
        
        print(f"\n[+] Analyzing video frames...")
        
        # Wrap in tqdm progress bar
        for frame_idx in tqdm(range(total_frames), desc="Analyzing Frames"):
            ret, frame = cap.read()
            if not ret:
                break
                
            elapsed_sec = frame_idx / video_fps
            should_infer = (frame_idx % sample_step == 0)
            
            if should_infer:
                # Prepare image for PyTorch model
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                # Model inference
                raw_detections = self.detector.detect(pil_img)
                
                # Extract and filter target detections
                last_detected_boxes = []
                frame_counts = {"person": 0, "car": 0, "dog": 0}
                
                for det in raw_detections:
                    lbl = det["label"].lower()
                    if lbl in ["person", "car", "dog"]:
                        last_detected_boxes.append(det)
                        frame_counts[lbl] += 1
                
                # Update summary metrics
                for k in max_counts:
                    max_counts[k] = max(max_counts[k], frame_counts[k])
            
            # Recalculate counts for the active frame (drawn boxes)
            active_counts = {"person": 0, "car": 0, "dog": 0}
            for det in last_detected_boxes:
                active_counts[det["label"]] += 1
                
            # Log second boundary data
            if frame_idx % int(video_fps) == 0 or frame_idx == total_frames - 1:
                sec_rounded = int(elapsed_sec)
                if not any(d["second"] == sec_rounded for d in timeline_data):
                    timeline_data.append({
                        "second": sec_rounded,
                        "timestamp": str(timedelta(seconds=sec_rounded)),
                        "people": active_counts["person"],
                        "cars": active_counts["car"],
                        "dogs": active_counts["dog"]
                    })
                    for k in sum_counts:
                        sum_counts[k] += active_counts[k]
                    seconds_tracked += 1
            
            # HUD overlay
            annotated_frame = draw_overlays(
                frame,
                last_detected_boxes,
                active_counts,
                elapsed_sec,
                duration,
                model_name,
                device
            )
            
            # Write to Output Video
            out_writer.write(annotated_frame)
            
        # Release resources
        cap.release()
        out_writer.release()
        
        # Calculate Averages
        avg_counts = {
            "person": round(sum_counts["person"] / seconds_tracked, 2) if seconds_tracked > 0 else 0,
            "car": round(sum_counts["car"] / seconds_tracked, 2) if seconds_tracked > 0 else 0,
            "dog": round(sum_counts["dog"] / seconds_tracked, 2) if seconds_tracked > 0 else 0,
        }
        
        # Build report summary data
        report_summary = {
            "metadata": {
                "video_file": filename,
                "resolution": f"{width}x{height}",
                "fps": round(video_fps, 2),
                "total_frames": total_frames,
                "duration_seconds": round(duration, 2),
                "model": getattr(self.detector, "model_id", "unknown"),
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
        
        # Save JSON and CSV Reports (inside unique run folder)
        out_csv_path, _ = generate_reports(run_output_dir, base_name, report_summary)
        
        # Print CLI Summary Table
        self.print_cli_summary(filename, duration, max_counts, avg_counts, out_video_path, out_csv_path)

    def print_cli_summary(self, filename: str, duration: float, max_counts: dict, avg_counts: dict, out_video: str, out_csv: str) -> None:
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
