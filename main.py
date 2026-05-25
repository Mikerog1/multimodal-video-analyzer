import os
import sys
import time
import argparse
import torch

from core.video_processor import VideoProcessor

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multimodal video analyzer which counts objects, draws bounding boxes, and outputs CSV, JSON and the edited video."
    )
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
        "--model-type", 
        type=str, 
        choices=["detr", "yolo"], 
        default="detr", 
        help="Type of model architecture to use ('detr' or 'yolo')"
    )
    parser.add_argument(
        "--model-id", 
        type=str, 
        default=None, 
        help="Model ID (Hugging Face model ID for DETR, or YOLOv8 model path like 'yolov8n.pt'). Defaults to 'facebook/detr-resnet-50' for DETR or 'yolov8n.pt' for YOLO."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto", 
        help="Device to run inference on ('cuda', 'cpu', or 'auto')"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validate input exists
    if not os.path.exists(args.input):
        print(f"[-] Error: Input path does not exist: {args.input}")
        sys.exit(1)
        
    # Determine execution device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        
    print(f"\n[+] Target Execution Device: {device.upper()}")
    
    # Determine the model ID based on type if not specified
    model_id = args.model_id
    if not model_id:
        if args.model_type == "detr":
            model_id = "facebook/detr-resnet-50"
        elif args.model_type == "yolo":
            model_id = os.path.join("models", "yolov8n.pt")
            
    # Instantiate the correct detector
    if args.model_type == "yolo":
        from models.yolo_detector import YoloDetector
        print(f"[+] Loading YOLO model: {model_id}...")
        t0 = time.time()
        try:
            detector = YoloDetector(model_id, device, args.confidence)
        except Exception as e:
            print(f"[-] Error loading YOLO model {model_id}: {e}")
            sys.exit(1)
    else:
        from models.detr_detector import DetrDetector
        print(f"[+] Loading DETR model: {model_id} from Hugging Face...")
        t0 = time.time()
        try:
            detector = DetrDetector(model_id, device, args.confidence)
        except Exception as e:
            print(f"[-] Error loading DETR model {model_id}: {e}")
            sys.exit(1)
            
    print(f"[+] Model loaded in {time.time() - t0:.2f} seconds.")
    
    # Initialize the video processor pipeline
    processor = VideoProcessor(detector)
    
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
            processor.process_video(vid, args.output_dir, args.fps_sample)
    else:
        # Single file processing
        processor.process_video(args.input, args.output_dir, args.fps_sample)

if __name__ == "__main__":
    main()
