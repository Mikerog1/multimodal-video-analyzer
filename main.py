import os
import sys

# Add core directory to PATH on Windows to allow OpenCV to locate the openh264 DLL
if os.name == 'nt':
    core_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'core'))
    os.environ['PATH'] = core_dir + os.pathsep + os.environ.get('PATH', '')

import argparse
import time
from pathlib import Path

import torch

from core.video_processor import VideoProcessor
from utils.report_generator import create_total_report, find_video_reports


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mva",
        description="Analyze videos with YOLO object tracking and create analyzer-style CSV reports.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a video file or directory and create per-output-video CSV reports.",
    )
    analyze_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input video file or a folder of videos",
    )
    analyze_parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save analyzed videos and reports",
    )
    analyze_parser.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        help="Minimum confidence threshold for detections (0.0 to 1.0)",
    )
    analyze_parser.add_argument(
        "--fps-sample",
        type=float,
        default=1.0,
        help="Frames per second to sample and run YOLO tracking (0 = analyze all frames)",
    )
    analyze_parser.add_argument(
        "--model-type",
        type=str,
        choices=["detr", "yolo"],
        default="yolo",
        help="Model architecture type. Analyzer-style object tracking requires 'yolo'.",
    )
    analyze_parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="YOLOv8 model path like 'yolov8n.pt'. Defaults to bundled models/yolov8n.pt for YOLO.",
    )
    analyze_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on ('cuda', 'cpu', or 'auto')",
    )
    analyze_parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="Video codec to use for the output video (e.g., 'mp4v', 'avc1'). Default is 'mp4v'.",
    )
    analyze_parser.add_argument(
        "--resize-factor",
        type=float,
        default=1.0,
        help="Scale factor for the output video resolution (0.1 to 1.0).",
    )
    analyze_parser.add_argument(
        "--save-sampled-only",
        action="store_true",
        help="Only write frames that are analyzed by the YOLO tracker.",
    )
    analyze_parser.add_argument(
        "--json",
        action="store_true",
        help="Also write JSON reports. Disabled by default.",
    )

    total_report_parser = subparsers.add_parser(
        "total-report",
        help="Aggregate per-video CSV reports into total_report.csv.",
    )
    total_report_parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        help="Input directory to scan recursively for analyzer-style video reports.",
    )

    return parser


def find_input_videos(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in VIDEO_EXTENSIONS else []

    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
        and not path.name.lower().startswith("debug")
    )


def load_detector(args):
    if args.model_type != "yolo":
        raise ValueError("Analyzer-style object tracking requires --model-type yolo.")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if args.device == "auto" and not torch.cuda.is_available():
        device = "cpu"

    model_id = args.model_id or os.path.join("models", "yolov8n.pt")

    from models.yolo_detector import YoloDetector

    print(f"\n[+] Target Execution Device: {device.upper()}")
    print(f"[+] Loading YOLO model: {model_id}...")
    t0 = time.time()
    detector = YoloDetector(model_id, device, args.confidence)
    print(f"[+] Model loaded in {time.time() - t0:.2f} seconds.")
    return detector


def run_analyze(args, parser: argparse.ArgumentParser) -> int:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        parser.error(f"Input path does not exist: {input_path}")

    try:
        detector = load_detector(args)
    except Exception as exc:
        print(f"[-] Error loading detector: {exc}")
        return 1

    videos = find_input_videos(input_path)
    if not videos:
        print(f"[-] Error: No supported video files found: {input_path}")
        return 1

    processor = VideoProcessor(detector)
    print(f"[+] Found {len(videos)} video(s) to process")
    for idx, video_path in enumerate(videos):
        print(f"\n[+] Processing video {idx + 1}/{len(videos)}: {video_path.name}")
        processor.process_video(
            str(video_path),
            args.output_dir,
            args.fps_sample,
            codec=args.codec,
            resize_factor=args.resize_factor,
            save_sampled_only=args.save_sampled_only,
            write_json=args.json,
        )

    return 0


def run_total_report(args, parser: argparse.ArgumentParser) -> int:
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        parser.error(f"Input directory does not exist or is not a directory: {input_dir}")

    report_count = len(find_video_reports(input_dir))
    print(f"Found {report_count} video report(s).")
    total_report_path = create_total_report(input_dir)
    print(f"Wrote {total_report_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        return run_analyze(args, parser)

    if args.command == "total-report":
        return run_total_report(args, parser)

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
