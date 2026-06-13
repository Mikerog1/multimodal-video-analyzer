import os
import sys

# Add the directory containing this file to PATH on Windows to allow OpenCV to locate the openh264 DLL
if os.name == 'nt':
    core_dir = os.path.abspath(os.path.dirname(__file__))
    os.environ['PATH'] = core_dir + os.pathsep + os.environ.get('PATH', '')

import time
import cv2
from tqdm import tqdm
from datetime import timedelta

from utils.overlay_renderer import draw_overlays
from utils.report_generator import get_video_report_path, write_json_report, write_video_report
from utils.time_utils import seconds_to_timestamp


class VideoProcessor:
    def __init__(self, detector):
        """Initializes the VideoProcessor with a detector."""
        self.detector = detector

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        fps_sample: float,
        codec: str = "mp4v",
        resize_factor: float = 1.0,
        save_sampled_only: bool = False,
        write_json: bool = False,
        progress_callback=None,
    ) -> None:
        """Processes the video with YOLO tracking and writes analyzer-style reports."""
        if not hasattr(self.detector, "track"):
            print("[-] Error: Analyzer-style reports require YOLO object tracking. Use --model-type yolo.")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[-] Error: Could not open video file: {video_path}")
            return

        filename = os.path.basename(video_path)
        base_name = os.path.splitext(filename)[0]
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if video_fps <= 0 or total_frames <= 0:
            cap.release()
            print(f"[-] Error: Could not read video metadata: {video_path}")
            return

        duration = total_frames / video_fps
        out_width = int(width * resize_factor) if 0.1 <= resize_factor < 1.0 else width
        out_height = int(height * resize_factor) if 0.1 <= resize_factor < 1.0 else height

        if fps_sample > 0:
            sample_step = int(max(1, round(video_fps / fps_sample)))
            print(f"[+] Sampling Mode: Running YOLO tracking every {sample_step} frames (~{fps_sample} frame(s) per second)")
        else:
            sample_step = 1
            print("[+] Sampling Mode: Running YOLO tracking on EVERY frame")

        sample_duration = sample_step / video_fps
        out_fps = min(fps_sample, video_fps) if save_sampled_only and fps_sample > 0 else video_fps

        timestamp_str = time.strftime("%Y%m%d_%H%M")
        run_output_dir = os.path.join(output_dir, f"results_{base_name}_{timestamp_str}")
        os.makedirs(run_output_dir, exist_ok=True)
        out_video_path = os.path.join(run_output_dir, f"{base_name}_analyzed.mp4")
        out_csv_path = get_video_report_path(out_video_path)
        out_json_path = os.path.join(run_output_dir, f"report_{base_name}_analyzed.mp4.json")

        fourcc = cv2.VideoWriter_fourcc(*codec)
        out_writer = cv2.VideoWriter(out_video_path, fourcc, out_fps, (out_width, out_height))
        if not out_writer.isOpened():
            cap.release()
            print(f"[-] Error: Could not create output video: {out_video_path}")
            return

        print(f"[+] Video Properties: {width}x{height} | {video_fps:.2f} FPS | {total_frames} frames | {duration:.2f} seconds")
        if resize_factor < 1.0 or save_sampled_only or codec != "mp4v":
            print(f"[+] Output Options: Resolution={out_width}x{out_height} | Codec={codec} | FPS={out_fps:.2f} (sampled-only: {save_sampled_only})")

        tracked_objects = {}
        active_detections = []
        model_name = getattr(self.detector, "model_id", "custom").split('/')[-1]
        device = getattr(self.detector, "device", "cpu")

        print("\n[+] Analyzing video frames with object tracking...")
        for frame_idx in tqdm(range(total_frames), desc="Analyzing Frames"):
            if progress_callback:
                progress_callback(frame_idx, total_frames)
                
            ret, frame = cap.read()
            if not ret:
                break

            elapsed_sec = frame_idx / video_fps
            should_infer = frame_idx % sample_step == 0

            if should_infer:
                remaining_duration = max(0.0, duration - elapsed_sec)
                screen_time_increment = min(sample_duration, remaining_duration)
                active_detections = self.detector.track(
                    frame,
                    elapsed_sec,
                    screen_time_increment,
                    tracked_objects,
                )

            cumulative_counts = self._count_tracked_objects(tracked_objects)
            overlay_detections = [detection.to_overlay_detection() for detection in active_detections]

            if not save_sampled_only or should_infer:
                annotated_frame = draw_overlays(
                    frame,
                    overlay_detections,
                    cumulative_counts,
                    elapsed_sec,
                    duration,
                    model_name,
                    device,
                )

                if resize_factor < 1.0:
                    annotated_frame = cv2.resize(annotated_frame, (out_width, out_height))

                out_writer.write(annotated_frame)

        cap.release()
        out_writer.release()

        out_csv_path = write_video_report(out_csv_path, tracked_objects)
        if write_json:
            write_json_report(
                out_json_path,
                self._build_json_report(
                    filename,
                    out_video_path,
                    width,
                    height,
                    video_fps,
                    total_frames,
                    duration,
                    fps_sample,
                    tracked_objects,
                ),
            )

        self.print_cli_summary(filename, duration, tracked_objects, out_video_path, str(out_csv_path), out_json_path if write_json else None)

    def _count_tracked_objects(self, tracked_objects: dict) -> dict:
        counts = {}
        for tracked_object in tracked_objects.values():
            counts[tracked_object.object_type] = counts.get(tracked_object.object_type, 0) + 1
        return counts

    def _build_json_report(
        self,
        filename: str,
        out_video_path: str,
        width: int,
        height: int,
        video_fps: float,
        total_frames: int,
        duration: float,
        fps_sample: float,
        tracked_objects: dict,
    ) -> dict:
        return {
            "metadata": {
                "video_file": filename,
                "output_video_file": os.path.basename(out_video_path),
                "resolution": f"{width}x{height}",
                "fps": round(video_fps, 2),
                "total_frames": total_frames,
                "duration_seconds": round(duration, 2),
                "duration": seconds_to_timestamp(duration),
                "model": getattr(self.detector, "model_id", "unknown"),
                "device": getattr(self.detector, "device", "cpu"),
                "confidence_threshold": getattr(self.detector, "confidence_threshold", 0.0),
                "fps_sample": fps_sample,
                "analysis_time_utc": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
            },
            "objects": [
                {
                    "object_id": object_id,
                    "first_time_seen": seconds_to_timestamp(tracked_object.first_time_seen_seconds),
                    "total_screen_time": seconds_to_timestamp(tracked_object.screen_time_seconds),
                    "object_type": tracked_object.object_type,
                    "bbox_observations": [observation.__dict__ for observation in tracked_object.bbox_observations],
                }
                for object_id, tracked_object in sorted(tracked_objects.items())
            ],
        }

    def print_cli_summary(
        self,
        filename: str,
        duration: float,
        tracked_objects: dict,
        out_video: str,
        out_csv: str,
        out_json: str | None = None,
    ) -> None:
        """Prints a summary table of tracked object counts to the console."""
        counts = {"person": 0, "car": 0, "truck": 0, "other": 0}
        screen_times = {"person": 0.0, "car": 0.0, "truck": 0.0, "other": 0.0}

        for tracked_object in tracked_objects.values():
            bucket = tracked_object.object_type if tracked_object.object_type in counts else "other"
            counts[bucket] += 1
            screen_times[bucket] += tracked_object.screen_time_seconds

        border = "=" * 70
        thin_line = "-" * 70
        print(f"\n{border}")
        print(f"  ANALYSIS COMPLETE: {filename}")
        print(f"  Duration: {str(timedelta(seconds=int(duration)))} ({duration:.1f}s)")
        print(f"{border}")
        print("  Class      | Object Count | Total Screen Time")
        print(f"  {thin_line[0:11]}+{thin_line[0:14]}+{thin_line[0:19]}")
        print(f"  People     | {counts['person']:<12} | {seconds_to_timestamp(screen_times['person'])}")
        print(f"  Cars       | {counts['car']:<12} | {seconds_to_timestamp(screen_times['car'])}")
        print(f"  Trucks     | {counts['truck']:<12} | {seconds_to_timestamp(screen_times['truck'])}")
        print(f"  Other      | {counts['other']:<12} | {seconds_to_timestamp(screen_times['other'])}")
        print(f"{border}")
        print("  Outputs Saved:")
        print(f"  - Annotated Video : {os.path.abspath(out_video)}")
        print(f"  - CSV Report      : {os.path.abspath(out_csv)}")
        if out_json is not None:
            print(f"  - JSON Report     : {os.path.abspath(out_json)}")
        print(f"{border}\n")
