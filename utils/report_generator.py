import os
import csv
import json
from pathlib import Path

import cv2

from utils.time_utils import seconds_to_timestamp, timestamp_to_seconds


VIDEO_REPORT_COLUMNS = [
    "object_id",
    "first_time_seen",
    "total_screen_time",
    "object_type",
    "bbox-coords",
]

TOTAL_REPORT_COLUMNS = [
    "filename",
    "file-dir",
    "video-duration",
    "total-amount-of-persons",
    "total-amount-of-person-screen-time",
    "total-amount-of-cars",
    "total-amount-of-cars-screen-time",
    "total-amount-of-trucks",
    "total-amount-of-trucks-screen-time",
    "total-amount-of-other-objects",
    "total-amount-of-other-objects-screen-time",
]


def get_video_report_path(video_path: str | Path, output_dir: str | Path | None = None) -> Path:
    video_path = Path(video_path)
    report_dir = Path(output_dir) if output_dir is not None else video_path.parent
    return report_dir / f"report_{video_path.name}.csv"


def write_video_report(report_path: str | Path, tracked_objects: dict) -> Path:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=VIDEO_REPORT_COLUMNS)
        writer.writeheader()

        for object_id, tracked_object in sorted(tracked_objects.items()):
            if not tracked_object.bbox_observations:
                continue

            first_observation = tracked_object.bbox_observations[0]
            writer.writerow(
                {
                    "object_id": object_id,
                    "first_time_seen": seconds_to_timestamp(
                        tracked_object.first_time_seen_seconds
                    ),
                    "total_screen_time": seconds_to_timestamp(
                        tracked_object.screen_time_seconds
                    ),
                    "object_type": tracked_object.object_type,
                    "bbox-coords": json.dumps(
                        {
                            "timestamp": first_observation.timestamp,
                            "x1": first_observation.x1,
                            "y1": first_observation.y1,
                            "x2": first_observation.x2,
                            "y2": first_observation.y2,
                            "confidence": first_observation.confidence,
                        },
                        separators=(",", ":"),
                    ),
                }
            )

    return report_path


def write_json_report(report_path: str | Path, report_data: dict) -> Path:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as json_file:
        json.dump(report_data, json_file, indent=2)
    return report_path


def find_video_reports(input_dir: str | Path) -> list[Path]:
    input_dir = Path(input_dir)
    return sorted(
        path
        for path in input_dir.rglob("report_*.csv")
        if path.is_file() and path.name != "total_report.csv"
    )


def original_filename_from_report(report_path: str | Path) -> str:
    filename = Path(report_path).name.removeprefix("report_")
    return filename.removesuffix(".csv")


def get_video_duration_seconds(video_path: str | Path) -> float:
    video_path = Path(video_path)
    if not video_path.exists():
        return 0.0

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return 0.0

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()

    if fps <= 0 or frame_count <= 0:
        return 0.0

    return frame_count / fps


def summarize_video_report(report_path: str | Path) -> dict[str, str | int]:
    report_path = Path(report_path)
    original_filename = original_filename_from_report(report_path)
    video_duration = get_video_duration_seconds(report_path.with_name(original_filename))

    person_count = 0
    person_seconds = 0.0
    car_count = 0
    car_seconds = 0.0
    truck_count = 0
    truck_seconds = 0.0
    other_count = 0
    other_seconds = 0.0

    with report_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            object_type = row.get("object_type", "")
            try:
                screen_time = timestamp_to_seconds(
                    row.get("total_screen_time", "00:00:00:000")
                )
            except ValueError:
                screen_time = 0.0

            if object_type == "person":
                person_count += 1
                person_seconds += screen_time
            elif object_type == "car":
                car_count += 1
                car_seconds += screen_time
            elif object_type == "truck":
                truck_count += 1
                truck_seconds += screen_time
            else:
                other_count += 1
                other_seconds += screen_time

    return {
        "filename": original_filename,
        "file-dir": str(report_path.parent),
        "video-duration": seconds_to_timestamp(video_duration),
        "total-amount-of-persons": person_count,
        "total-amount-of-person-screen-time": seconds_to_timestamp(person_seconds),
        "total-amount-of-cars": car_count,
        "total-amount-of-cars-screen-time": seconds_to_timestamp(car_seconds),
        "total-amount-of-trucks": truck_count,
        "total-amount-of-trucks-screen-time": seconds_to_timestamp(truck_seconds),
        "total-amount-of-other-objects": other_count,
        "total-amount-of-other-objects-screen-time": seconds_to_timestamp(other_seconds),
    }


def create_total_report(input_dir: str | Path) -> Path:
    input_dir = Path(input_dir)
    report_paths = find_video_reports(input_dir)
    total_report_path = input_dir / "total_report.csv"

    with total_report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=TOTAL_REPORT_COLUMNS)
        writer.writeheader()
        for report_path in report_paths:
            writer.writerow(summarize_video_report(report_path))

    return total_report_path

def generate_reports(output_dir: str, base_name: str, report_data: dict) -> tuple:
    """Generates CSV and JSON report files in the output directory.
    
    Args:
        output_dir: Directory where reports will be saved.
        base_name: Base filename (without extension) for the reports.
        report_data: Dictionary containing metadata, summary, and timeline.
        
    Returns:
        A tuple of absolute paths: (out_csv_path, out_json_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    out_csv_path = os.path.join(output_dir, f"{base_name}_analysis.csv")
    out_json_path = os.path.join(output_dir, f"{base_name}_analysis.json")
    
    # Save CSV Report
    with open(out_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Timestamp", "Second", "People Count", "Car Count", "Dog Count"])
        for entry in report_data.get("timeline", []):
            writer.writerow([
                entry.get("timestamp", ""),
                entry.get("second", 0),
                entry.get("people", 0),
                entry.get("cars", 0),
                entry.get("dogs", 0)
            ])
            
    # Save JSON Report
    with open(out_json_path, mode="w") as json_file:
        json.dump(report_data, json_file, indent=2)
        
    return out_csv_path, out_json_path
