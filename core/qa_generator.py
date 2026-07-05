import re
from datetime import timedelta

from core.constants import CLASS_SYNONYMS, VEHICLE_CLASSES
from core.counting import count_unique_tracks, aggregate_video_stats
from utils.time_utils import timestamp_to_seconds


def parse_filename_lighting(filename):
    """Parses filename to identify if it is a night run based on Karlsruhe naming conventions."""
    match = re.search(r'_(\d{2})(\d{2})(\d{2})_', filename)
    if match:
        hour = int(match.group(1))
        if hour >= 19 or hour < 7:
            return "night"
        return "day"
    return None


def format_time(seconds):
    """Formats seconds into HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))


class QAGenerator:
    def __init__(self, filename, processed_frames, duration, tracked_objects=None,
                 video_path=None, qa_categories=None, captions=None):
        self.filename = filename
        self.processed_frames = processed_frames
        self.duration = duration
        self.tracked_objects = tracked_objects if tracked_objects is not None else {}
        self.video_path = video_path
        self.qa_categories = qa_categories if qa_categories is not None else ["counting"]
        self.captions = captions

        self.file_lighting = parse_filename_lighting(filename)

    def generate_qa_pairs(self):
        """Generates counting-only QA pairs (rule-based, no LLM)."""
        return self.generate_rule_based()

    def generate_rule_based(self):
        """Generates high-quality rule-based counting QA pairs."""
        qa_by_category = {cat: [] for cat in self.qa_categories}

        if "counting" not in qa_by_category:
            return qa_by_category

        # Check caption for keywords to customize rule-based questions
        caption_lower = self.captions.lower() if self.captions else ""
        env_ref = "in the video"
        if "highway" in caption_lower or "motorway" in caption_lower or "autobahn" in caption_lower:
            env_ref = "on the highway"
        elif "intersection" in caption_lower:
            env_ref = "at the intersection"
        elif "city" in caption_lower or "town" in caption_lower:
            env_ref = "on the city street"
        elif "tunnel" in caption_lower:
            env_ref = "in the tunnel"

        # Segment the video into 10-second intervals
        segment_duration = 10.0
        segments = []
        t = 0.0
        while t < self.duration:
            t_end = min(t + segment_duration, self.duration)
            if t_end - t >= 2.0:
                segments.append((t, t_end))
            t += segment_duration

        if not segments and self.duration > 0:
            segments.append((0.0, self.duration))

        for t_start, t_end in segments:
            t_start_str = format_time(t_start)
            t_end_str = format_time(t_end)
            span_str = f"{t_start_str} - {t_end_str}"

            segment_frames = [f for f in self.processed_frames if t_start <= f["timestamp"] <= t_end]
            if not segment_frames:
                continue

            avg_blur = sum(f.get("blur_var", 200.0) for f in segment_frames) / len(segment_frames)
            is_blurred = avg_blur < 80.0

            avg_brightness = sum(f.get("brightness", 128.0) for f in segment_frames) / len(segment_frames)
            day_night = "night" if (self.file_lighting == "night" or avg_brightness < 55.0) else "day"

            visibility = "blurred" if is_blurred else ("dark" if day_night == "night" else "clear")

            # Calculate counts using central count function
            ped_count_res = count_unique_tracks(self.tracked_objects, ["person"], t_start, t_end)
            num_pedestrians = ped_count_res["count"]

            veh_count_res = count_unique_tracks(self.tracked_objects, VEHICLE_CLASSES, t_start, t_end)
            num_vehicles = veh_count_res["count"]

            # --- Counting QA: Pedestrians ---
            if num_pedestrians > 0:
                difficulty = "hard" if is_blurred or num_pedestrians >= 4 else ("medium" if num_pedestrians >= 2 else "easy")
                expected_ans = f"at least {num_pedestrians}" if ped_count_res["confidence_signal"] == "low" else str(num_pedestrians)
                qa_by_category["counting"].append({
                    "Question": f"How many pedestrians are visible {env_ref} in the video segment from {t_start_str} to {t_end_str}?",
                    "Answer": expected_ans,
                    "Answer format": "open-ended",
                    "Evidence spans the video": span_str,
                    "Reasoning type": "counting",
                    "Difficulty level": difficulty,
                    "Visibility quality": visibility,
                    "Day or night tag": day_night,
                    "Trajectory linkage": None,
                    "Unanswerable flag": False,
                    "_target_class": "person",
                    "_segment_start_s": t_start,
                    "_segment_end_s": t_end,
                    "_track_ids_counted": ped_count_res["track_ids"],
                    "_confidence_signal": ped_count_res["confidence_signal"],
                    "_verification_status": "flagged_for_review" if ped_count_res["confidence_signal"] == "low" else "auto_verified"
                })

            # --- Counting QA: Vehicles ---
            if num_vehicles > 0:
                difficulty = "hard" if is_blurred or num_vehicles >= 5 else ("medium" if num_vehicles >= 2 else "easy")
                expected_ans = f"at least {num_vehicles}" if veh_count_res["confidence_signal"] == "low" else str(num_vehicles)
                qa_by_category["counting"].append({
                    "Question": f"How many vehicles are visible {env_ref} in the video segment from {t_start_str} to {t_end_str}?",
                    "Answer": expected_ans,
                    "Answer format": "open-ended",
                    "Evidence spans the video": span_str,
                    "Reasoning type": "counting",
                    "Difficulty level": difficulty,
                    "Visibility quality": visibility,
                    "Day or night tag": day_night,
                    "Trajectory linkage": None,
                    "Unanswerable flag": False,
                    "_target_class": "vehicle",
                    "_segment_start_s": t_start,
                    "_segment_end_s": t_end,
                    "_track_ids_counted": veh_count_res["track_ids"],
                    "_confidence_signal": veh_count_res["confidence_signal"],
                    "_verification_status": "flagged_for_review" if veh_count_res["confidence_signal"] == "low" else "auto_verified"
                })

        # --- Video-wide Aggregate Statistics QA ---
        agg_stats = aggregate_video_stats(self.tracked_objects, self.duration)

        for cls, total in agg_stats.get("unique_total_per_class", {}).items():
            if total > 0:
                qa_by_category["counting"].append({
                    "Question": f"How many unique {cls}s appear in the entire video?",
                    "Answer": str(total),
                    "Answer format": "open-ended",
                    "Evidence spans the video": f"0:00:00 - {format_time(self.duration)}",
                    "Reasoning type": "counting",
                    "Difficulty level": "medium",
                    "Visibility quality": "clear",
                    "Day or night tag": self.file_lighting if self.file_lighting else "day",
                    "Trajectory linkage": None,
                    "Unanswerable flag": False,
                    "_target_class": cls,
                    "_segment_start_s": 0.0,
                    "_segment_end_s": self.duration,
                    "_track_ids_counted": [tid for tid, t in self.tracked_objects.items() if t.object_type == cls],
                    "_confidence_signal": "high",
                    "_verification_status": "auto_verified"
                })

        for cls, peak in agg_stats.get("peak_concurrent_per_class", {}).items():
            if peak > 0:
                qa_by_category["counting"].append({
                    "Question": f"What is the peak concurrent number of {cls}s visible at the same time in the video?",
                    "Answer": str(peak),
                    "Answer format": "open-ended",
                    "Evidence spans the video": f"0:00:00 - {format_time(self.duration)}",
                    "Reasoning type": "counting",
                    "Difficulty level": "medium",
                    "Visibility quality": "clear",
                    "Day or night tag": self.file_lighting if self.file_lighting else "day",
                    "Trajectory linkage": None,
                    "Unanswerable flag": False,
                    "_target_class": cls,
                    "_segment_start_s": 0.0,
                    "_segment_end_s": self.duration,
                    "_track_ids_counted": [],
                    "_confidence_signal": "high",
                    "_verification_status": "auto_verified"
                })

        return qa_by_category
