import re
import json
import requests
from datetime import timedelta

VOCABULARY = [
    "traffic light", 
    "bicycle", 
    "stroller", 
    "motorcycle", 
    "bus", 
    "truck", 
    "traffic sign", 
    "dog", 
    "cat", 
    "fire hydrant", 
    "bench"
]

def calculate_iou(boxA, boxB):
    """Calculates Intersection over Union (IoU) of two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0.0
    return iou

def parse_filename_lighting(filename):
    """Parses filename to identify if it is a night run based on Karlsruhe naming conventions."""
    match = re.search(r'_(\d{2})(\d{2})(\d{2})_', filename)
    if match:
        hour = int(match.group(1))
        if hour >= 19 or hour < 7:
            return "night"
        return "day"
    return None

CLASS_SYNONYMS = {
    "person": ["person", "people", "pedestrian", "pedestrians", "walker", "walkers", "man", "woman", "men", "women"],
    "car": ["car", "cars"],
    "truck": ["truck", "trucks"],
    "bus": ["bus", "buses"],
    "motorcycle": ["motorcycle", "motorcycles", "motorbike", "motorbikes"],
    "bicycle": ["bicycle", "bicycles", "bike", "bikes"],
    "dog": ["dog", "dogs", "puppy", "puppies"],
    "cat": ["cat", "cats", "kitten", "kittens"],
    "traffic light": ["traffic light", "traffic lights"],
    "traffic sign": ["traffic sign", "traffic signs", "stop sign", "stop signs"],
    "bench": ["bench", "benches"],
    "fire hydrant": ["fire hydrant", "fire hydrants"],
    "stroller": ["stroller", "strollers", "baby carriage"]
}

VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle"]

def parse_timestamp_to_seconds(ts_str: str) -> float:
    """Converts a timestamp string in HH:MM:SS:MS, HH:MM:SS, or MM:SS to seconds."""
    parts = ts_str.split(':')
    try:
        if len(parts) == 4:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2]) + int(parts[3]) / 1000.0
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
    except ValueError:
        pass
    return 0.0

def format_time(seconds):
    """Formats seconds into HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))

class QAGenerator:
    def __init__(self, filename, processed_frames, duration, tracked_objects=None, video_path=None, qa_categories=None, captions=None, example_questions=None, gemini_api_key=None, custom_vlm_url=None, custom_vlm_key=None, custom_vlm_model_id=None):
        self.filename = filename
        self.processed_frames = processed_frames
        self.duration = duration
        self.tracked_objects = tracked_objects if tracked_objects is not None else {}
        self.video_path = video_path
        self.qa_categories = qa_categories if qa_categories is not None else ["counting", "negative", "ambiguity", "day_night"]
        self.captions = captions
        self.example_questions = example_questions
        self.gemini_api_key = gemini_api_key
        self.custom_vlm_url = custom_vlm_url
        self.custom_vlm_key = custom_vlm_key
        self.custom_vlm_model_id = custom_vlm_model_id
        
        self.file_lighting = parse_filename_lighting(filename)

    def generate_rule_based(self):
        """Generates high-quality enhanced rule-based QA pairs."""
        from core.counting import count_unique_tracks, aggregate_video_stats
        
        qa_by_category = {cat: [] for cat in self.qa_categories}
        
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
            
        weather_ref = ""
        if "rain" in caption_lower or "wet" in caption_lower:
            weather_ref = "rainy "
        elif "fog" in caption_lower or "mist" in caption_lower:
            weather_ref = "foggy "
        elif "snow" in caption_lower:
            weather_ref = "snowy "
            
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
            
        # Determine maximum x-coordinate to estimate frame width
        max_x = 1280.0
        for track in self.tracked_objects.values():
            for obs in track.bbox_observations:
                max_x = max(max_x, obs.x2)
        
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
            
            # Determine detected labels in segment
            detected_labels = set()
            for track in self.tracked_objects.values():
                for obs in track.bbox_observations:
                    try:
                        obs_time = parse_timestamp_to_seconds(obs.timestamp)
                    except Exception:
                        continue
                    if t_start <= obs_time <= t_end:
                        detected_labels.add(track.object_type)
                        break
            
            # Calculate counts using central count function
            ped_count_res = count_unique_tracks(self.tracked_objects, ["person"], t_start, t_end)
            num_pedestrians = ped_count_res["count"]
            
            veh_count_res = count_unique_tracks(self.tracked_objects, VEHICLE_CLASSES, t_start, t_end)
            num_vehicles = veh_count_res["count"]
            
            dog_count_res = count_unique_tracks(self.tracked_objects, ["dog"], t_start, t_end)
            num_dogs = dog_count_res["count"]
            
            # --- 1. Counting QA ---
            if "counting" in self.qa_categories:
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

            # --- 2. Negative / Absence QA ---
            if "negative" in self.qa_categories:
                detected_mapped = set()
                for lbl in detected_labels:
                    if lbl in VEHICLE_CLASSES:
                        detected_mapped.add("vehicle")
                    elif lbl in ["traffic light", "dog", "cat", "fire hydrant", "bench"]:
                        detected_mapped.add(lbl)
                    elif lbl == "stop sign":
                        detected_mapped.add("traffic sign")
                
                absent_candidates = []
                for item in VOCABULARY:
                    if item == "stroller":
                        absent_candidates.append(item)
                    elif item == "traffic sign" and "traffic sign" not in detected_mapped:
                        absent_candidates.append(item)
                    elif item in ["motorcycle", "bus", "truck", "bicycle"] and "vehicle" not in detected_mapped:
                        absent_candidates.append(item)
                    elif item not in detected_mapped:
                        absent_candidates.append(item)
                        
                if absent_candidates:
                    seed_idx = int(t_start * 100) % len(absent_candidates)
                    selected_absent = absent_candidates[seed_idx]
                    qa_by_category["negative"].append({
                        "Question": f"Is there any {selected_absent} present {env_ref} in the video segment from {t_start_str} to {t_end_str}?",
                        "Answer": "no",
                        "Answer format": "yes-no",
                        "Evidence spans the video": span_str,
                        "Reasoning type": "presence-absence",
                        "Difficulty level": "easy",
                        "Visibility quality": visibility,
                        "Day or night tag": day_night,
                        "Trajectory linkage": None,
                        "Unanswerable flag": False
                    })

            # --- 3. Spatial-Temporal QA ---
            if "ambiguity" in self.qa_categories or "spatial-temporal" in self.qa_categories:
                # 3a. Spatial Position Query (Screen side)
                for cls_name, display_name in [("person", "pedestrian"), ("car", "vehicle"), ("truck", "vehicle")]:
                    matching_tracks = [t for t in self.tracked_objects.values() if t.object_type == cls_name]
                    # Filter active in segment
                    seg_tracks = []
                    for t in matching_tracks:
                        obs_in_seg = []
                        for obs in t.bbox_observations:
                            try:
                                obs_time = parse_timestamp_to_seconds(obs.timestamp)
                            except Exception:
                                continue
                            if t_start <= obs_time <= t_end:
                                obs_in_seg.append(obs)
                        if obs_in_seg:
                            seg_tracks.append((t, obs_in_seg))
                            
                    if len(seg_tracks) == 1:
                        track, boxes = seg_tracks[0]
                        avg_x = sum((b.x1 + b.x2) / 2 for b in boxes) / len(boxes)
                        side = "left side" if avg_x < (max_x * 0.4) else ("right side" if avg_x > (max_x * 0.6) else "middle")
                        
                        qa_by_category["ambiguity"].append({
                            "Question": f"On which side of the screen is the {display_name} visible {env_ref} in the segment from {t_start_str} to {t_end_str}?",
                            "Answer": side,
                            "Answer format": "open-ended",
                            "Evidence spans the video": span_str,
                            "Reasoning type": "spatial-temporal",
                            "Difficulty level": "easy",
                            "Visibility quality": visibility,
                            "Day or night tag": day_night,
                            "Trajectory linkage": None,
                            "Unanswerable flag": False
                        })
                        break

            # --- 4. Day vs. Night Robustness QA ---
            if "day_night" in self.qa_categories:
                if day_night == "night":
                    target_ref = "vehicle" if num_vehicles > 0 else ("pedestrian" if num_pedestrians > 0 else "road layout")
                    qa_by_category["day_night"].append({
                        "Question": f"Is the {target_ref} clearly visible despite the {weather_ref}low lighting in the night segment from {t_start_str} to {t_end_str}?",
                        "Answer": "yes" if target_ref != "road layout" else "partially",
                        "Answer format": "yes-no" if target_ref != "road layout" else "open-ended",
                        "Evidence spans the video": span_str,
                        "Reasoning type": "low-light-robustness",
                        "Difficulty level": "medium",
                        "Visibility quality": "dark",
                        "Day or night tag": "night",
                        "Trajectory linkage": None,
                        "Unanswerable flag": False
                    })
                else:
                    qa_by_category["day_night"].append({
                        "Question": f"Is the video lighting representative of clear daylight conditions in the segment from {t_start_str} to {t_end_str}?",
                        "Answer": "yes",
                        "Answer format": "yes-no",
                        "Evidence spans the video": span_str,
                        "Reasoning type": "low-light-robustness",
                        "Difficulty level": "easy",
                        "Visibility quality": "clear",
                        "Day or night tag": "day",
                        "Trajectory linkage": None,
                        "Unanswerable flag": False
                    })

        # --- 5. User Queries (Captions and Custom Predefined Questions) ---
        if "user_queries" in self.qa_categories or self.captions or self.example_questions:
            user_qa = []
            if "user_queries" not in qa_by_category:
                qa_by_category["user_queries"] = user_qa
            else:
                user_qa = qa_by_category["user_queries"]

            if self.captions and self.captions.strip():
                user_qa.append({
                    "Question": "What is the context of this video?",
                    "Answer": self.captions.strip(),
                    "Answer format": "open-ended",
                    "Evidence spans the video": f"00:00:00 - {format_time(self.duration)}",
                    "Reasoning type": "summary-description",
                    "Difficulty level": "easy",
                    "Visibility quality": visibility if 'visibility' in locals() else "clear",
                    "Day or night tag": self.file_lighting if self.file_lighting else "day",
                    "Trajectory linkage": None,
                    "Unanswerable flag": False
                })

            if self.example_questions:
                questions_list = [q.strip() for q in self.example_questions.split('\n') if q.strip()]
                for q in questions_list:
                    q_lower = q.lower()
                    
                    times = []
                    for m in re.finditer(r'\b(?:\d{1,2}:)?\d{1,2}:\d{2}\b', q):
                        times.append(parse_timestamp_to_seconds(m.group(0)))
                    
                    if len(times) >= 2:
                        start_time, end_time = times[0], times[1]
                    elif len(times) == 1:
                        if "after" in q_lower:
                            start_time, end_time = times[0], self.duration
                        elif "before" in q_lower:
                            start_time, end_time = 0.0, times[0]
                        else:
                            start_time, end_time = max(0.0, times[0] - 1.0), min(self.duration, times[0] + 1.0)
                    else:
                        start_time, end_time = 0.0, self.duration
                    
                    start_time = max(0.0, min(start_time, self.duration))
                    end_time = max(0.0, min(end_time, self.duration))
                    if start_time > end_time:
                        start_time, end_time = end_time, start_time
                    
                    requested_classes = []
                    if "vehicle" in q_lower or "vehicles" in q_lower:
                        requested_classes = VEHICLE_CLASSES
                    else:
                        for cls, synonyms in CLASS_SYNONYMS.items():
                            if any(re.search(r'\b' + re.escape(syn) + r'\b', q_lower) for syn in synonyms):
                                requested_classes.append(cls)
                                if cls == "traffic sign":
                                    requested_classes.append("stop sign")
                    
                    # Count using count_unique_tracks
                    count_res = count_unique_tracks(self.tracked_objects, requested_classes, start_time, end_time)
                    count = count_res["count"]
                    
                    is_yes_no = False
                    if any(w in q_lower for w in ["is there", "are there", "can you see", "do you see", "do we see", "present in"]):
                        is_yes_no = True
                    elif q_lower.startswith(("is", "are", "can", "do", "does", "has", "have", "was", "were")):
                        is_yes_no = True
                    
                    if is_yes_no:
                        answer = "yes" if count > 0 else "no"
                        answer_format = "yes-no"
                        reasoning_type = "presence-absence"
                    else:
                        answer = str(count)
                        answer_format = "open-ended"
                        reasoning_type = "counting"
                    
                    segment_frames = [f for f in self.processed_frames if start_time <= f["timestamp"] <= end_time]
                    if segment_frames:
                        avg_blur = sum(f.get("blur_var", 200.0) for f in segment_frames) / len(segment_frames)
                        is_blurred = avg_blur < 80.0
                        avg_brightness = sum(f.get("brightness", 128.0) for f in segment_frames) / len(segment_frames)
                        seg_day_night = "night" if (self.file_lighting == "night" or avg_brightness < 55.0) else "day"
                        seg_visibility = "blurred" if is_blurred else ("dark" if seg_day_night == "night" else "clear")
                    else:
                        seg_day_night = self.file_lighting if self.file_lighting else "day"
                        seg_visibility = "clear"
                    
                    user_qa.append({
                        "Question": q,
                        "Answer": answer,
                        "Answer format": answer_format,
                        "Evidence spans the video": f"{format_time(start_time)} - {format_time(end_time)}",
                        "Reasoning type": reasoning_type,
                        "Difficulty level": "easy" if count <= 2 else "medium",
                        "Visibility quality": seg_visibility,
                        "Day or night tag": seg_day_night,
                        "Trajectory linkage": None,
                        "Unanswerable flag": False
                    })

        # --- 6. Video-wide Aggregate Statistics QA ---
        if "counting" in self.qa_categories:
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

        # --- 7. Grounded segment captions (B1 - B4) ---
        if "user_queries" in self.qa_categories:
            from core.vlm_client import VLMClient
            from core.verifier import verify_caption
            import os
            
            # Instantiate VLM Client
            backend_type = "none"
            api_key = None
            api_url = None
            model_id = None
            
            if self.custom_vlm_url:
                backend_type = "custom_vlm"
                api_url = self.custom_vlm_url
                api_key = self.custom_vlm_key
                model_id = self.custom_vlm_model_id
            elif self.gemini_api_key:
                backend_type = "gemini"
                api_key = self.gemini_api_key
                
            vlm_client = VLMClient(backend_type=backend_type, api_key=api_key, api_url=api_url, model_id=model_id)
            
            # Use a subdirectory keyframes in the video path folder or a temporary folder
            if self.video_path:
                keyframes_dir = os.path.join(os.path.dirname(self.video_path), "keyframes")
            else:
                keyframes_dir = os.path.join("output", "keyframes")
                
            for t_start, t_end in segments:
                t_start_str = format_time(t_start)
                t_end_str = format_time(t_end)
                span_str = f"{t_start_str} - {t_end_str}"
                
                # Get verified counts for this segment
                counts = {}
                ped_res = count_unique_tracks(self.tracked_objects, ["person"], t_start, t_end)
                if ped_res["count"] > 0:
                    counts["person"] = ped_res["count"]
                for cls in VEHICLE_CLASSES:
                    cls_res = count_unique_tracks(self.tracked_objects, [cls], t_start, t_end)
                    if cls_res["count"] > 0:
                        counts[cls] = cls_res["count"]
                        
                # Extract frames
                frame_paths = []
                if self.video_path:
                    try:
                        frame_paths = extract_segment_keyframes(self.video_path, t_start, t_end, keyframes_dir)
                    except Exception as e:
                        print(f"[-] Frame extraction error for segment {span_str}: {e}")
                        
                # Set up prompt context
                avg_brightness = 128.0
                segment_frames = [f for f in self.processed_frames if t_start <= f["timestamp"] <= t_end]
                if segment_frames:
                    avg_brightness = sum(f.get("brightness", 128.0) for f in segment_frames) / len(segment_frames)
                lighting = "night" if (self.file_lighting == "night" or avg_brightness < 55.0) else "day"
                
                context = {
                    "verified_counts": counts,
                    "lighting": lighting,
                    "segment_range": span_str
                }
                
                # Generate caption
                try:
                    caption_res = vlm_client.generate_caption(frame_paths, context)
                except Exception as e:
                    print(f"[-] Caption generation failed for segment {span_str}: {e}")
                    caption_res = {
                        "caption": f"A segment recorded during {lighting} showing: " + ", ".join(f"{v} {k}" for k, v in counts.items()),
                        "claims": {"objects_mentioned": list(counts.keys()), "counts_mentioned": counts}
                    }
                    
                # Cross-check and verify caption
                verified_caption = verify_caption(caption_res, counts, vlm_client, frame_paths, context)
                
                # Add to QA pairs
                qa_by_category["user_queries"].append({
                    "Question": f"Describe what happens in the video segment from {t_start_str} to {t_end_str}.",
                    "Answer": verified_caption.get("caption", ""),
                    "Answer format": "open-ended",
                    "Evidence spans the video": span_str,
                    "Reasoning type": "summary-description",
                    "Difficulty level": "medium",
                    "Visibility quality": "clear" if lighting == "day" else "dark",
                    "Day or night tag": lighting,
                    "Trajectory linkage": None,
                    "Unanswerable flag": False,
                    "_verification_status": verified_caption.get("_verification_status", "auto_verified")
                })

        return qa_by_category

    def generate_qa_pairs(self):
        """Generates QA pairs. Fallback to Custom VLM or Gemini if credentials are available."""
        if self.custom_vlm_url:
            try:
                return self.generate_with_custom_vlm()
            except Exception as e:
                print(f"[-] Custom VLM QA generation failed: {e}. Trying Gemini or falling back to rule-based.")
                
        if self.gemini_api_key:
            try:
                return self.generate_with_gemini()
            except Exception as e:
                print(f"[-] Gemini QA generation failed: {e}. Falling back to rule-based.")
        
        return self.generate_rule_based()

    def generate_with_gemini(self):
        """Calls Google Gemini API to generate clean natural language QA pairs."""
        # Summarize object tracking for prompt context
        summary_counts = {}
        for track in self.tracked_objects.values():
            summary_counts[track.object_type] = summary_counts.get(track.object_type, 0) + 1
            
        prompt = f"""You are a precise annotation assistant for autonomous driving video datasets.
You are given metadata and tracking results for a video.
Generate between 5 to 10 realistic, high-quality question-answer pairs about this video.
Ensure the questions test counting, object presence, spatial positions, and low-light or day/night visibility.

Video filename: {self.filename}
Video duration: {self.duration:.1f} seconds
User Description/Captions: {self.captions or "No captions provided."}
Objects tracked: {json.dumps(summary_counts)}

Please generate QA pairs conforming exactly to the following JSON schema:
[
  {{
    "Question": "Question text in English or German matching the language of User Description",
    "Answer": "Answer text (concise, e.g. '3', 'yes', 'no', 'on the left')",
    "Answer format": "open-ended" or "yes-no",
    "Evidence spans the video": "HH:MM:SS - HH:MM:SS",
    "Reasoning type": "counting", "presence-absence", "spatial-temporal", "low-light-robustness", or "summary-description",
    "Difficulty level": "easy", "medium", or "hard",
    "Visibility quality": "clear", "blurred", or "dark",
    "Day or night tag": "day" or "night",
    "Trajectory linkage": null,
    "Unanswerable flag": false
  }}
]

Output ONLY the raw JSON array. Do not include markdown code block syntax.
"""

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        
        res_data = response.json()
        text_response = res_data["contents"][0]["parts"][0]["text"].strip()
        
        # Clean markdown code blocks if any
        if text_response.startswith("```json"):
            text_response = text_response.split("```json", 1)[1].rsplit("```", 1)[0].strip()
        elif text_response.startswith("```"):
            text_response = text_response.split("```", 1)[1].rsplit("```", 1)[0].strip()
            
        qa_pairs = json.loads(text_response)
        
        # Map generated QAs back into standard categories
        qa_by_category = {cat: [] for cat in self.qa_categories}
        for qa in qa_pairs:
            rtype = qa.get("Reasoning type", "summary-description")
            category_mapping = {
                "counting": "counting",
                "presence-absence": "negative",
                "spatial-temporal": "ambiguity",
                "low-light-robustness": "day_night",
                "summary-description": "user_queries"
            }
            target_cat = category_mapping.get(rtype, "user_queries")
            if target_cat not in qa_by_category:
                qa_by_category[target_cat] = []
            qa_by_category[target_cat].append(qa)
            
        return qa_by_category

    def generate_with_custom_vlm(self):
        """Calls OpenAI-compatible custom VLM API to generate clean natural language QA pairs."""
        summary_counts = {}
        for track in self.tracked_objects.values():
            summary_counts[track.object_type] = summary_counts.get(track.object_type, 0) + 1
            
        prompt = f"""You are a precise annotation assistant for autonomous driving video datasets.
You are given metadata and tracking results for a video.
Generate between 5 to 10 realistic, high-quality question-answer pairs about this video.
Ensure the questions test counting, object presence, spatial positions, and low-light or day/night visibility.

Video filename: {self.filename}
Video duration: {self.duration:.1f} seconds
User Description/Captions: {self.captions or "No captions provided."}
Objects tracked: {json.dumps(summary_counts)}

Please generate QA pairs conforming exactly to the following JSON schema:
[
  {{
    "Question": "Question text in English or German matching the language of User Description",
    "Answer": "Answer text (concise, e.g. '3', 'yes', 'no', 'on the left')",
    "Answer format": "open-ended" or "yes-no",
    "Evidence spans the video": "HH:MM:SS - HH:MM:SS",
    "Reasoning type": "counting", "presence-absence", "spatial-temporal", "low-light-robustness", or "summary-description",
    "Difficulty level": "easy", "medium", or "hard",
    "Visibility quality": "clear", "blurred", or "dark",
    "Day or night tag": "day" or "night",
    "Trajectory linkage": null,
    "Unanswerable flag": false
  }}
]

Output ONLY the raw JSON array. Do not include markdown code block syntax.
"""
        url = self.custom_vlm_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"
            
        headers = {
            "Content-Type": "application/json",
        }
        if self.custom_vlm_key:
            headers["Authorization"] = f"Bearer {self.custom_vlm_key}"
            
        payload = {
            "model": self.custom_vlm_model_id or "gpt-4o",
            "messages": [{"role": "user", "content": prompt}]
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        res_data = response.json()
        text_response = res_data["choices"][0]["message"]["content"].strip()
        
        # Clean markdown code blocks if any
        if text_response.startswith("```json"):
            text_response = text_response.split("```json", 1)[1].rsplit("```", 1)[0].strip()
        elif text_response.startswith("```"):
            text_response = text_response.split("```", 1)[1].rsplit("```", 1)[0].strip()
            
        parsed_data = json.loads(text_response)
        if isinstance(parsed_data, dict) and "qa_pairs" in parsed_data:
            qa_pairs = parsed_data["qa_pairs"]
        elif isinstance(parsed_data, dict):
            qa_pairs = list(parsed_data.values())[0] if isinstance(list(parsed_data.values())[0], list) else []
        else:
            qa_pairs = parsed_data
            
        # Map generated QAs back into standard categories
        qa_by_category = {cat: [] for cat in self.qa_categories}
        for qa in qa_pairs:
            rtype = qa.get("Reasoning type", "summary-description")
            category_mapping = {
                "counting": "counting",
                "presence-absence": "negative",
                "spatial-temporal": "ambiguity",
                "low-light-robustness": "day_night",
                "summary-description": "user_queries"
            }
            target_cat = category_mapping.get(rtype, "user_queries")
            if target_cat not in qa_by_category:
                qa_by_category[target_cat] = []
            qa_by_category[target_cat].append(qa)
            
        return qa_by_category

def extract_segment_keyframes(video_path: str, start_s: float, end_s: float, output_dir: str, max_keyframes: int = 3) -> list[str]:
    """Extracts 2-3 frames from the video within [start_s, end_s] and saves them as JPEGs.
    Returns a list of absolute paths to the saved JPEGs.
    """
    import cv2
    import os
    
    if not video_path or not os.path.exists(video_path):
        return []
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0:
        cap.release()
        return []
        
    start_frame = int(start_s * fps)
    end_frame = int(end_s * fps)
    
    # Bound check
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(0, min(end_frame, total_frames - 1))
    
    if end_frame <= start_frame:
        cap.release()
        return []
        
    step = max(1, (end_frame - start_frame) // max_keyframes)
    frame_paths = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(max_keyframes):
        frame_idx = start_frame + i * step
        if frame_idx > end_frame:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_name = f"frame_{start_s:.1f}_{end_s:.1f}_{i}.jpg"
        frame_path = os.path.join(output_dir, frame_name)
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        
    cap.release()
    return frame_paths

