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
    """Converts a timestamp string in HH:MM:SS or MM:SS to seconds."""
    parts = ts_str.split(':')
    try:
        if len(parts) == 3:
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
    def __init__(self, filename, processed_frames, duration, qa_categories=None, captions=None, example_questions=None, gemini_api_key=None, custom_vlm_url=None, custom_vlm_key=None, custom_vlm_model_id=None):
        self.filename = filename
        self.processed_frames = processed_frames
        self.duration = duration
        self.qa_categories = qa_categories if qa_categories is not None else ["counting", "negative", "ambiguity", "day_night"]
        self.captions = captions
        self.example_questions = example_questions
        self.gemini_api_key = gemini_api_key
        self.custom_vlm_url = custom_vlm_url
        self.custom_vlm_key = custom_vlm_key
        self.custom_vlm_model_id = custom_vlm_model_id
        
        self.file_lighting = parse_filename_lighting(filename)

    def track_objects(self):
        """Runs a lightweight tracker over the detections across frames."""
        tracks = []
        next_track_id = 0
        active_tracks = []
        
        for frame in self.processed_frames:
            timestamp = frame["timestamp"]
            frame_idx = frame["frame_idx"]
            detections = frame["detections"]
            
            active_tracks = [t for t in active_tracks if timestamp - t["last_seen_time"] <= 3.0]
            
            matched_detections = set()
            matched_tracks = set()
            
            # Phase 1: Match using IoU
            for track in active_tracks:
                best_iou = 0.0
                best_det_idx = -1
                
                for idx, det in enumerate(detections):
                    if idx in matched_detections:
                        continue
                    if det["label"] != track["label"]:
                        continue
                    
                    iou = calculate_iou(track["last_box"], det["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_det_idx = idx
                
                if best_iou >= 0.1 and best_det_idx != -1:
                    matched_detections.add(best_det_idx)
                    track["last_box"] = detections[best_det_idx]["box"]
                    track["last_seen_time"] = timestamp
                    track["seen_frames"].append(frame_idx)
                    track["boxes"].append((frame_idx, timestamp, detections[best_det_idx]["box"]))
                    matched_tracks.add(track["id"])
            
            # Phase 2: Match remaining tracks using proximity/distance
            for track in active_tracks:
                if track["id"] in matched_tracks:
                    continue
                
                best_dist = float('inf')
                best_det_idx = -1
                
                track_center = [
                    (track["last_box"][0] + track["last_box"][2]) / 2,
                    (track["last_box"][1] + track["last_box"][3]) / 2
                ]
                
                for idx, det in enumerate(detections):
                    if idx in matched_detections:
                        continue
                    if det["label"] != track["label"]:
                        continue
                    
                    det_center = [
                        (det["box"][0] + det["box"][2]) / 2,
                        (det["box"][1] + det["box"][3]) / 2
                    ]
                    
                    dist = ((track_center[0] - det_center[0])**2 + (track_center[1] - det_center[1])**2)**0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_det_idx = idx
                
                if best_det_idx != -1 and best_dist < 250:
                    matched_detections.add(best_det_idx)
                    track["last_box"] = detections[best_det_idx]["box"]
                    track["last_seen_time"] = timestamp
                    track["seen_frames"].append(frame_idx)
                    track["boxes"].append((frame_idx, timestamp, detections[best_det_idx]["box"]))
                    matched_tracks.add(track["id"])
            
            # Phase 3: Instantiate new tracks
            for idx, det in enumerate(detections):
                if idx in matched_detections:
                    continue
                
                new_track = {
                    "id": next_track_id,
                    "label": det["label"],
                    "last_box": det["box"],
                    "first_seen_time": timestamp,
                    "last_seen_time": timestamp,
                    "seen_frames": [frame_idx],
                    "boxes": [(frame_idx, timestamp, det["box"])]
                }
                next_track_id += 1
                tracks.append(new_track)
                active_tracks.append(new_track)
                
        return tracks

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

    def generate_rule_based(self):
        """Generates high-quality enhanced rule-based QA pairs."""
        tracks = self.track_objects()
        qa_by_category = {cat: [] for cat in self.qa_categories}
        
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
        for track in tracks:
            for _, _, box in track["boxes"]:
                max_x = max(max_x, box[2])
        
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
                
            segment_tracks = []
            for track in tracks:
                track_boxes_in_seg = [b for b in track["boxes"] if t_start <= b[1] <= t_end]
                if track_boxes_in_seg:
                    segment_tracks.append((track, track_boxes_in_seg))
            
            detected_labels = set(t[0]["label"] for t in segment_tracks)
            
            # Entity groups
            pedestrians = [t for t in segment_tracks if t[0]["label"] == "person"]
            vehicles = [t for t in segment_tracks if t[0]["label"] in VEHICLE_CLASSES]
            dogs = [t for t in segment_tracks if t[0]["label"] == "dog"]
            
            num_pedestrians = len(pedestrians)
            num_vehicles = len(vehicles)
            num_dogs = len(dogs)
            
            # --- 1. Counting QA ---
            if "counting" in self.qa_categories:
                if num_pedestrians > 0:
                    difficulty = "hard" if is_blurred or num_pedestrians >= 4 else ("medium" if num_pedestrians >= 2 else "easy")
                    qa_by_category["counting"].append({
                        "Question": f"How many pedestrians are visible in the video segment from {t_start_str} to {t_end_str}?",
                        "Answer": str(num_pedestrians),
                        "Answer format": "open-ended",
                        "Evidence spans the video": span_str,
                        "Reasoning type": "counting",
                        "Difficulty level": difficulty,
                        "Visibility quality": visibility,
                        "Day or night tag": day_night,
                        "Trajectory linkage": None,
                        "Unanswerable flag": False
                    })
                    
                if num_vehicles > 0:
                    difficulty = "hard" if is_blurred or num_vehicles >= 5 else ("medium" if num_vehicles >= 2 else "easy")
                    qa_by_category["counting"].append({
                        "Question": f"How many vehicles are visible in the video segment from {t_start_str} to {t_end_str}?",
                        "Answer": str(num_vehicles),
                        "Answer format": "open-ended",
                        "Evidence spans the video": span_str,
                        "Reasoning type": "counting",
                        "Difficulty level": difficulty,
                        "Visibility quality": visibility,
                        "Day or night tag": day_night,
                        "Trajectory linkage": None,
                        "Unanswerable flag": False
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
                        "Question": f"Is there any {selected_absent} present in the video segment from {t_start_str} to {t_end_str}?",
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
                for group, name in [(pedestrians, "pedestrian"), (vehicles, "vehicle")]:
                    if len(group) == 1:
                        track_data, boxes = group[0]
                        avg_x = sum((b[2][0] + b[2][2]) / 2 for b in boxes) / len(boxes)
                        side = "left side" if avg_x < (max_x * 0.4) else ("right side" if avg_x > (max_x * 0.6) else "middle")
                        
                        qa_by_category["ambiguity"].append({
                            "Question": f"On which side of the screen is the {name} visible in the segment from {t_start_str} to {t_end_str}?",
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

                # 3b. Temporal Order Query (First Appearance)
                if len(segment_tracks) >= 2:
                    sorted_tracks = sorted(segment_tracks, key=lambda x: x[0]["first_seen_time"])
                    first_label = sorted_tracks[0][0]["label"]
                    second_label = sorted_tracks[1][0]["label"]
                    if first_label != second_label:
                        qa_by_category["ambiguity"].append({
                            "Question": f"Which appears first in the segment {t_start_str} - {t_end_str}: a {first_label} or a {second_label}?",
                            "Answer": f"a {first_label}",
                            "Answer format": "open-ended",
                            "Evidence spans the video": span_str,
                            "Reasoning type": "spatial-temporal",
                            "Difficulty level": "medium",
                            "Visibility quality": visibility,
                            "Day or night tag": day_night,
                            "Trajectory linkage": None,
                            "Unanswerable flag": False
                        })

            # --- 4. Day vs. Night Robustness QA ---
            if "day_night" in self.qa_categories:
                if day_night == "night":
                    target_ref = "vehicle" if num_vehicles > 0 else ("pedestrian" if num_pedestrians > 0 else "road layout")
                    qa_by_category["day_night"].append({
                        "Question": f"Is the {target_ref} clearly visible despite the low lighting in the night segment from {t_start_str} to {t_end_str}?",
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
                    
                    count = 0
                    for track in tracks:
                        if track["label"] in requested_classes:
                            track_boxes_in_seg = [b for b in track["boxes"] if start_time <= b[1] <= end_time]
                            if track_boxes_in_seg:
                                count += 1
                    
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

        return qa_by_category

    def generate_with_gemini(self):
        """Calls Google Gemini API to generate clean natural language QA pairs."""
        tracks = self.track_objects()
        
        # Summarize object tracking for prompt context
        summary_counts = {}
        for track in tracks:
            summary_counts[track["label"]] = summary_counts.get(track["label"], 0) + 1
            
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
        tracks = self.track_objects()
        summary_counts = {}
        for track in tracks:
            summary_counts[track["label"]] = summary_counts.get(track["label"], 0) + 1
            
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

