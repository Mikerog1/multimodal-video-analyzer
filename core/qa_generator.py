import re
import random
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
    # Karlsruhe dataset format: Walk_DDMMYY_HHMMSS
    match = re.search(r'_(\d{2})(\d{2})(\d{2})_', filename)
    if match:
        hour = int(match.group(1))
        # Karlsruhe evening/night run usually after 19:00 or before 07:00
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
    def __init__(self, filename, processed_frames, duration, qa_categories=None, captions=None, example_questions=None):
        self.filename = filename
        self.processed_frames = processed_frames
        self.duration = duration
        self.qa_categories = qa_categories if qa_categories is not None else ["counting", "negative", "ambiguity", "day_night"]
        self.captions = captions
        self.example_questions = example_questions
        
        # Determine global day/night from filename
        self.file_lighting = parse_filename_lighting(filename)

    def track_objects(self):
        """Runs a lightweight tracker over the detections across frames."""
        tracks = []
        next_track_id = 0
        active_tracks = []
        
        # We group similar classes into target tracked entities
        # e.g., 'person' -> 'pedestrian', and vehicles -> 'vehicle'
        # but we preserve the raw classes for checking tracking labels.
        for frame in self.processed_frames:
            timestamp = frame["timestamp"]
            frame_idx = frame["frame_idx"]
            detections = frame["detections"]
            
            # Remove inactive tracks (not seen for more than 3 seconds)
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
            
            # Phase 2: Match remaining tracks using proximity/distance (for fast motion/low FPS sampling)
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
                
                # If distance is under a reasonable threshold (e.g., 250px)
                if best_det_idx != -1 and best_dist < 250:
                    matched_detections.add(best_det_idx)
                    track["last_box"] = detections[best_det_idx]["box"]
                    track["last_seen_time"] = timestamp
                    track["seen_frames"].append(frame_idx)
                    track["boxes"].append((frame_idx, timestamp, detections[best_det_idx]["box"]))
                    matched_tracks.add(track["id"])
            
            # Phase 3: Instantiate new tracks for unmatched detections
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
        """Generates QA pairs according to the specified categories and schema.
        
        Returns:
            A dict keyed by category name, each value being a list of QA pair dicts.
            Example: {"counting": [...], "negative": [...], "ambiguity": [...], "day_night": [...]}
        """
        tracks = self.track_objects()
        qa_by_category = {cat: [] for cat in self.qa_categories}
        
        # Segment the video into 10-second intervals
        segment_duration = 10.0
        segments = []
        t = 0.0
        while t < self.duration:
            t_end = min(t + segment_duration, self.duration)
            if t_end - t >= 2.0:  # Ignore tiny trailing segments less than 2s
                segments.append((t, t_end))
            t += segment_duration
            
        if not segments and self.duration > 0:
            segments.append((0.0, self.duration))
            
        
        for t_start, t_end in segments:
            t_start_str = format_time(t_start)
            t_end_str = format_time(t_end)
            span_str = f"{t_start_str} - {t_end_str}"
            
            # Filter frame data in segment
            segment_frames = [f for f in self.processed_frames if t_start <= f["timestamp"] <= t_end]
            if not segment_frames:
                continue
                
            # Compute blur (average Laplacian variance)
            avg_blur = sum(f.get("blur_var", 200.0) for f in segment_frames) / len(segment_frames)
            is_blurred = avg_blur < 80.0  # Threshold under 80 indicates significant blur/motion
            
            # Compute day/night tag
            avg_brightness = sum(f.get("brightness", 128.0) for f in segment_frames) / len(segment_frames)
            # If filename override says night OR average brightness is very low
            day_night = "night" if (self.file_lighting == "night" or avg_brightness < 55.0) else "day"
            
            # Set default visibility quality
            if is_blurred:
                visibility = "blurred"
            elif day_night == "night":
                visibility = "dark"
            else:
                visibility = "clear"
                
            # Filter tracks active in this segment
            segment_tracks = []
            for track in tracks:
                # Track is active in segment if it has boxes within the segment time boundaries
                track_boxes_in_seg = [b for b in track["boxes"] if t_start <= b[1] <= t_end]
                if track_boxes_in_seg:
                    segment_tracks.append((track, len(track_boxes_in_seg)))
            
            # Retrieve active class counts
            detected_labels = set(t[0]["label"] for t in segment_tracks)
            
            # Track count statistics
            pedestrian_tracks = [t[0] for t in segment_tracks if t[0]["label"] == "person"]
            vehicle_classes = {"car", "truck", "bus", "motorcycle", "bicycle"}
            vehicle_tracks = [t[0] for t in segment_tracks if t[0]["label"] in vehicle_classes]
            dog_tracks = [t[0] for t in segment_tracks if t[0]["label"] == "dog"]
            
            num_pedestrians = len(pedestrian_tracks)
            num_vehicles = len(vehicle_tracks)
            num_dogs = len(dog_tracks)
            
            # --- 1. Counting QA ---
            # Generate Counting questions if entities exist in the segment
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
            # Sample objects from the vocabulary that are NOT present in this segment
            # Normalize vocabulary labels to compare with detected ones (e.g. stop sign -> traffic sign)
            # COCO detections: traffic light, bicycle, motorcycle, bus, truck, stop sign, dog, cat, fire hydrant, bench.
            # We construct a matching check:
            if "negative" in self.qa_categories:
                detected_mapped = set()
                for lbl in detected_labels:
                    if lbl in ["car", "truck", "bus", "motorcycle", "bicycle"]:
                        detected_mapped.add("vehicle")
                    if lbl in ["traffic light"]:
                        detected_mapped.add("traffic light")
                    if lbl in ["stop sign"]:
                        detected_mapped.add("traffic sign")
                    if lbl in ["dog"]:
                        detected_mapped.add("dog")
                    if lbl in ["cat"]:
                        detected_mapped.add("cat")
                    if lbl in ["fire hydrant"]:
                        detected_mapped.add("fire hydrant")
                    if lbl in ["bench"]:
                        detected_mapped.add("bench")
                
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
                    # Select an item deterministically based on timestamp to avoid random shifts on rerun
                    seed_idx = int(t_start * 100) % len(absent_candidates)
                    selected_absent = absent_candidates[seed_idx]
                    qa_by_category["negative"].append({
                        "Question": f"Is there any {selected_absent} present in the video segment from {t_start_str} to {t_end_str}?",
                        "Answer": "no",
                        "Answer format": "yes-no",
                        "Evidence spans the video": span_str,
                        "Reasoning type": "negative-absence",
                        "Difficulty level": "easy",
                        "Visibility quality": visibility,
                        "Day or night tag": day_night,
                        "Trajectory linkage": None,
                        "Unanswerable flag": False
                    })

            # --- 3. Ambiguity-Aware QA ---
            # If the segment has high motion blur/ego-motion, flag it and create a question
            if "ambiguity" in self.qa_categories:
                if is_blurred:
                    # Pick a random candidate class that was detected or query generally
                    target_object = "street details"
                    if num_pedestrians > 0:
                        target_object = "pedestrian"
                    elif num_vehicles > 0:
                        target_object = "vehicle"
                    elif num_dogs > 0:
                        target_object = "dog"
                        
                    qa_by_category["ambiguity"].append({
                        "Question": f"Are the details of the {target_object} in the segment from {t_start_str} to {t_end_str} clearly readable, or too blurred to identify?",
                        "Answer": "too blurred to identify",
                        "Answer format": "open-ended",
                        "Evidence spans the video": span_str,
                        "Reasoning type": "spatial-temporal",
                        "Difficulty level": "hard",
                        "Visibility quality": "blurred",
                        "Day or night tag": day_night,
                        "Trajectory linkage": None,
                        "Unanswerable flag": True
                    })

            # --- 4. Day vs. Night Robustness QA ---
            # If low-light / night run, explicitly query low light perception
            if "day_night" in self.qa_categories:
                if day_night == "night":
                    # Check for a pedestrian or vehicle to query low light visibility
                    if num_vehicles > 0:
                        qa_by_category["day_night"].append({
                            "Question": f"Is the vehicle still visible in this night segment from {t_start_str} to {t_end_str}?",
                            "Answer": "yes",
                            "Answer format": "yes-no",
                            "Evidence spans the video": span_str,
                            "Reasoning type": "low-light-robustness",
                            "Difficulty level": "medium",
                            "Visibility quality": "dark",
                            "Day or night tag": "night",
                            "Trajectory linkage": None,
                            "Unanswerable flag": False
                        })
                    elif num_pedestrians > 0:
                        qa_by_category["day_night"].append({
                            "Question": f"Is the crossing pedestrian visible in this night segment from {t_start_str} to {t_end_str}?",
                            "Answer": "yes",
                            "Answer format": "yes-no",
                            "Evidence spans the video": span_str,
                            "Reasoning type": "low-light-robustness",
                            "Difficulty level": "medium",
                            "Visibility quality": "dark",
                            "Day or night tag": "night",
                            "Trajectory linkage": None,
                            "Unanswerable flag": False
                        })
                    else:
                        # Ask about pedestrian absence under dark settings
                        qa_by_category["day_night"].append({
                            "Question": f"Are there any pedestrians visible in this dark night segment from {t_start_str} to {t_end_str}?",
                            "Answer": "no",
                            "Answer format": "yes-no",
                            "Evidence spans the video": span_str,
                            "Reasoning type": "low-light-robustness",
                            "Difficulty level": "medium",
                            "Visibility quality": "dark",
                            "Day or night tag": "night",
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
                # Parse and evaluate each question
                questions_list = [q.strip() for q in self.example_questions.split('\n') if q.strip()]
                for q in questions_list:
                    q_lower = q.lower()
                    
                    # 1. Parse timestamps
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
                    
                    # Clamp start and end times to video duration
                    start_time = max(0.0, min(start_time, self.duration))
                    end_time = max(0.0, min(end_time, self.duration))
                    if start_time > end_time:
                        start_time, end_time = end_time, start_time
                    
                    # 2. Parse target classes
                    requested_classes = []
                    if "vehicle" in q_lower or "vehicles" in q_lower:
                        requested_classes = VEHICLE_CLASSES
                    else:
                        for cls, synonyms in CLASS_SYNONYMS.items():
                            if any(re.search(r'\b' + re.escape(syn) + r'\b', q_lower) for syn in synonyms):
                                requested_classes.append(cls)
                                if cls == "traffic sign":
                                    requested_classes.append("stop sign")
                    
                    # 3. Count matching objects in timeframe
                    count = 0
                    for track in tracks:
                        if track["label"] in requested_classes:
                            track_boxes_in_seg = [b for b in track["boxes"] if start_time <= b[1] <= end_time]
                            if track_boxes_in_seg:
                                count += 1
                    
                    # 4. Classify query type (count vs presence)
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
                    
                    # Compute segment properties for metadata
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
