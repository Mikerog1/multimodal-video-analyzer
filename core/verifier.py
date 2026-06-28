import re
from core.counting import count_unique_tracks, aggregate_video_stats
from utils.time_utils import timestamp_to_seconds

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

def parse_time_span(span_str: str) -> tuple[float, float]:
    """Parses a time span string like '00:00:00 - 00:00:10' into (start_s, end_s)."""
    parts = span_str.split("-")
    if len(parts) != 2:
        return 0.0, 0.0
    
    def parse_time(ts_str: str) -> float:
        ts_str = ts_str.strip()
        sub_parts = ts_str.split(":")
        try:
            if len(sub_parts) == 4:
                return int(sub_parts[0]) * 3600 + int(sub_parts[1]) * 60 + int(sub_parts[2]) + int(sub_parts[3]) / 1000.0
            elif len(sub_parts) == 3:
                return int(sub_parts[0]) * 3600 + int(sub_parts[1]) * 60 + float(sub_parts[2])
            elif len(sub_parts) == 2:
                return int(sub_parts[0]) * 60 + float(sub_parts[1])
        except ValueError:
            pass
        return 0.0
        
    return parse_time(parts[0]), parse_time(parts[1])

def detect_target_classes(question_text: str) -> list[str]:
    """Determines target classes from the question text using CLASS_SYNONYMS."""
    q_lower = question_text.lower()
    
    if "vehicle" in q_lower or "vehicles" in q_lower:
        return VEHICLE_CLASSES
        
    target_classes = []
    for cls, synonyms in CLASS_SYNONYMS.items():
        if any(re.search(r'\b' + re.escape(syn) + r'\b', q_lower) for syn in synonyms):
            target_classes.append(cls)
            
    return target_classes

def verify_counting_qa(qa_item: dict, tracked_objects: dict) -> dict:
    """
    Recomputes the count from tracked_objects for a counting QA item,
    and updates/auto-corrects its Answer if a mismatch is found.
    Adds internal metadata fields:
        _target_class
        _segment_start_s
        _segment_end_s
        _track_ids_counted
        _confidence_signal
        _verification_status
    """
    # Only verify counting items or summary-description items that are actually counts
    reasoning_type = qa_item.get("Reasoning type", qa_item.get("reasoning_type", ""))
    if reasoning_type != "counting":
        return qa_item
        
    question = qa_item.get("Question", qa_item.get("question", ""))
    evidence_span = qa_item.get("Evidence spans the video", qa_item.get("evidence_spans_the_video", ""))
    
    if not question or not evidence_span:
        return qa_item
        
    start_s, end_s = parse_time_span(evidence_span)
    target_classes = detect_target_classes(question)
    
    if not target_classes:
        return qa_item
        
    # Recompute counts using core/counting.py
    counting_result = count_unique_tracks(tracked_objects, target_classes, start_s, end_s)
    recomputed_count = counting_result["count"]
    confidence_signal = counting_result["confidence_signal"]
    track_ids = counting_result["track_ids"]
    
    # Store internal metadata fields
    qa_item["_target_class"] = ",".join(target_classes)
    qa_item["_segment_start_s"] = start_s
    qa_item["_segment_end_s"] = end_s
    qa_item["_track_ids_counted"] = track_ids
    qa_item["_confidence_signal"] = confidence_signal
    
    # Calibrate answer format (A5)
    # If confidence is low: "at least N"
    # If confidence is high: "N"
    if confidence_signal == "low":
        expected_answer = f"at least {recomputed_count}"
    else:
        expected_answer = str(recomputed_count)
        
    current_answer = str(qa_item.get("Answer", qa_item.get("answer", ""))).strip()
    
    if current_answer != expected_answer:
        qa_item["_original_answer"] = current_answer
        qa_item["Answer"] = expected_answer
        # Also handle lowercase key if present
        if "answer" in qa_item:
            qa_item["answer"] = expected_answer
        qa_item["_verification_status"] = "auto_corrected"
    else:
        qa_item["_verification_status"] = "auto_verified"
        
    # If confidence was flagged as low or verification has warnings, we can also mark status as review needed
    if confidence_signal == "low":
        qa_item["_verification_status"] = "flagged_for_review"
        
    return qa_item

def verify_all_qa(qa_by_category: dict, tracked_objects: dict) -> dict:
    """Runs verifier on all QA items in the dictionary."""
    verified_qa = {}
    for category, qa_list in qa_by_category.items():
        verified_list = []
        for qa in qa_list:
            verified_qa_item = verify_counting_qa(qa, tracked_objects)
            verified_list.append(verified_qa_item)
        verified_qa[category] = verified_list
    return verified_qa

def verify_caption(caption_item: dict, verified_counts: dict, vlm_client=None, frame_paths=None, context=None) -> dict:
    """
    Cross-checks caption claims against verified counting QA counts.
    If caption counts mismatch, it attempts to regenerate (max 2 retries).
    If it still mismatches, it flags the status as flagged_for_review.
    """
    caption = caption_item.get("caption", "")
    claims = caption_item.get("claims", {})
    counts_mentioned = claims.get("counts_mentioned", {})
    
    # Normalize counts_mentioned keys to standard classes using CLASS_SYNONYMS
    normalized_mentioned = {}
    for key, count in counts_mentioned.items():
        matched_cls = None
        key_lower = key.lower()
        if "vehicle" in key_lower or key_lower in VEHICLE_CLASSES:
            matched_cls = "vehicle"
        else:
            for base_cls, synonyms in CLASS_SYNONYMS.items():
                if key_lower == base_cls or key_lower in synonyms:
                    if base_cls in VEHICLE_CLASSES:
                        matched_cls = "vehicle"
                    else:
                        matched_cls = base_cls
                    break
        if matched_cls:
            try:
                normalized_mentioned[matched_cls] = int(count)
            except ValueError:
                pass
            
    # Normalize verified_counts keys to standard classes
    normalized_verified = {}
    for cls, count in verified_counts.items():
        cls_lower = cls.lower()
        if cls_lower in VEHICLE_CLASSES:
            normalized_verified["vehicle"] = normalized_verified.get("vehicle", 0) + int(count)
        else:
            normalized_verified[cls_lower] = int(count)
            
    # Check for contradictions
    mismatch = False
    mismatched_details = []
    for cls, count in normalized_mentioned.items():
        verified_count = normalized_verified.get(cls, 0)
        if count != verified_count:
            mismatch = True
            mismatched_details.append(f"Caption claims {count} {cls}s, but verified count is {verified_count}.")
            
    if not mismatch:
        caption_item["_verification_status"] = "auto_verified"
        return caption_item
        
    # Attempt regeneration if VLM client, frame paths, and context are provided
    if vlm_client and frame_paths and context:
        print(f"[!] Mismatch found in caption: {mismatched_details}. Attempting regeneration...")
        for attempt in range(2):
            try:
                # Add correction instruction to prompt context
                original_prompt = context.get("prompt_instruction", "")
                context["prompt_instruction"] = original_prompt + f"\nCORRECTION: In your previous attempt, your caption claims were inconsistent: {', '.join(mismatched_details)}. You MUST write a description that strictly matches the verified counts: {verified_counts}."
                
                regenerated_item = vlm_client.generate_caption(frame_paths, context)
                
                # Check the regenerated item
                reg_claims = regenerated_item.get("claims", {})
                reg_counts = reg_claims.get("counts_mentioned", {})
                
                reg_normalized_mentioned = {}
                for key, count in reg_counts.items():
                    matched_cls = None
                    key_lower = key.lower()
                    if "vehicle" in key_lower or key_lower in VEHICLE_CLASSES:
                        matched_cls = "vehicle"
                    else:
                        for base_cls, synonyms in CLASS_SYNONYMS.items():
                            if key_lower == base_cls or key_lower in synonyms:
                                if base_cls in VEHICLE_CLASSES:
                                    matched_cls = "vehicle"
                                else:
                                    matched_cls = base_cls
                                break
                    if matched_cls:
                        try:
                            reg_normalized_mentioned[matched_cls] = int(count)
                        except ValueError:
                            pass
                            
                reg_mismatch = False
                for cls, count in reg_normalized_mentioned.items():
                    verified_count = normalized_verified.get(cls, 0)
                    if count != verified_count:
                        reg_mismatch = True
                        break
                        
                if not reg_mismatch:
                    print(f"[+] Caption regenerated successfully on attempt {attempt + 1}.")
                    regenerated_item["_verification_status"] = "auto_corrected"
                    return regenerated_item
            except Exception as e:
                print(f"[-] Regeneration attempt {attempt + 1} failed: {e}")
                
    # If we get here, mismatch persists or we couldn't regenerate
    caption_item["_verification_status"] = "flagged_for_review"
    return caption_item
