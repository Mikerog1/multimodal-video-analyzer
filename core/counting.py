import sys
import os
from utils.time_utils import timestamp_to_seconds

def count_unique_tracks(tracked_objects: dict, target_classes: list[str], start_s: float, end_s: float) -> dict:
    """
    Count unique tracks of given classes visible in [start_s, end_s].
    
    Returns:
        A dict with keys:
            - "track_ids": list of int (unique track IDs counted)
            - "count": int (number of unique tracks)
            - "confidence_signal": "high" | "low"
            - "short_lived_track_ids": list of int
    """
    counted_track_ids = []
    short_lived_track_ids = []
    
    # Lowercase target classes for robust comparison
    target_classes_lower = [c.lower() for c in target_classes]
    
    for track_id, track in tracked_objects.items():
        # Check if the track's class matches
        if track.object_type.lower() not in target_classes_lower:
            continue
            
        # Find observations within the time window [start_s, end_s]
        observations_in_seg = []
        for obs in track.bbox_observations:
            try:
                obs_time = timestamp_to_seconds(obs.timestamp)
            except Exception:
                continue
            if start_s <= obs_time <= end_s:
                observations_in_seg.append((obs_time, obs.confidence))
                
        if observations_in_seg:
            counted_track_ids.append(track_id)
            
            # Confidence Heuristic:
            # - Short-lived: track has < 3 observations in this segment (when segment is at least 3 seconds long)
            # - ID-switch symptom: track starts/ends near segment boundaries (within 1 second of boundaries)
            # - Low detection confidence: average confidence of observations in segment < 0.3
            is_suspicious = False
            
            segment_duration = end_s - start_s
            num_obs = len(observations_in_seg)
            
            if num_obs < 3 and segment_duration >= 3.0:
                is_suspicious = True
                short_lived_track_ids.append(track_id)
            else:
                avg_conf = sum(o[1] for o in observations_in_seg) / num_obs
                if avg_conf < 0.3:
                    is_suspicious = True
                    
                # Boundary checks
                obs_times = [o[0] for o in observations_in_seg]
                min_time = min(obs_times)
                max_time = max(obs_times)
                if abs(min_time - start_s) < 1.0 or abs(max_time - end_s) < 1.0:
                    # If it's a short track at the boundary, flag it
                    if num_obs < 5:
                        is_suspicious = True
            
    # Compute overall confidence signal for this segment count
    # If any counted track is marked suspicious, or if we have high density (e.g. > 5 objects in a segment),
    # we might flag the signal as low confidence. Let's make it simple: if there are suspicious/short-lived tracks,
    # or if any track in segment has average confidence < 0.4.
    confidence_signal = "high"
    if short_lived_track_ids:
        confidence_signal = "low"
    else:
        # Check if any counted track has very low confidence in this segment
        for track_id in counted_track_ids:
            track = tracked_objects[track_id]
            obs_confs = [obs.confidence for obs in track.bbox_observations]
            if obs_confs and (sum(obs_confs) / len(obs_confs)) < 0.4:
                confidence_signal = "low"
                break
                
    return {
        "track_ids": counted_track_ids,
        "count": len(counted_track_ids),
        "confidence_signal": confidence_signal,
        "short_lived_track_ids": short_lived_track_ids
    }

def aggregate_video_stats(tracked_objects: dict, duration_seconds: float) -> dict:
    """
    Computes video-wide statistics using global track IDs.
    
    Returns a dict:
        - "unique_total_per_class": dict of class_name -> unique count
        - "peak_concurrent_per_class": dict of class_name -> peak concurrent count
        - "avg_concurrent_per_class": dict of class_name -> average concurrent count
    """
    unique_total_per_class = {}
    peak_concurrent_per_class = {}
    avg_concurrent_per_class = {}
    
    # Group tracks by class
    tracks_by_class = {}
    for track_id, track in tracked_objects.items():
        cls = track.object_type
        if cls not in tracks_by_class:
            tracks_by_class[cls] = []
        tracks_by_class[cls].append(track)
        
    for cls, tracks in tracks_by_class.items():
        unique_total_per_class[cls] = len(tracks)
        
        # To compute peak and average concurrent, we need to gather all timestamps where any track of this class was observed
        all_timestamps = set()
        for track in tracks:
            for obs in track.bbox_observations:
                try:
                    all_timestamps.add(timestamp_to_seconds(obs.timestamp))
                except Exception:
                    pass
                    
        sorted_times = sorted(list(all_timestamps))
        
        if not sorted_times:
            peak_concurrent_per_class[cls] = 0
            avg_concurrent_per_class[cls] = 0.0
            continue
            
        # For each timestamp, count how many tracks were active (observed within a window of, say, 1.0 second around the timestamp)
        concurrent_counts = []
        for t in sorted_times:
            active_count = 0
            for track in tracks:
                # Is track active at time t?
                # A track is active at t if it has an observation close to t (e.g. within 1.0 second)
                has_close_obs = False
                for obs in track.bbox_observations:
                    try:
                        obs_time = timestamp_to_seconds(obs.timestamp)
                        if abs(obs_time - t) <= 0.5: # 0.5s window
                            has_close_obs = True
                            break
                    except Exception:
                        pass
                if has_close_obs:
                    active_count += 1
            concurrent_counts.append(active_count)
            
        peak_concurrent_per_class[cls] = max(concurrent_counts) if concurrent_counts else 0
        avg_concurrent_per_class[cls] = round(sum(concurrent_counts) / len(concurrent_counts), 2) if concurrent_counts else 0.0
        
    return {
        "unique_total_per_class": unique_total_per_class,
        "peak_concurrent_per_class": peak_concurrent_per_class,
        "avg_concurrent_per_class": avg_concurrent_per_class
    }
