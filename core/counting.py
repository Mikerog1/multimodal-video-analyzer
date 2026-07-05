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
            - "low_confidence_track_ids": list of int
    """
    counted_track_ids = []
    short_lived_track_ids = []
    low_confidence_track_ids = []
    
    # Lowercase target classes for robust comparison
    target_classes_lower = [c.lower() for c in target_classes]
    
    segment_duration = end_s - start_s
    
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
                
        if not observations_in_seg:
            continue
            
        counted_track_ids.append(track_id)
        num_obs = len(observations_in_seg)
        
        # --- Confidence heuristics ---
        
        # 1. Short-lived: track has < 3 observations in a segment that is >= 3s long
        if num_obs < 3 and segment_duration >= 3.0:
            short_lived_track_ids.append(track_id)
            continue
        
        # 2. Low average detection confidence
        avg_conf = sum(o[1] for o in observations_in_seg) / num_obs
        if avg_conf < 0.3:
            low_confidence_track_ids.append(track_id)
            continue
            
        # 3. Boundary check: only flag if the track appears at the edge of the
        #    segment AND is very brief (< 2s screen time in segment AND < 5 obs).
        #    A track that spans the entire segment naturally touches the boundaries
        #    and should NOT be flagged.
        obs_times = [o[0] for o in observations_in_seg]
        min_time = min(obs_times)
        max_time = max(obs_times)
        track_span_in_seg = max_time - min_time
        
        at_start_boundary = abs(min_time - start_s) < 1.0
        at_end_boundary = abs(max_time - end_s) < 1.0
        
        if (at_start_boundary or at_end_boundary) and track_span_in_seg < 2.0 and num_obs < 5:
            low_confidence_track_ids.append(track_id)
    
    # Overall confidence signal for this segment count
    confidence_signal = "high"
    if short_lived_track_ids or low_confidence_track_ids:
        confidence_signal = "low"
                
    return {
        "track_ids": counted_track_ids,
        "count": len(counted_track_ids),
        "confidence_signal": confidence_signal,
        "short_lived_track_ids": short_lived_track_ids,
        "low_confidence_track_ids": low_confidence_track_ids,
    }


def aggregate_video_stats(tracked_objects: dict, duration_seconds: float) -> dict:
    """
    Computes video-wide statistics using global track IDs.
    
    Uses a sweep-line approach for concurrent-count computation
    instead of the previous O(N²) brute force.
    
    Returns a dict:
        - "unique_total_per_class": dict of class_name -> unique count
        - "peak_concurrent_per_class": dict of class_name -> peak concurrent count
        - "avg_concurrent_per_class": dict of class_name -> average concurrent count
    """
    unique_total_per_class = {}
    peak_concurrent_per_class = {}
    avg_concurrent_per_class = {}
    
    # Group tracks by class
    tracks_by_class: dict[str, list] = {}
    for track_id, track in tracked_objects.items():
        cls = track.object_type
        if cls not in tracks_by_class:
            tracks_by_class[cls] = []
        tracks_by_class[cls].append(track)
        
    for cls, tracks in tracks_by_class.items():
        unique_total_per_class[cls] = len(tracks)
        
        # Build intervals [first_obs_time, last_obs_time] per track
        # using a sweep-line to compute concurrent counts efficiently.
        events: list[tuple[float, int]] = []  # (time, +1 or -1)
        
        for track in tracks:
            obs_times = []
            for obs in track.bbox_observations:
                try:
                    obs_times.append(timestamp_to_seconds(obs.timestamp))
                except Exception:
                    pass
            if not obs_times:
                continue
            t_start = min(obs_times)
            t_end = max(obs_times)
            events.append((t_start, +1))
            events.append((t_end + 0.01, -1))  # small epsilon so end is inclusive
        
        if not events:
            peak_concurrent_per_class[cls] = 0
            avg_concurrent_per_class[cls] = 0.0
            continue
            
        # Sort events: by time, then -1 before +1 at same time (end before start)
        events.sort(key=lambda e: (e[0], e[1]))
        
        # Sweep-line
        current_count = 0
        peak = 0
        weighted_sum = 0.0
        prev_time = events[0][0]
        
        for time_val, delta in events:
            if time_val > prev_time:
                weighted_sum += current_count * (time_val - prev_time)
            current_count += delta
            peak = max(peak, current_count)
            prev_time = time_val
            
        total_span = events[-1][0] - events[0][0]
        
        peak_concurrent_per_class[cls] = peak
        avg_concurrent_per_class[cls] = round(weighted_sum / total_span, 2) if total_span > 0 else float(peak)
        
    return {
        "unique_total_per_class": unique_total_per_class,
        "peak_concurrent_per_class": peak_concurrent_per_class,
        "avg_concurrent_per_class": avg_concurrent_per_class,
    }
