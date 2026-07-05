def seconds_to_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0

    total_ms = round(seconds * 1000)
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    s = total_seconds % 60
    total_minutes = total_seconds // 60
    m = total_minutes % 60
    h = total_minutes // 60

    return f"{h:02d}:{m:02d}:{s:02d}:{ms:03d}"


def timestamp_to_seconds(value: str) -> float:
    """Convert a timestamp string to seconds.

    Supported formats:
        HH:MM:SS:MS  → e.g. "00:01:30:500"
        HH:MM:SS     → e.g. "00:01:30"
        MM:SS        → e.g. "01:30"
    """
    parts = value.split(":")
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
