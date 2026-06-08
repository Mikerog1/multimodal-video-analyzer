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
    h, m, s, ms = value.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
