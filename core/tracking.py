from dataclasses import dataclass, field

from utils.time_utils import seconds_to_timestamp


TRACKER = "bytetrack.yaml"
YOLO_CLASS_TO_OBJECT_TYPE = {
    "person": "person",
    "car": "car",
    "truck": "truck",
    "bicycle": "bicycle",
    "boat": "boat/ship",
    "airplane": "plane",
    "dog": "dog",
    "cat": "cat",
    "bird": "bird",
}


@dataclass
class BBoxObservation:
    timestamp: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


@dataclass
class TrackedObject:
    object_type: str
    first_time_seen_seconds: float
    screen_time_seconds: float = 0.0
    bbox_observations: list[BBoxObservation] = field(default_factory=list)


@dataclass
class DebugDetection:
    object_id: int
    object_type: str
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    def to_overlay_detection(self) -> dict:
        return {
            "label": self.object_type,
            "score": self.confidence,
            "box": [self.x1, self.y1, self.x2, self.y2],
            "object_id": self.object_id,
        }


def update_tracked_objects(
    boxes,
    model_names: dict[int, str],
    current_time_seconds: float,
    screen_time_increment: float,
    tracked_objects: dict[int, TrackedObject],
) -> list[DebugDetection]:
    detections: list[DebugDetection] = []
    if boxes is None or boxes.id is None:
        return detections

    for xyxy, track_id, class_id, confidence in zip(
        boxes.xyxy,
        boxes.id,
        boxes.cls,
        boxes.conf,
    ):
        class_name = model_names[int(class_id)]
        object_type = YOLO_CLASS_TO_OBJECT_TYPE.get(class_name)
        if object_type is None:
            continue

        object_id = int(track_id)
        if object_id not in tracked_objects:
            tracked_objects[object_id] = TrackedObject(
                object_type=object_type,
                first_time_seen_seconds=current_time_seconds,
            )

        x1, y1, x2, y2 = xyxy.tolist()
        tracked_object = tracked_objects[object_id]
        tracked_object.screen_time_seconds += screen_time_increment
        tracked_object.bbox_observations.append(
            BBoxObservation(
                timestamp=seconds_to_timestamp(current_time_seconds),
                x1=round(float(x1), 2),
                y1=round(float(y1), 2),
                x2=round(float(x2), 2),
                y2=round(float(y2), 2),
                confidence=round(float(confidence), 4),
            )
        )
        detections.append(
            DebugDetection(
                object_id=object_id,
                object_type=object_type,
                x1=int(round(float(x1))),
                y1=int(round(float(y1))),
                x2=int(round(float(x2))),
                y2=int(round(float(y2))),
                confidence=round(float(confidence), 4),
            )
        )

    return detections
