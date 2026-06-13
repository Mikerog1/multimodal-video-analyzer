from dataclasses import dataclass, field

from utils.time_utils import seconds_to_timestamp


TRACKER = "bytetrack.yaml"
CLASS_TO_OBJECT_TYPE = {
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
        object_type = CLASS_TO_OBJECT_TYPE.get(class_name)
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


def _compute_iou(box_a: list, box_b: list) -> float:
    """Compute Intersection over Union between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class SimpleTracker:
    """Lightweight IoU-based frame-by-frame tracker for detectors without built-in tracking.

    Produces the same TrackedObject / DebugDetection output as the YOLO ByteTrack
    path so all downstream code (CSV, JSON, QA, overlay) works identically.
    """

    def __init__(self, iou_threshold: float = 0.15, max_age_seconds: float = 3.0,
                 distance_threshold: float = 200.0):
        self._next_id = 0
        self._iou_threshold = iou_threshold
        self._max_age_seconds = max_age_seconds
        self._distance_threshold = distance_threshold
        # Active track state: id -> {"box": [x1,y1,x2,y2], "object_type": str, "last_seen": float}
        self._active: dict[int, dict] = {}

    def update(
        self,
        raw_detections: list[dict],
        current_time_seconds: float,
        screen_time_increment: float,
        tracked_objects: dict[int, TrackedObject],
    ) -> list[DebugDetection]:
        """Match raw detections to existing tracks and update tracked_objects.

        Args:
            raw_detections: List of {"label": str, "score": float, "box": [x1,y1,x2,y2]}.
            current_time_seconds: Current timestamp in the video.
            screen_time_increment: Duration to add to screen time for matched objects.
            tracked_objects: Shared dict of TrackedObject, keyed by track ID.

        Returns:
            List of DebugDetection for the current frame.
        """
        # Prune stale tracks
        stale_ids = [
            tid for tid, info in self._active.items()
            if current_time_seconds - info["last_seen"] > self._max_age_seconds
        ]
        for tid in stale_ids:
            del self._active[tid]

        debug_detections: list[DebugDetection] = []
        matched_det_indices: set[int] = set()
        matched_track_ids: set[int] = set()

        # Phase 1: IoU matching (same class required)
        for track_id, track_info in self._active.items():
            best_iou = 0.0
            best_idx = -1
            for idx, det in enumerate(raw_detections):
                if idx in matched_det_indices:
                    continue
                obj_type = CLASS_TO_OBJECT_TYPE.get(det["label"])
                if obj_type != track_info["object_type"]:
                    continue
                iou = _compute_iou(track_info["box"], det["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= self._iou_threshold and best_idx != -1:
                self._apply_match(
                    track_id, raw_detections[best_idx], current_time_seconds,
                    screen_time_increment, tracked_objects, debug_detections,
                )
                matched_det_indices.add(best_idx)
                matched_track_ids.add(track_id)

        # Phase 2: Distance fallback for unmatched tracks (handles fast motion / low FPS)
        for track_id, track_info in self._active.items():
            if track_id in matched_track_ids:
                continue
            tc = [(track_info["box"][0] + track_info["box"][2]) / 2,
                  (track_info["box"][1] + track_info["box"][3]) / 2]
            best_dist = float("inf")
            best_idx = -1
            for idx, det in enumerate(raw_detections):
                if idx in matched_det_indices:
                    continue
                obj_type = CLASS_TO_OBJECT_TYPE.get(det["label"])
                if obj_type != track_info["object_type"]:
                    continue
                dc = [(det["box"][0] + det["box"][2]) / 2,
                      (det["box"][1] + det["box"][3]) / 2]
                dist = ((tc[0] - dc[0]) ** 2 + (tc[1] - dc[1]) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx != -1 and best_dist < self._distance_threshold:
                self._apply_match(
                    track_id, raw_detections[best_idx], current_time_seconds,
                    screen_time_increment, tracked_objects, debug_detections,
                )
                matched_det_indices.add(best_idx)
                matched_track_ids.add(track_id)

        # Phase 3: Create new tracks for unmatched detections
        for idx, det in enumerate(raw_detections):
            if idx in matched_det_indices:
                continue
            object_type = CLASS_TO_OBJECT_TYPE.get(det["label"])
            if object_type is None:
                continue

            track_id = self._next_id
            self._next_id += 1

            self._active[track_id] = {
                "box": det["box"],
                "object_type": object_type,
                "last_seen": current_time_seconds,
            }
            tracked_objects[track_id] = TrackedObject(
                object_type=object_type,
                first_time_seen_seconds=current_time_seconds,
            )
            self._apply_match(
                track_id, det, current_time_seconds,
                screen_time_increment, tracked_objects, debug_detections,
            )

        return debug_detections

    def _apply_match(
        self,
        track_id: int,
        det: dict,
        current_time_seconds: float,
        screen_time_increment: float,
        tracked_objects: dict[int, TrackedObject],
        debug_detections: list[DebugDetection],
    ) -> None:
        """Update active state, TrackedObject, and emit a DebugDetection for one match."""
        box = det["box"]
        confidence = round(det["score"], 4)

        # Update active track state
        self._active[track_id]["box"] = box
        self._active[track_id]["last_seen"] = current_time_seconds

        # Update TrackedObject
        tracked_obj = tracked_objects[track_id]
        tracked_obj.screen_time_seconds += screen_time_increment
        tracked_obj.bbox_observations.append(
            BBoxObservation(
                timestamp=seconds_to_timestamp(current_time_seconds),
                x1=round(float(box[0]), 2),
                y1=round(float(box[1]), 2),
                x2=round(float(box[2]), 2),
                y2=round(float(box[3]), 2),
                confidence=confidence,
            )
        )

        debug_detections.append(
            DebugDetection(
                object_id=track_id,
                object_type=tracked_obj.object_type,
                x1=int(round(float(box[0]))),
                y1=int(round(float(box[1]))),
                x2=int(round(float(box[2]))),
                y2=int(round(float(box[3]))),
                confidence=confidence,
            )
        )
