"""Centralised constants shared across core modules.

Eliminates duplicate definitions of CLASS_SYNONYMS, VEHICLE_CLASSES,
CLASS_TO_OBJECT_TYPE, and VOCABULARY that previously lived in
qa_generator.py, verifier.py, and tracking.py.
"""

# ---------------------------------------------------------------------------
# Object-type mapping used by trackers to normalise detector class names
# to the canonical types stored in TrackedObject.
# ---------------------------------------------------------------------------
CLASS_TO_OBJECT_TYPE: dict[str, str] = {
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

# ---------------------------------------------------------------------------
# Synonyms: maps a canonical class name to the set of words that may appear
# in natural-language questions / captions and should be treated as equivalent.
# ---------------------------------------------------------------------------
CLASS_SYNONYMS: dict[str, list[str]] = {
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
    "stroller": ["stroller", "strollers", "baby carriage"],
}

VEHICLE_CLASSES: list[str] = ["car", "truck", "bus", "motorcycle", "bicycle"]

# Vocabulary of rare / interesting object classes used for negative-QA
# generation (kept for potential future use).
VOCABULARY: list[str] = [
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
    "bench",
]
