from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any


JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
AnswerValue = str | int | float | bool | None


class QAFamily(str, Enum):
    """Detailed QA families used to sample balanced benchmark questions."""

    OBJECT_ATTRIBUTE = "object_attribute"
    ACTION_EVENT = "action_event"
    TEMPORAL_REASONING = "temporal_reasoning"
    SPATIAL_RELATION = "spatial_relation"
    SCENE_PLACE = "scene_place"
    TRAJECTORY_GROUNDED = "trajectory_grounded"
    DAY_NIGHT_ROBUSTNESS = "day_night_robustness"
    COUNTING = "counting"
    EVENT_MEMORY = "event_memory"
    NEGATIVE_ABSENCE = "negative_absence"
    AMBIGUITY_AWARE = "ambiguity_aware"

class AnswerFormat(str, Enum):
    OPEN_ENDED = "open_ended"
    MULTIPLE_CHOICE = "multiple_choice"
    YES_NO = "yes_no"
    NUMERIC = "numeric"


class ReasoningType(str, Enum):
    PERCEPTION = "perception"
    ACTION_RECOGNITION = "action_recognition"
    TEMPORAL_ORDERING = "temporal_ordering"
    EVENT_LOCALIZATION = "event_localization"
    SPATIAL_RELATION = "spatial_relation"
    SCENE_UNDERSTANDING = "scene_understanding"
    TRAJECTORY_ALIGNMENT = "trajectory_alignment"
    COUNTING = "counting"
    ABSENCE_DETECTION = "absence_detection"
    AMBIGUITY_HANDLING = "ambiguity_handling"


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class VisibilityQuality(str, Enum):
    CLEAR = "clear"
    BLURRED = "blurred"
    OCCLUDED = "occluded"
    DARK = "dark"
    GLARE = "glare"
    MIXED = "mixed"


class DayNightTag(str, Enum):
    DAY = "day"
    NIGHT = "night"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    """Temporal location in the video that supports the answer."""

    start_seconds: float
    end_seconds: float
    start_frame: int | None = None
    end_frame: int | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if self.start_seconds < 0 or self.end_seconds < 0:
            raise ValueError("Evidence span times must be non-negative.")
        if self.end_seconds < self.start_seconds:
            raise ValueError("Evidence span end_seconds must be >= start_seconds.")
        if self.start_frame is not None and self.start_frame < 0:
            raise ValueError("Evidence span start_frame must be non-negative.")
        if self.end_frame is not None and self.end_frame < 0:
            raise ValueError("Evidence span end_frame must be non-negative.")
        if (
            self.start_frame is not None
            and self.end_frame is not None
            and self.end_frame < self.start_frame
        ):
            raise ValueError("Evidence span end_frame must be >= start_frame.")

    def to_dict(self) -> dict[str, JsonValue]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceSpan:
        return cls(**data)


@dataclass(frozen=True, slots=True)
class TrajectoryLinkage:
    """Optional route evidence aligned with a video QA item."""

    route_id: str
    trajectory_file: str | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None
    start_sample_index: int | None = None
    end_sample_index: int | None = None
    route_stage: str | None = None
    movement_pattern: str | None = None

    def __post_init__(self) -> None:
        if not self.route_id:
            raise ValueError("Trajectory linkage route_id must not be empty.")
        if self.start_seconds is not None and self.start_seconds < 0:
            raise ValueError("Trajectory linkage start_seconds must be non-negative.")
        if self.end_seconds is not None and self.end_seconds < 0:
            raise ValueError("Trajectory linkage end_seconds must be non-negative.")
        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds < self.start_seconds
        ):
            raise ValueError("Trajectory linkage end_seconds must be >= start_seconds.")
        if self.start_sample_index is not None and self.start_sample_index < 0:
            raise ValueError("Trajectory linkage start_sample_index must be non-negative.")
        if self.end_sample_index is not None and self.end_sample_index < 0:
            raise ValueError("Trajectory linkage end_sample_index must be non-negative.")

    def to_dict(self) -> dict[str, JsonValue]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrajectoryLinkage:
        return cls(**data)


@dataclass(frozen=True, slots=True)
class BenchmarkQAPair:
    """One annotated benchmark question-answer item for a running video clip."""

    id: str
    video_id: str
    question: str
    answer: AnswerValue
    answer_format: AnswerFormat
    family: QAFamily
    reasoning_types: tuple[ReasoningType, ...]
    difficulty: DifficultyLevel
    visibility: VisibilityQuality
    day_night: DayNightTag
    evidence_spans: tuple[EvidenceSpan, ...]
    trajectory_linkage: TrajectoryLinkage | None = None
    choices: tuple[str, ...] = ()
    answer_aliases: tuple[str, ...] = ()
    unanswerable: bool = False

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("QA id must not be empty.")
        if not self.video_id:
            raise ValueError("QA video_id must not be empty.")
        if not self.question:
            raise ValueError("QA question must not be empty.")
        if not self.reasoning_types:
            raise ValueError("QA item must include at least one reasoning type.")
        if not self.evidence_spans and not self.unanswerable:
            raise ValueError("Answerable QA items must include at least one evidence span.")
        if self.answer is None and not self.unanswerable:
            raise ValueError("Answerable QA items must include an answer.")
        if self.answer_format == AnswerFormat.MULTIPLE_CHOICE and not self.choices:
            raise ValueError("Multiple-choice QA items must define choices.")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "video_id": self.video_id,
            "question": self.question,
            "answer": self.answer,
            "answer_format": self.answer_format.value,
            "family": self.family.value,
            "reasoning_types": [item.value for item in self.reasoning_types],
            "difficulty": self.difficulty.value,
            "visibility": self.visibility.value,
            "day_night": self.day_night.value,
            "evidence_spans": [span.to_dict() for span in self.evidence_spans],
            "trajectory_linkage": (
                self.trajectory_linkage.to_dict() if self.trajectory_linkage else None
            ),
            "choices": list(self.choices),
            "answer_aliases": list(self.answer_aliases),
            "unanswerable": self.unanswerable,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkQAPair:
        payload = dict(data)
        payload["answer_format"] = AnswerFormat(payload["answer_format"])
        payload["family"] = QAFamily(payload["family"])
        payload["reasoning_types"] = tuple(
            ReasoningType(item) for item in payload["reasoning_types"]
        )
        payload["difficulty"] = DifficultyLevel(payload["difficulty"])
        payload["visibility"] = VisibilityQuality(payload["visibility"])
        payload["day_night"] = DayNightTag(payload["day_night"])
        payload["evidence_spans"] = tuple(
            EvidenceSpan.from_dict(item) for item in payload.get("evidence_spans", [])
        )
        trajectory_linkage = payload.get("trajectory_linkage")
        payload["trajectory_linkage"] = (
            TrajectoryLinkage.from_dict(trajectory_linkage)
            if trajectory_linkage is not None
            else None
        )
        payload["choices"] = tuple(payload.get("choices", ()))
        payload["answer_aliases"] = tuple(payload.get("answer_aliases", ()))
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class BenchmarkQADataset:
    """A collection of QA items for one benchmark split or annotation file."""

    name: str
    qa_pairs: tuple[BenchmarkQAPair, ...]
    version: str = "1.0"
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Benchmark dataset name must not be empty.")
        ids = [qa_pair.id for qa_pair in self.qa_pairs]
        if len(ids) != len(set(ids)):
            raise ValueError("Benchmark dataset QA ids must be unique.")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "qa_pairs": [qa_pair.to_dict() for qa_pair in self.qa_pairs],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def write_json(self, path: str | Path, indent: int = 2) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent), encoding="utf-8")
        return output_path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkQADataset:
        return cls(
            name=data["name"],
            version=data.get("version", "1.0"),
            description=data.get("description"),
            qa_pairs=tuple(
                BenchmarkQAPair.from_dict(item) for item in data.get("qa_pairs", [])
            ),
        )

    @classmethod
    def from_json(cls, text: str) -> BenchmarkQADataset:
        return cls.from_dict(json.loads(text))

    @classmethod
    def read_json(cls, path: str | Path) -> BenchmarkQADataset:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
