# Benchmark QA Pair Format

This directory defines the data format for benchmark question-answer pairs used to evaluate multimodal models on first-person running videos. The schema is implemented in `qa_schema.py` and can be serialized to JSON.

## Core Classes

`BenchmarkQAPair` represents one annotated question-answer item for a video segment.

`EvidenceSpan` stores the temporal video evidence that supports the answer.

`TrajectoryLinkage` optionally links a QA item to route or movement trajectory data.

`BenchmarkQADataset` groups multiple QA pairs into one dataset or split.

## QA Pair Fields

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `id` | `str` | Yes | Unique QA item identifier. |
| `video_id` | `str` | Yes | Identifier of the source video or clip. |
| `question` | `str` | Yes | Natural-language question asked about the video. |
| `answer` | `str`, `int`, `float`, `bool`, or `null` | Yes unless unanswerable | Ground-truth answer. Use `null` for unanswerable items. |
| `answer_format` | enum | Yes | Expected answer style. |
| `family` | enum | Yes | Detailed QA family. |
| `reasoning_types` | list of enum | Yes | Reasoning skills needed to answer the question. |
| `difficulty` | enum | Yes | Annotation difficulty. |
| `visibility` | enum | Yes | Visual evidence quality. |
| `day_night` | enum | Yes | Lighting condition tag. |
| `evidence_spans` | list of objects | Yes for answerable items | Video time spans that support the answer. |
| `trajectory_linkage` | object or `null` | Optional | Route or GPS trace linkage for trajectory-aware QA. |
| `choices` | list of `str` | Required for multiple choice | Candidate answers for multiple-choice questions. |
| `answer_aliases` | list of `str` | Optional | Accepted alternative answer strings. |
| `unanswerable` | `bool` | Yes | Whether the question cannot be answered from the evidence. |

## Enum Values

`answer_format`:

`open_ended`, `multiple_choice`, `yes_no`, `numeric`

`family`:

`object_attribute`, `action_event`, `temporal_reasoning`, `spatial_relation`, `scene_place`, `trajectory_grounded`, `day_night_robustness`, `counting`, `event_memory`, `negative_absence`, `ambiguity_aware`

`reasoning_types`:

`perception`, `action_recognition`, `temporal_ordering`, `event_localization`, `spatial_relation`, `scene_understanding`, `trajectory_alignment`, `counting`, `absence_detection`, `ambiguity_handling`

`difficulty`:

`easy`, `medium`, `hard`

`visibility`:

`clear`, `blurred`, `occluded`, `dark`, `glare`, `mixed`

`day_night`:

`day`, `night`, `mixed`, `unknown`

## Evidence Span Format

Each `evidence_spans` item has this shape:

```json
{
  "start_seconds": 4.2,
  "end_seconds": 6.8,
  "start_frame": null,
  "end_frame": null,
  "description": "A bicycle is visible near the right side of the path."
}
```

`start_seconds` and `end_seconds` are required and must be non-negative. Frame indices are optional.

## Trajectory Linkage Format

Use `trajectory_linkage` when a question requires route, turn, loop, or movement-stage context.

```json
{
  "route_id": "route_001",
  "trajectory_file": "routes/route_001.gpx",
  "start_seconds": 12.0,
  "end_seconds": 18.5,
  "start_sample_index": 30,
  "end_sample_index": 45,
  "route_stage": "after_crossing",
  "movement_pattern": "left_turn"
}
```

Set `trajectory_linkage` to `null` when no route data is needed.

## Minimal Example

See `example_minimal_qa_pair.json` for a complete single-item example:

```json
{
  "id": "qa_0001",
  "video_id": "karlsruhe_run_day_001",
  "question": "Is there a bicycle visible in this segment?",
  "answer": true,
  "answer_format": "yes_no",
  "family": "negative_absence",
  "reasoning_types": ["absence_detection"],
  "difficulty": "easy",
  "visibility": "clear",
  "day_night": "day",
  "evidence_spans": [
    {
      "start_seconds": 4.2,
      "end_seconds": 6.8,
      "start_frame": null,
      "end_frame": null,
      "description": "A bicycle is visible near the right side of the path."
    }
  ],
  "trajectory_linkage": null,
  "choices": [],
  "answer_aliases": ["yes"],
  "unanswerable": false
}
```

## Python Usage

```python
import json

from benchmarks import BenchmarkQAPair

with open("benchmarks/example_minimal_qa_pair.json", encoding="utf-8") as file:
    qa_pair = BenchmarkQAPair.from_dict(json.load(file))

print(qa_pair.question)
print(qa_pair.to_dict())
```

For a benchmark file containing many items, wrap them in `BenchmarkQADataset` and store them under the `qa_pairs` field.
