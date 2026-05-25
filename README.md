# Multimodal Video Analyzer

A modular, local command-line tool that analyzes videos using state-of-the-art Object Detection models (supporting **DETR** and **YOLOv8**) to detect and count **people**, **cars**, and **dogs**.

It runs entirely locally, processes videos frame-by-frame, draws premium real-time HUD overlays with bounding boxes onto the output video, and saves structured CSV/JSON time-series reports into timestamped results folders.

---

## Features
*   **Modular Architecture:** Easily plug in new object detection models by implementing the `BaseDetector` interface.
*   **Multi-Model Support:** Native support for Hugging Face transformer models (like **DETR**) and Ultralytics PyTorch models (like **YOLOv8**).
*   **HUD Overlays:** Draws custom bounding boxes with corner brackets, confidence tags, and a top-bar HUD displaying live object counts, execution time, active model, and hardware device.
*   **Highly Optimized Sampling:** Skip AI inference on intermediate frames (customizable via `--fps-sample`) and carry over/interpolate detections to speed up execution.
*   **Unique Result Directories:** Automatically organizes each run inside a timestamped folder format `results_{video_name}_YYYYMMDD_HHMM/` (no seconds to prevent directory clutter).
*   **Reporting:** Exports second-by-second analytics to CSV and JSON formats, and prints a final statistics summary table directly in the console.

---

## Project Structure

```text
multimodal-video-analyzer/
├── core/
│   └── video_processor.py    # Core video processing pipeline & stats compiler
├── models/
│   ├── detector_interface.py  # Abstract base class for interchangeable detectors
│   ├── detr_detector.py      # Hugging Face DETR model wrapper
│   └── yolo_detector.py      # Ultralytics YOLOv8 model wrapper
├── utils/
│   ├── overlay_renderer.py   # HUD and Bounding Box visual overlay renderer
│   └── report_generator.py   # CSV and JSON report writers
├── main.py                    # Main CLI entry point & model loader factory
├── requirements.txt           # Package dependencies (PyTorch, Ultralytics, Transformers, OpenCV, etc.)
└── README.md                  # Documentation
```

---

## Installation

1.  Make sure Python 3.12 (or higher) is installed.
2.  Open your terminal in this directory and install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

### 1. Basic Runs
To run the analyzer on a video file with default settings:

*   **Using DETR model** (default):
    ```bash
    python main.py --input input/sample.mp4 --model-type detr
    ```
*   **Using YOLOv8 model** (runs extremely fast on CPU/GPU):
    ```bash
    python main.py --input input/sample.mp4 --model-type yolo
    ```

### 2. Custom Model IDs
You can specify a custom model or size:
```bash
python main.py --input input/sample.mp4 --model-type yolo --model-id yolov8m.pt
```

### 3. Batch Folder Run
To process all videos inside a specific directory:
```bash
python main.py --input input/
```

---

## CLI Parameters

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | `str` | *Required* | Path to the video file or folder containing videos. |
| `--output-dir` | `str` | `output` | Base directory where results folders will be created. |
| `--confidence` | `float` | `0.7` | Confidence threshold (0.0 to 1.0) below which detections are ignored. |
| `--fps-sample` | `float` | `1.0` | Target frame rate for running AI inference (e.g. `2.0` means analyze two frames per second; set to `0` to analyze every frame). |
| `--model-type` | `str` | `detr` | Model architecture type to load: `detr` or `yolo`. |
| `--model-id` | `str` | `None` | Specific weights or Hugging Face ID. Defaults dynamically based on type: `facebook/detr-resnet-50` for DETR or `yolov8n.pt` for YOLO. |
| `--device` | `str` | `auto` | Execution device: `cuda`, `cpu`, or `auto` (auto-detects CUDA). |

---

## Output Files

For every video processed, a unique directory is created (e.g. `output/results_{video_name}_YYYYMMDD_HHMM/`):
*   **Annotated Video:** `{video_name}_analyzed.mp4` (video with overlay HUD and box boundaries).
*   **CSV Log:** `{video_name}_analysis.csv` (second-by-second target class counts).
*   **JSON Report:** `{video_name}_analysis.json` (runs metadata, overall statistics, and full timeline log).
