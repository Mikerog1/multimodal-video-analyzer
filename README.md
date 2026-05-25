# Multimodal Video Analyzer

A local, command-line tool that analyzes videos using an open-source Transformer-based Object Detection model (`facebook/detr-resnet-50` pretrained on COCO) to count **people**, **cars**, and **dogs**.

It runs entirely locally, processes videos frame-by-frame, draws high-quality visual bounding boxes with HUD stats onto the output video, and saves structured CSV/JSON time-series reports.

---

## Features
- **Zero-setup local AI**: Uses `facebook/detr-resnet-50` via PyTorch and Hugging Face `transformers`—no API keys or external requests required.
- **HUD Overlays**: Draws bounding boxes, labels, confidence scores, and real-time count metrics inside the output video frames.
- **Highly Optimized**: Features custom frame-sampling (configurable via `--fps-sample`) that skips AI inference on intermediate frames and interpolates bounding boxes, speeding up execution by up to **30x** on CPU.
- **Batch Processing**: Point it at a single video or a whole folder, and it will analyze them sequentially.
- **Reporting**: Exports second-by-second analytics to both CSV and JSON formats, and prints a final statistics table directly in the console.

---

## Installation

1. Make sure Python 3.12 (or higher) is installed.
2. Open your terminal in this directory and install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Basic Run
To run the analyzer on a video file with default settings:
```bash
python analyzer.py --input input/your_video.mp4
```

This will run inference at `1.0 FPS` (once per second) and save results to the `output/` directory:
- Annotated Video: `output/your_video_analyzed.mp4`
- CSV Count Logs: `output/your_video_analysis.csv`
- JSON Detailed Metrics: `output/your_video_analysis.json`

### 2. Batch Folder Run
To process all videos inside a specific directory:
```bash
python analyzer.py --input input/
```

### 3. Customize Arguments
You can control the confidence threshold, sampling rate, and hardware device:
```bash
python analyzer.py --input input/video.mp4 --confidence 0.65 --fps-sample 2.0 --device cuda
```

#### CLI Parameters:
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | `str` | *Required* | Path to the video file or folder containing videos. |
| `--output-dir` | `str` | `output` | Directory where output files will be written. |
| `--confidence` | `float` | `0.7` | Confidence threshold (0.0 to 1.0) below which detections are ignored. |
| `--fps-sample` | `float` | `1.0` | Target frame rate for running AI inference (e.g. `2.0` means analyze two frames per second; set to `0` to analyze every frame). |
| `--model-id` | `str` | `facebook/detr-resnet-50` | Hugging Face Hub identifier for the object detection model. |
| `--device` | `str` | `auto` | Execution device: `cuda`, `cpu`, or `auto` (auto-detects CUDA). |

---

## Output Examples

### 1. Console Summary Table
```
============================================================
  ANALYSIS COMPLETE: traffic_street.mp4
  Duration: 0:00:15 (15.2s)
============================================================
  Class      │ Max Count  │ Average Count
  ───────────┼────────────┼───────────────
  People     │ 4          │ 2.45
  Cars       │ 9          │ 7.12
  Dogs       │ 1          │ 0.08
============================================================
  Outputs Saved:
  - Annotated Video : C:\...\output\traffic_street_analyzed.mp4
  - CSV Count Log   : C:\...\output\traffic_street_analysis.csv
============================================================
```

### 2. Output Video Frame (HUD Overlay)
- Shows bounding boxes styled with corner brackets.
- A styled top-bar displays: `People: X | Cars: Y | Dogs: Z`, time elapsed, and model / system hardware metrics.
