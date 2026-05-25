# Multimodal Video Analyzer

A modular, local command-line tool that analyzes videos using state-of-the-art Object Detection models (supporting DETR and YOLOv8) to detect and count people, cars, and dogs. It runs entirely locally, processes videos frame-by-frame, draws real-time HUD overlays with bounding boxes, and saves structured CSV/JSON time-series reports into timestamped results folders.

---

## Features

* **Modular Architecture:** Easily swap detectors by implementing the `BaseDetector` interface.
* **Multi-Model Support:** Native support for Hugging Face Transformer models (DETR) and Ultralytics PyTorch models (YOLOv8).
* **HUD Overlay:** Renders bounding boxes with corner brackets, confidence tags, and a transparent HUD status bar displaying live counts, elapsed time, active model, and hardware acceleration.
* **Optimized Sampling:** Run inference every N frames (`--fps-sample`) and carry over/interpolate detections to speed up execution.
* **Clean Outputs:** Organizes results inside unique, timestamped folders: `output/results_{video_name}_YYYYMMDD_HHMM/`.
* **Structured Reports:** Exports second-by-second analytics to CSV/JSON format and prints summary statistics in the console.
* **Output Size Reduction:** Flexible parameters to compress and scale down video files, reducing storage usage by over 90%.

---

## Project Structure

```text
multimodal-video-analyzer/
├── core/
│   ├── openh264-1.8.0-win64.dll # Cisco library for H.264 encoding under Windows
│   └── video_processor.py      # Core video processing pipeline & stats compiler
├── models/
│   ├── detector_interface.py    # Interface class for interchangeable detectors
│   ├── detr_detector.py        # Hugging Face DETR model wrapper
│   └── yolo_detector.py        # Ultralytics YOLOv8 model wrapper
├── utils/
│   ├── overlay_renderer.py     # HUD & bounding box rendering
│   └── report_generator.py     # CSV & JSON report writers
├── main.py                     # CLI entry point & model loader factory
├── requirements.txt            # Package dependencies
└── README.md                   # This documentation
```

---

## Installation

1. Ensure Python 3.12 (or higher) is installed.
2. Open your terminal in this directory and install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Basic Runs
Run the analyzer on a video file using default settings (DETR model, 1 FPS inference, full resolution):

```bash
# Using the default DETR model
python main.py --input input/sample.mp4 --model-type detr

# Using the fast YOLOv8 model (recommended)
python main.py --input input/sample.mp4 --model-type yolo
```

### 2. Batch Processing
To process all videos inside a folder sequentially:
```bash
python main.py --input input/
```

---

## CLI Parameters

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | `str` | *Required* | Path to the video file or folder containing videos. |
| `--output-dir` | `str` | `output` | Main output directory for results. |
| `--confidence` | `float` | `0.7` | Confidence threshold (0.0 to 1.0) below which detections are ignored. |
| `--fps-sample` | `float` | `1.0` | Target frame rate for AI inference (e.g. `2.0` means sample twice per second; `0` analyzes all frames). |
| `--model-type` | `str` | `detr` | Model architecture type: `detr` or `yolo`. |
| `--model-id` | `str` | *Dynamic* | Specific model path or Hugging Face ID (e.g., `yolov8m.pt`). |
| `--device` | `str` | `auto` | Execution device: `cuda` (GPU), `cpu`, or `auto`. |
| `--codec` | `str` | `mp4v` | Video codec to use for output (`mp4v` or `avc1` for H.264). |
| `--resize-factor` | `float` | `1.0` | Scale factor for output video resolution (0.1 to 1.0). |
| `--save-sampled-only`| `Flag` | *Off* | If set, only writes frames analyzed by the AI model. |

---

## Optimizing Detection Accuracy

If the default run detects too few objects, adjust the following parameters:

1. **Lower Confidence Threshold (`--confidence`):**
   The default threshold is `0.7`. Lowering it to `0.4` or `0.5` captures smaller or partially obscured objects:
   ```bash
   python main.py --input input/sample.mp4 --model-type yolo --confidence 0.4
   ```

2. **Use a Larger YOLO Model (`--model-id`):**
   The default `yolov8n.pt` (Nano) model is optimized for speed, but may miss objects. Choose a larger model variant (automatically downloaded by Ultralytics):
   * `yolov8s.pt` (Small) - Good speed/accuracy balance
   * `yolov8m.pt` (Medium) - Recommended default for most systems
   * `yolov8l.pt` (Large) - Highly accurate
   * `yolov8x.pt` (Extra Large) - Maximum accuracy
   
   ```bash
   python main.py --input input/sample.mp4 --model-type yolo --model-id yolov8m.pt
   ```

3. **Increase Inference Frequency (`--fps-sample`):**
   Increase the sampling rate (e.g. to `5.0` or `0` for all frames) to capture fast-moving objects.

---

## Reducing Output Video File Size

By default, output videos are written at full resolution and framerate. To compress the output video file size:

1. **Save Sampled Frames Only (`--save-sampled-only`):**
   By default, the tool duplicates the bounding boxes across all intermediate frames of the original video (e.g. 30 FPS) to match the input video's frame rate. Enabling this option tells the tool to discard these intermediate duplicate frames and only write the frames that were actually analyzed by the AI model. The playback framerate of the output video is adjusted to match the sampling frequency (`--fps-sample`), ensuring it plays at the correct real-time speed. (Reduces file size by **90-95%**).
2. **Downscale Resolution (`--resize-factor`):**
   Allows scaling down the output video dimensions. For example, `--resize-factor 0.5` reduces a 1080p (`1920x1080`) video to 540p (`960x540`). Since this decreases the total number of pixels by 75%, it results in a significantly smaller file size without affecting the AI's internal detection resolution.
3. **Use H.264 Compression (`--codec avc1`):**
   Changes the video compression format from the default legacy MPEG-4 (`mp4v`) codec to H.264 (`avc1`). H.264 offers vastly superior compression efficiency (much smaller file sizes for the same quality) and is natively playable in modern web browsers and mobile devices. On Windows, the program dynamically loads the Cisco OpenH264 library located in the `core/` folder to enable this codec out of the box.

### Recommended Compression Command:
```bash
python main.py --input input/sample.mp4 --model-type yolo --codec avc1 --resize-factor 0.5 --save-sampled-only
```

---

## Output Files

For every video processed, a unique directory is created containing:
* **`{video_name}_analyzed.mp4`**: The annotated video with overlays. (Audio is discarded automatically to save space).
* **`{video_name}_analysis.csv`**: Time-series log containing second-by-second target class counts.
* **`{video_name}_analysis.json`**: Metadata, overall statistics, and a full timeline log.
