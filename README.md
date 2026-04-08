# AiT — Anonymization in Time

Video anonymization tool for chat recordings. Detects names (OCR) and profile pictures (segmentation) in video frames, then lets you review and export anonymized videos with blurring applied.

## Features

- **Processing pipeline**: Frame extraction, OCR text detection, SAM3 segmentation, scene transition detection
- **Annotation viewer**: Navigate frames, toggle annotation visibility, preview hidden annotations on hover
- **Video export**: Apply Gaussian blur to visible annotations for anonymization
- **Cross-platform**: Works on Windows, Mac, and Linux

## Project Structure

```
AiT_app/
├── ait/                     # Main Python package
│   ├── ocr/                 # OCR text detection pipeline (EasyOCR)
│   ├── segmentation/        # Profile picture segmentation pipeline (SAM3)
│   ├── viewer/              # Tkinter annotation viewer app
│   ├── utils.py             # Shared utilities (frame extraction, device management)
│   ├── process_videos.py    # Pipeline orchestrator
│   ├── export_video.py      # Anonymized video export
│   ├── launcher.py          # GUI launcher
│   └── ...
├── tools/                   # Development/debug tools
│   ├── inspect_sam3_pipeline.py
│   └── inspect_ocr_pipeline.py
├── pyproject.toml
├── requirements.txt
└── sam3.pt                  # Model weights (download separately)
```

## Installation

### 1. Install the package

**Viewer only** (for reviewing annotations):
```bash
pip install .
```

**Full install** (viewer + processing pipelines):
```bash
pip install ".[processing]"
```

Or with requirements.txt (all dependencies):
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. PyTorch (for processing)

If you need GPU acceleration, install PyTorch with CUDA before the package:
```bash
# CUDA 12.6
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# CPU/MPS (default)
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
```

### 3. SAM3 model weights

> ⚠️ **SAM 3 Model Weights Required**
> Unlike other Ultralytics models, SAM 3 weights are not automatically downloaded. You must:
> 1. Request access on the [SAM 3 model page on Hugging Face](https://huggingface.co/facebook/sam3.1)
> 2. Once approved, download the `sam3.1_multiplex.pt` file
> 3. Rename it to `sam3.pt` and place it in the project root

The segmentation pipeline looks for `sam3.pt` in the working directory.

### 4. ffmpeg

Required for frame extraction:
- **Windows**: Download from https://ffmpeg.org and add to PATH
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

## Usage

After installation, these CLI commands are available:

```bash
ait              # Launch the GUI (choose between processor and viewer)
ait-process      # Run the video processing pipeline
ait-viewer       # Open the annotation viewer
ait-export       # Export anonymized video
```

Or run directly with Python:
```bash
python -m ait.launcher
python -m ait.process_videos --help
python -m ait.annotation_viewer
python -m ait.export_video --help
```
