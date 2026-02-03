# TiT Video Annotation App

Interactive application for viewing and managing OCR and SAM3 annotations on video frames.

**Built with Tkinter** - No external dependencies beyond Python standard library + numpy/Pillow!

## Features

- Navigate through video frames with slider
- View OCR bounding boxes and SAM3 masks
- Toggle visibility of annotations with click
- Preview hidden annotations on hover
- Export annotation visibility state
- **Works on Windows/Mac/Linux without DLL issues**

## Installation

1. Pytorch

- Pytorch with CUDA support
```bash
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
- Pytorch with CPU/MPS support
```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
```

2. Additional dependencies
```bash
pip install -r requirements.txt
# OR with uv:
uv pip install -r requirements.txt
```

3. SAM3
```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

4. FFMPEG
- `ffmpeg` is needed for frame extraction; on macOS install via Homebrew: `brew install ffmpeg`.