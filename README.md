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

1. Install dependencies (numpy and Pillow only):
```bash
pip install -r requirements.txt
# OR with uv:
uv pip install -r requirements.txt
```

Note: Tkinter comes bundled with Python - no additional GUI framework needed!

2. Prepare your data:
   - Place frame images in `data/frames/` (0000.jpg, 0001.jpg, ...)
   - Place pickle files in `data/`:
     - `ocr.pkl` - OCR annotations
     - `sam3.pkl` - SAM3 annotations
     - OR `annotations.pkl` - Combined annotations

## Usage

```bash
python src/main.py
```

### Keyboard Shortcuts

- **Arrow Left/Right**: Navigate frames
- **H**: Toggle hidden items preview
- **S**: Save state
- **Ctrl+Q**: Quit

### Mouse Interactions

- **Hover**: Preview hidden annotations (shown with dashed border)
- **Click on visible item**: Hide annotation
- **Click on hidden item**: Show annotation

## Data Format

The pickle files should contain a dictionary with this structure:

```python
frames = {
    frame_idx: [
        {
            "bbox": (x1, y1, x2, y2),
            "score": float,
            "text": str or None,  # OCR only
            "mask": np.array or dict or None,  # SAM3 only (see below)
            "source": "ocr" or "sam3",
            "to_show": True or False
        },
        ...
    ]
}
```

### Mask Format

The app supports **two mask formats**:

1. **Full numpy array** (original):
   ```python
   "mask": np.array([H, W], dtype=bool)  # Full frame mask
   ```

2. **Packed/cropped format** (recommended for SAM3 - saves space):
   ```python
   "mask": {
       "packed": np.packbits(cropped_mask),  # Packed binary data
       "shape": (crop_h, crop_w),            # Crop dimensions
       "bbox": [x1, y1, x2, y2]              # Location in full frame
   }
   ```

The app automatically unpacks and reconstructs full masks during rendering.

## Building Executable

### Windows
```bash
pyinstaller --windowed --onefile src/main.py --name TiT_Annotator
```

### macOS
```bash
pyinstaller --windowed --onefile src/main.py --name TiT_Annotator
```
