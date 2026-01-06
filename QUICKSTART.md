# Quick Start Guide

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

If you plan to build executables, also install PyInstaller:

```bash
pip install pyinstaller
```

### 2. Prepare Your Data

Organize your data in the following structure:

```
TiT_app/
└── data/
    ├── frames/           # Place frame images here
    │   ├── 0000.jpg
    │   ├── 0001.jpg
    │   └── ...
    └── annotations.pkl   # Combined annotations
    # OR separate files:
    ├── ocr.pkl          # OCR annotations
    └── sam3.pkl         # SAM3 annotations
```

### 3. Test with Sample Data (Optional)

Generate sample data for testing:

```bash
python create_sample_data.py
```

This creates 20 sample frames with random annotations in the `data/` directory.

## Running the Application

### Development Mode

```bash
python src/main.py
```

Or from the project root:

```bash
cd src
python main.py
```

## Usage

### Navigation
- **Slider**: Drag to navigate through frames
- **Previous/Next buttons**: Move one frame at a time
- **Arrow Left/Right**: Navigate frames (keyboard)
- **Arrow Up/Down**: Jump 10 frames (keyboard)
- **Home/End**: Jump to first/last frame

### Annotation Interaction
- **Click on visible annotation**: Hide it (sets `to_show = False`)
- **Hover over hidden area**: Preview hidden annotations (yellow/orange dashed border)
- **Click on hidden annotation**: Show it (sets `to_show = True`)

### Color Coding
- **Green boxes**: Visible OCR annotations
- **Blue semi-transparent**: Visible SAM3 masks
- **Yellow dashed**: Hidden OCR annotations (when hovering)
- **Orange dashed**: Hidden SAM3 annotations (when hovering)
- **Red highlight**: Currently hovered annotation

### Keyboard Shortcuts
- **H**: Toggle hidden preview mode (show all hidden annotations)
- **S** or **Ctrl+S**: Save current state
- **Ctrl+Q**: Quit application
- **Arrow keys**: Navigate frames

### Saving and Exporting
- **Save State**: Saves the complete annotations with updated `to_show` values to `data/state.pkl`
- **Export Visibility**: Exports only the visibility state (lighter file) to a custom location

## Building Executables

### Windows

```bash
pyinstaller TiT_Annotator.spec
```

The executable will be in `dist/TiT_Annotator.exe`

### macOS

```bash
pyinstaller TiT_Annotator.spec
```

The app bundle will be in `dist/TiT_Annotator.app`

### Alternative (Simple Build)

```bash
# Windows
pyinstaller --windowed --onefile src/main.py --name TiT_Annotator

# macOS
pyinstaller --windowed --onefile src/main.py --name TiT_Annotator
```

## Data Format

Your pickle files should contain dictionaries in this format:

```python
frames = {
    0: [  # frame_idx
        {
            "bbox": (x1, y1, x2, y2),       # Bounding box coordinates
            "score": 0.95,                   # Confidence score
            "text": "Sample Text",           # OCR only, None for SAM3
            "mask": numpy_array,             # SAM3 only, None for OCR
            "source": "ocr",                 # "ocr" or "sam3"
            "to_show": True                  # Visibility flag
        },
        # ... more annotations
    ],
    1: [...],  # Next frame
    # ...
}
```

## Troubleshooting

### "No frames loaded"
- Ensure frames are in `data/frames/` with format `0000.jpg`, `0001.jpg`, etc.
- Check that frame indices in pickle files match actual frame files

### "Failed to load annotations"
- Verify pickle files exist in `data/` directory
- Check pickle file structure matches expected format
- Try loading files in Python to debug: `pickle.load(open('data/annotations.pkl', 'rb'))`

### Application crashes on startup
- Run with console mode for debugging: Set `console=True` in `.spec` file
- Or run directly with Python: `python src/main.py`
- Check Python version compatibility (requires Python 3.8+)

### Masks not displaying
- Ensure mask arrays are 2D numpy arrays (H, W)
- Verify mask values are 0 or 1 (binary)
- Check that mask dimensions match frame dimensions

## Performance Tips

- The app caches up to 50 frames in memory for smooth navigation
- For videos with >10,000 frames, consider processing in segments
- Large masks may slow rendering; consider downsampling if needed
- Close other applications for better performance with large datasets

## Support

For issues or questions, check:
1. README.md for general information
2. Code comments for implementation details
3. Sample data generation script for data format examples
