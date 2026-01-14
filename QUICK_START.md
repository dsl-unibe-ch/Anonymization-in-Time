# TiT Application Suite - Quick Start Guide

## Launching the Applications

### Option 1: Unified Launcher (Recommended)

Double-click **`start_tit.bat`** (Windows) or run:
```bash
python launcher.py
```

This opens a menu where you can:
- Launch **Video Processor** (batch processing)
- Launch **Annotation Viewer** (view/edit annotations)
- Launch **Both** simultaneously (process videos while annotating others)

### Option 2: Direct Launch

**Video Processor:**
```bash
python video_processor_gui.py
```

**Annotation Viewer:**
```bash
python annotation_viewer.py
```

**Command-line batch processing:**
```bash
python process_videos.py --video_folder videos/ --output_dir output/ --dict_path names.json
```

## Workflow

### 1. Process Videos (Video Processor)

1. Launch the **Video Processor** from the launcher
2. Select video file(s) or folder
3. Choose output directory
4. Select names dictionary (JSON file)
5. Configure parameters (or use defaults)
6. Click "Start Processing"

Each video gets a folder with:
- `frames/` - Extracted frames
- `ocr.pkl` - OCR detections
- `sam3.pkl` - Face/profile masks
- `transitions.txt` - Scene transitions

### 2. Annotate Videos (Annotation Viewer)

1. Launch the **Annotation Viewer** from the launcher
2. Browse and select a processed video folder
3. Use the viewer to:
   - Toggle OCR text visibility (left-click individual, right-click parent box)
   - Toggle SAM3 mask visibility (left-click)
   - Mark/remove scene transitions (T/E/R keys)
   - Navigate frames (arrow keys, mouse wheel)
   - Export visibility state

### 3. Work Simultaneously

Launch **both applications** to:
- Process new videos in the Video Processor
- Annotate already-processed videos in the Annotation Viewer
- Switch between apps as needed

## File Structure

```
TiT_app/
├── launcher.py                    # Unified launcher (START HERE)
├── start_tit.bat                  # Windows quick launcher
├── video_processor_gui.py         # Batch processing GUI
├── annotation_viewer.py           # Annotation viewer with folder selection
├── process_videos.py              # Command-line batch processing
├── ocr.py                  # OCR processing module
├── batch_inference_sam3.py        # SAM3 processing module
├── transition_detection.py        # Transition detection module
├── utils.py                       # Video utilities
├── BATCH_PROCESSING_GUIDE.md      # Detailed batch processing docs
└── src/
    ├── main.py                    # Legacy annotation viewer (hardcoded data/)
    ├── models/                    # Data models
    │   ├── annotation_manager.py
    │   └── transition_manager.py
    └── ui/                        # UI components
        ├── main_window_tk.py
        └── canvas_widget_tk.py
```

## Output Structure

Processing a video creates this structure:

```
output_dir/
└── video_name/
    ├── frames/              # Ready for annotation viewer
    │   ├── frame_00000.png
    │   └── ...
    ├── ocr.pkl             # Ready for annotation viewer
    ├── sam3.pkl            # Ready for annotation viewer
    ├── transitions.txt     # Ready for annotation viewer
    ├── boxes.pkl           # Intermediate OCR data
    └── detected_masks*.pkl # Intermediate SAM3 data
```

## Keyboard Shortcuts (Annotation Viewer)

- **←/→** - Previous/Next frame
- **Space** - Play/Pause
- **H** - Toggle hidden preview
- **T** - Mark transition start
- **E** - Mark transition end
- **R** - Remove transition at current frame
- **Ctrl+S** - Save state
- **Ctrl+Q** - Quit
- **Mouse Wheel** - Scroll through frames (with acceleration)

## Mouse Controls (Annotation Viewer)

- **Left-click** on annotation - Toggle individual box/mask
- **Right-click** on OCR box - Toggle entire parent box
- **Slider** - Jump to specific frame

## Tips

1. **Processing speed**: Processing takes ~1 hour per video. Launch multiple Video Processor instances for parallel processing.

2. **While processing**: Launch Annotation Viewer separately to work on already-processed videos.

3. **Frame step**: Use `--frame_step 2` to extract every 2nd frame (faster processing, smaller output).

4. **CUDA memory**: If SAM3 crashes, reduce batch size or use `--sam3_device cpu`.

5. **Recent folders**: The annotation viewer will remember recently opened folders (planned feature).

## Troubleshooting

**"No folder selected" error**:
- Make sure you click "Browse" and select a folder
- The folder should contain a `frames/` subdirectory

**"CUDA out of memory"**:
- Reduce SAM3 batch size to 1-2
- Use CPU mode: `--sam3_device cpu`

**Processing stuck**:
- Check the log window for errors
- Processing can take 30-60 minutes per video
- Progress bar shows activity

**Annotations not loading**:
- Verify `ocr.pkl` and/or `sam3.pkl` exist in the folder
- Check file paths in error messages

For detailed batch processing options, see [BATCH_PROCESSING_GUIDE.md](BATCH_PROCESSING_GUIDE.md).
