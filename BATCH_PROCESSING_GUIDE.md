# Video Processing Pipeline - Batch Processing Guide

## Overview

The batch processing system allows you to process multiple videos through the complete pipeline:
1. **Frame Extraction** - Extract frames from videos
2. **OCR Detection** - Detect and track text with word-level boxes
3. **SAM3 Segmentation** - Detect and track masks (e.g., profile pictures)
4. **Transition Detection** - Identify scene transitions for blurring

Each video gets its own folder with all outputs organized for easy use with the annotation viewer.

## Folder Structure

For each processed video, the following structure is created:

```
output_dir/
├── video_name_1/
│   ├── frames/              # Extracted video frames
│   │   ├── frame_00000.png
│   │   ├── frame_00001.png
│   │   └── ...
│   ├── ocr.pkl             # OCR detections (ready for annotation viewer)
│   ├── sam3.pkl            # SAM3 masks (ready for annotation viewer)
│   ├── transitions.txt     # Scene transition ranges (ready for annotation viewer)
│   ├── boxes.pkl           # Intermediate OCR boxes
│   ├── detected_masks*.pkl # Intermediate SAM3 results
│   └── mask_tracks.pkl     # SAM3 tracking data
├── video_name_2/
│   └── ...
```

## Usage

### Option 1: GUI Application (Recommended)

The easiest way to process videos is using the graphical interface:

```bash
python video_processor_gui.py
```

**GUI Features:**
- Select single video file or entire folder of videos
- Configure all processing parameters
- Enable/disable specific processing steps
- Monitor real-time progress
- View processing logs

**Steps:**
1. Click "Select File(s)" or "Select Folder" to choose videos
2. Click "Browse" next to "Output Dir" to choose where outputs go
3. Click "Browse" next to "Names Dict" to select your JSON dictionary (e.g., `laet.json`)
4. Adjust parameters as needed (defaults work well for most cases)
5. Click "Start Processing"

### Option 2: Command Line

For automation or advanced use, use the command-line script:

**Process a single video:**
```bash
python process_videos.py \
    --video path/to/video.mp4 \
    --output_dir output/ \
    --dict_path laet.json
```

**Process a folder of videos:**
```bash
python process_videos.py \
    --video_folder path/to/videos/ \
    --output_dir output/ \
    --dict_path laet.json
```

**Advanced options:**
```bash
python process_videos.py \
    --video_folder path/to/videos/ \
    --output_dir output/ \
    --dict_path laet.json \
    --frame_step 2 \
    --ocr_languages en de fr \
    --ocr_workers 8 \
    --sam3_prompt "profile image, profile picture, avatar" \
    --sam3_batch_size 8 \
    --sam3_device cuda
```

**Skip specific steps:**
```bash
python process_videos.py \
    --video_folder path/to/videos/ \
    --output_dir output/ \
    --dict_path laet.json \
    --skip_transitions  # Skip transition detection
    --skip_sam3          # Skip SAM3 processing
```

**Use existing frames:**
```bash
python process_videos.py \
    --video_folder path/to/videos/ \
    --output_dir output/ \
    --dict_path laet.json \
    --skip_frames  # Use existing frames in output_dir/video_name/frames/
```

## Command-Line Options

### Required Arguments:
- `--video` - Path to a single video file
- `--video_folder` - Path to folder containing videos (use one of --video or --video_folder)
- `--output_dir` - Base output directory (each video gets a subfolder)
- `--dict_path` - Path to JSON file with names to detect for OCR

### Processing Options:
- `--frame_step INT` - Step between frames to extract (default: 1 = all frames)
- `--skip_frames` - Skip frame extraction (use existing frames)
- `--skip_ocr` - Skip OCR processing
- `--skip_sam3` - Skip SAM3 processing
- `--skip_transitions` - Skip transition detection

### OCR Options:
- `--ocr_languages LANG [LANG ...]` - Languages for OCR (default: en de)
- `--ocr_workers INT` - Number of parallel workers (default: 4)

### SAM3 Options:
- `--sam3_prompt TEXT` - Text prompt for segmentation (default: "profile image, profile picture")
- `--sam3_batch_size INT` - Batch size for inference (default: 4)
- `--sam3_device {cuda,cpu}` - Device to use (default: cuda)

## Using Processed Videos with Annotation Viewer

After processing, open the annotation viewer and load the video's folder:

```bash
python src/ui/main_window_tk.py
```

Then use the GUI to:
1. Load the frames folder (e.g., `output/video_name/frames/`)
2. Load the OCR file (`output/video_name/ocr.pkl`)
3. Load the SAM3 file (`output/video_name/sam3.pkl`)
4. The transitions file (`output/video_name/transitions.txt`) will be loaded automatically

## Examples

**Example 1: Process all videos in a folder**
```bash
python process_videos.py \
    --video_folder all_videos/ \
    --output_dir tests/ \
    --dict_path laet.json
```

**Example 2: Process a single video with custom SAM3 prompt**
```bash
python process_videos.py \
    --video all_videos/video_laet.mp4 \
    --output_dir tests/ \
    --dict_path laet.json \
    --sam3_prompt "person, face, profile picture"
```

**Example 3: Reprocess OCR only (skip frame extraction and SAM3)**
```bash
python process_videos.py \
    --video_folder all_videos/ \
    --output_dir tests/ \
    --dict_path laet.json \
    --skip_frames \
    --skip_sam3 \
    --skip_transitions
```

**Example 4: Extract every 2nd frame to save space**
```bash
python process_videos.py \
    --video_folder all_videos/ \
    --output_dir tests/ \
    --dict_path laet.json \
    --frame_step 2
```

## Technical Details

### Processing Pipeline

The pipeline executes in this order:
1. **Frame Extraction** - Uses `utils.extract_video_frames()`
2. **OCR Processing** - Uses `simple_ocr.process_video_ocr()`
   - Detects text boxes
   - Normalizes box heights
   - Tracks boxes across frames
   - Splits into word-level boxes
   - Filters by dictionary names
   - Assigns track_ids
3. **SAM3 Processing** - Uses `batch_inference_sam3.process_video_sam3()`
   - Detects masks with text prompt
   - Tracks masks across frames
   - Propagates missing masks
   - Assigns track_ids
4. **Transition Detection** - Uses `transition_detection.detect_scene_transitions()`
   - Multi-metric analysis (edges + histograms)
   - Saves to `transitions.txt`

### Output Files

**ocr.pkl** - Unified dictionary format:
```python
{
    frame_idx: {
        'annotations': [
            {
                'bbox': [x1, y1, x2, y2],
                'text': 'detected text',
                'confidence': 0.95,
                'track_id': 1,
                'parent_box': [x1, y1, x2, y2],
                'parent_box_text': 'full text'
            },
            ...
        ]
    },
    ...
}
```

**sam3.pkl** - Unified dictionary format:
```python
{
    frame_idx: {
        'annotations': [
            {
                'mask': binary_mask_array,
                'bbox': [x1, y1, x2, y2],
                'confidence': 0.98,
                'track_id': 1
            },
            ...
        ]
    },
    ...
}
```

**transitions.txt** - Simple text format:
```
# Scene Transition Ranges
# Format: start_frame,end_frame
10,25
150,165
```

## Programmatic Usage

You can also import and use the functions in your own scripts:

```python
from process_videos import process_single_video, process_multiple_videos

# Process a single video
result = process_single_video(
    video_path="video.mp4",
    output_base_dir="output",
    dict_path="names.json",
    ocr_languages=["en", "de"],
    sam3_prompt="profile image"
)

# Process multiple videos
results = process_multiple_videos(
    video_paths=["video1.mp4", "video2.mp4"],
    output_base_dir="output",
    dict_path="names.json"
)
```

Individual processing functions:

```python
from simple_ocr import process_video_ocr
from batch_inference_sam3 import process_video_sam3
from transition_detection import detect_scene_transitions
from utils import extract_video_frames

# Extract frames
frames_dir = extract_video_frames("video.mp4", output_dir="output/frames")

# Run OCR
ocr_data = process_video_ocr(
    video_path="video.mp4",
    output_dir="output",
    dict_path="names.json"
)

# Run SAM3
sam3_data = process_video_sam3(
    frames_folder="output/frames",
    output_folder="output",
    text_prompt="profile image"
)

# Detect transitions
transitions, file = detect_scene_transitions(
    video_path="video.mp4",
    output_dir="output"
)
```

## Troubleshooting

**CUDA out of memory:**
- Reduce `sam3_batch_size` to 1 or 2
- Use `--sam3_device cpu`

**OCR too slow:**
- Increase `frame_step` to process fewer frames
- Reduce `ocr_workers` if running out of RAM

**Wrong names detected:**
- Update your dictionary JSON file
- Check OCR language settings

**Transitions not detected:**
- Check the transitions.txt file manually
- Adjust threshold in transition_detection.py if needed

**Missing frames:**
- Don't use `--skip_frames` on first run
- Check video codec compatibility
