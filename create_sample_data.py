"""
Example script to create sample data for testing the TiT Annotation Viewer.
This creates dummy frames and annotations for demonstration purposes.
"""
import pickle
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def create_sample_frames(output_dir: Path, num_frames: int = 20):
    """
    Create sample frame images.
    
    Args:
        output_dir: Directory to save frames
        num_frames: Number of frames to create
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    width, height = 1280, 720
    
    for i in range(num_frames):
        # Create image with gradient background
        img = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(img)
        
        # Gradient background
        for y in range(height):
            color_value = int(50 + (y / height) * 100)
            draw.rectangle([(0, y), (width, y+1)], fill=(color_value, color_value, color_value + 50))
        
        # Draw frame number
        try:
            font = ImageFont.truetype("arial.ttf", 80)
        except:
            font = ImageFont.load_default()
        
        text = f"Frame {i:04d}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        draw.text(
            ((width - text_width) // 2, (height - text_height) // 2),
            text,
            fill=(255, 255, 255),
            font=font
        )
        
        # Save frame
        img.save(output_dir / f"{i:04d}.jpg", quality=95)
    
    print(f"Created {num_frames} sample frames in {output_dir}")


def create_sample_annotations(output_dir: Path, num_frames: int = 20):
    """
    Create sample OCR and SAM3 annotations with track_ids for testing temporal propagation.
    
    Args:
        output_dir: Directory to save pickle files
        num_frames: Number of frames
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ocr_data = {}
    sam_data = {}
    
    # Create persistent tracks for SAM3 objects (3 objects tracked across frames)
    num_tracked_objects = 3
    tracked_objects = []
    
    for obj_id in range(num_tracked_objects):
        # Initial position for this tracked object
        base_x = 200 + obj_id * 300
        base_y = 250
        tracked_objects.append({
            'x': base_x,
            'y': base_y,
            'w': np.random.randint(100, 200),
            'h': np.random.randint(100, 150)
        })
    
    for frame_idx in range(num_frames):
        # OCR annotations (text boxes) - no tracking
        ocr_annotations = []
        num_ocr = np.random.randint(2, 6)
        
        for j in range(num_ocr):
            x1 = np.random.randint(100, 900)
            y1 = np.random.randint(100, 500)
            w = np.random.randint(80, 200)
            h = np.random.randint(30, 60)
            
            ocr_annotations.append({
                "bbox": (x1, y1, x1 + w, y1 + h),
                "score": np.random.uniform(0.7, 0.99),
                "text": f"Text_{frame_idx}_{j}",
                "mask": None,
                "source": "ocr",
                "to_show": True,
                "track_id": None  # OCR boxes don't have tracks
            })
        
        ocr_data[frame_idx] = ocr_annotations
        
        # SAM3 annotations (masks and boxes) - with tracking
        sam_annotations = []
        
        for obj_id in range(num_tracked_objects):
            obj = tracked_objects[obj_id]
            
            # Add slight random movement to simulate tracking
            obj['x'] += np.random.randint(-10, 11)
            obj['y'] += np.random.randint(-5, 6)
            
            # Keep within bounds
            obj['x'] = max(50, min(1000, obj['x']))
            obj['y'] = max(50, min(550, obj['y']))
            
            x1, y1 = obj['x'], obj['y']
            w, h = obj['w'], obj['h']
            
            # Create simple circular mask (packed format)
            mask_h, mask_w = h, w
            center_x_local = w // 2
            center_y_local = h // 2
            radius = min(w, h) // 2
            
            yy, xx = np.ogrid[:mask_h, :mask_w]
            circle_mask = (xx - center_x_local)**2 + (yy - center_y_local)**2 <= radius**2
            
            # Pack the mask
            packed = np.packbits(circle_mask.astype(np.uint8).reshape(-1))
            
            sam_annotations.append({
                "bbox": (x1, y1, x1 + w, y1 + h),
                "score": np.random.uniform(0.6, 0.95),
                "text": "",
                "mask": {
                    "packed": packed,
                    "shape": (mask_h, mask_w),
                    "bbox": [x1, y1, x1 + w - 1, y1 + h - 1]
                },
                "source": "sam3",
                "to_show": True,
                "track_id": obj_id  # Same track_id across frames
            })
        
        sam_data[frame_idx] = sam_annotations
    
    # Save separate files
    with open(output_dir / "ocr.pkl", 'wb') as f:
        pickle.dump(ocr_data, f)
    
    with open(output_dir / "sam3.pkl", 'wb') as f:
        pickle.dump(sam_data, f)
    
    # Also create combined file
    combined_data = {}
    for frame_idx in range(num_frames):
        combined_data[frame_idx] = ocr_data[frame_idx] + sam_data[frame_idx]
    
    with open(output_dir / "annotations.pkl", 'wb') as f:
        pickle.dump(combined_data, f)
    
    print(f"Created sample annotations in {output_dir}")
    print(f"  - ocr.pkl: {len(ocr_data)} frames")
    print(f"  - sam3.pkl: {len(sam_data)} frames")
    print(f"  - annotations.pkl: {len(combined_data)} frames (combined)")
    print(f"  - {num_tracked_objects} tracked objects across all frames")


def main():
    """Create sample data."""
    # Get project root
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"  # Fixed: was script_dir.parent / "data"
    
    print("Creating sample data for TiT Annotation Viewer...")
    print(f"Output directory: {data_dir}")
    
    # Create frames
    create_sample_frames(data_dir / "frames", num_frames=20)
    
    # Create annotations
    create_sample_annotations(data_dir, num_frames=20)
    
    print("\nSample data created successfully!")
    print("\nYou can now run the application:")
    print("  python src/main.py")


if __name__ == "__main__":
    main()
