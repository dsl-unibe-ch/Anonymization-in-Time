"""
TiT Video Annotation Viewer - Main Entry Point

Interactive application for viewing and managing OCR and SAM3 annotations on video frames.
"""
import sys
from pathlib import Path
import tkinter as tk

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ui.main_window_tk import MainWindow


def main():
    """Main entry point for the application."""
    # Determine data directory
    # Default to ../data relative to this script
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    
    # Check if data directory exists
    if not data_dir.exists():
        print(f"Warning: Data directory not found at {data_dir}")
        print("Creating data directory structure...")
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "frames").mkdir(exist_ok=True)
        print(f"Please place your data in:")
        print(f"  - Frames: {data_dir / 'frames'}")
        print(f"  - Annotations: {data_dir / 'annotations.pkl'} or separate files")
    
    # Create tkinter root window
    root = tk.Tk()
    
    # Create and setup main window
    window = MainWindow(root, data_dir)
    
    # Run application
    root.mainloop()


if __name__ == "__main__":
    main()
