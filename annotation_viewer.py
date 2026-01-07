"""
Annotation Viewer - Standalone launcher with folder selection

This wrapper allows users to browse for and select video folders
instead of hardcoding the data directory.
"""

import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ui.main_window_tk import MainWindow


class VideoFolderSelector:
    """Dialog to select a video folder before launching the viewer"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Select Video Folder")
        self.root.geometry("600x300")
        self.root.resizable(False, False)
        
        self.selected_folder = None
        
        self._create_ui()
        
        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{x}+{y}")
    
    def _create_ui(self):
        """Create the folder selection UI"""
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(
            main_frame,
            text="Annotation Viewer",
            font=("Arial", 14, "bold")
        ).pack(pady=(0, 10))
        
        # Instructions
        instructions = ttk.Label(
            main_frame,
            text="Select the video folder containing:\n" +
                 "• frames/ directory with extracted frames\n" +
                 "• ocr.pkl with OCR annotations\n" +
                 "• sam3.pkl with SAM3 masks\n" +
                 "• transitions.txt with scene transitions (optional)",
            justify=tk.LEFT,
            foreground="gray"
        )
        instructions.pack(pady=(0, 20))
        
        # Folder selection frame
        folder_frame = ttk.LabelFrame(main_frame, text="Video Folder", padding="10")
        folder_frame.pack(fill=tk.X, pady=(0, 20))
        
        folder_inner = ttk.Frame(folder_frame)
        folder_inner.pack(fill=tk.X)
        
        self.folder_label = ttk.Label(
            folder_inner,
            text="No folder selected",
            foreground="gray",
            anchor=tk.W
        )
        self.folder_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        ttk.Button(
            folder_inner,
            text="Browse...",
            command=self._browse_folder
        ).pack(side=tk.RIGHT)
        
        # Recent folders (placeholder for future enhancement)
        recent_frame = ttk.LabelFrame(main_frame, text="Recent Folders", padding="10")
        recent_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        ttk.Label(
            recent_frame,
            text="No recent folders",
            foreground="gray"
        ).pack()
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(
            button_frame,
            text="Cancel",
            command=self.root.destroy
        ).pack(side=tk.LEFT)
        
        self.open_button = ttk.Button(
            button_frame,
            text="Open Viewer",
            command=self._open_viewer,
            state=tk.DISABLED
        )
        self.open_button.pack(side=tk.RIGHT)
    
    def _browse_folder(self):
        """Browse for video folder"""
        folder = filedialog.askdirectory(
            title="Select Video Folder",
            mustexist=True
        )
        
        if folder:
            folder_path = Path(folder)
            
            # Validate folder structure
            frames_dir = folder_path / "frames"
            if not frames_dir.exists():
                response = messagebox.askyesno(
                    "Missing Frames Folder",
                    f"The selected folder does not contain a 'frames' subdirectory.\n\n" +
                    f"Selected: {folder_path}\n\n" +
                    f"Continue anyway?",
                    icon='warning'
                )
                if not response:
                    return
            
            self.selected_folder = folder_path
            self.folder_label.config(text=str(folder_path), foreground="black")
            self.open_button.config(state=tk.NORMAL)
    
    def _open_viewer(self):
        """Open the annotation viewer with selected folder"""
        if not self.selected_folder:
            return
        
        self.root.destroy()
        
        # Create new root for main window
        viewer_root = tk.Tk()
        
        try:
            # Launch main window with selected folder
            window = MainWindow(viewer_root, self.selected_folder)
            viewer_root.mainloop()
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to open annotation viewer:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()


def main():
    """Main entry point"""
    # Create folder selection dialog
    root = tk.Tk()
    selector = VideoFolderSelector(root)
    root.mainloop()


if __name__ == "__main__":
    main()
