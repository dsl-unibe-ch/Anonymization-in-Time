"""
TiT Application Launcher - Unified entry point for all tools

Launch the Video Processor (batch processing) or Annotation Viewer
from a single interface. Both can run simultaneously.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import subprocess
import sys
from pathlib import Path


class TiTLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("TiT Application Launcher")
        self.root.geometry("550x680")
        self.root.resizable(False, False)
        
        # Track running processes
        self.processes = []
        
        # Selected video folder for annotation viewer
        self.video_folder = None
        
        self._create_ui()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _create_ui(self):
        """Create the launcher UI"""
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="TiT Video Processing Suite",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 10))
        
        subtitle_label = ttk.Label(
            main_frame,
            text="Choose which application to launch",
            font=("Arial", 10)
        )
        subtitle_label.pack(pady=(0, 30))
        
        # === VIDEO PROCESSOR SECTION ===
        processor_frame = ttk.LabelFrame(main_frame, text="Video Processor", padding="15")
        processor_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(
            processor_frame,
            text="Process videos through OCR, SAM3, and transition detection.\n" +
                 "Use this to prepare videos for annotation.",
            justify=tk.LEFT,
            foreground="gray"
        ).pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Button(
            processor_frame,
            text="Launch Video Processor",
            command=self._launch_processor,
            width=30
        ).pack()
        
        # === ANNOTATION VIEWER SECTION ===
        viewer_frame = ttk.LabelFrame(main_frame, text="Annotation Viewer", padding="15")
        viewer_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(
            viewer_frame,
            text="View, edit, and export annotations for processed videos.\n" +
                 "Toggle visibility of OCR text and SAM3 masks.",
            justify=tk.LEFT,
            foreground="gray"
        ).pack(anchor=tk.W, pady=(0, 10))
        
        # Folder selection for viewer
        folder_frame = ttk.Frame(viewer_frame)
        folder_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(folder_frame, text="Video Folder:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.folder_label = ttk.Label(
            folder_frame,
            text="Not selected",
            foreground="gray",
            anchor=tk.W,
            width=30
        )
        self.folder_label.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            folder_frame,
            text="Browse...",
            command=self._browse_video_folder,
            width=10
        ).pack(side=tk.LEFT)
        
        self.viewer_button = ttk.Button(
            viewer_frame,
            text="Launch Annotation Viewer",
            command=self._launch_viewer,
            width=30,
            state=tk.DISABLED
        )
        self.viewer_button.pack()
        
        # === VIDEO EXPORT SECTION ===
        export_frame = ttk.LabelFrame(main_frame, text="Export Anonymized Video", padding="15")
        export_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(
            export_frame,
            text="Export a reviewed video with blurring applied.\n" +
                 "Select a video folder and configure export settings.",
            justify=tk.LEFT,
            foreground="gray"
        ).pack(anchor=tk.W, pady=(0, 10))
        
        # Blur strength setting
        blur_frame = ttk.Frame(export_frame)
        blur_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(blur_frame, text="Blur Strength:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.blur_strength_var = tk.IntVar(value=51)
        blur_spinbox = ttk.Spinbox(
            blur_frame,
            from_=3,
            to=201,
            increment=2,
            textvariable=self.blur_strength_var,
            width=10
        )
        blur_spinbox.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Label(
            blur_frame,
            text="(odd number, 3-201)",
            foreground="gray"
        ).pack(side=tk.LEFT)
        
        ttk.Button(
            export_frame,
            text="Export Video...",
            command=self._launch_export,
            width=30
        ).pack()
        
        # === STATUS SECTION ===
        self.status_label = ttk.Label(
            main_frame,
            text="Ready",
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding="5"
        )
        self.status_label.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Exit button
        ttk.Button(
            main_frame,
            text="Exit",
            command=self._on_close
        ).pack(side=tk.BOTTOM, pady=(0, 10))
    
    def _launch_processor(self):
        """Launch the Video Processor GUI"""
        try:
            script_path = Path(__file__).parent / "video_processor_gui.py"
            if not script_path.exists():
                messagebox.showerror(
                    "File Not Found",
                    f"Video processor not found at:\n{script_path}"
                )
                return
            
            # Launch as separate process
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
            )
            self.processes.append(("Video Processor", process))
            
            self.status_label.config(text="Video Processor launched")
        except Exception as e:
            messagebox.showerror("Launch Error", f"Failed to launch Video Processor:\n{str(e)}")
    
    def _browse_video_folder(self):
        """Browse for video folder containing frames and annotations"""
        folder = filedialog.askdirectory(
            title="Select Video Folder (containing frames/, ocr.pkl, sam3.pkl)",
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
            
            self.video_folder = folder_path
            # Show just the folder name if path is too long
            display_name = folder_path.name
            self.folder_label.config(text=display_name, foreground="black")
            self.folder_label.bind("<Enter>", lambda e: self._show_full_path())
            self.viewer_button.config(state=tk.NORMAL)
            self.status_label.config(text=f"Video folder selected: {display_name}")
    
    def _show_full_path(self):
        """Show full path in status on hover"""
        if self.video_folder:
            self.status_label.config(text=str(self.video_folder))
    
    def _launch_viewer(self):
        """Launch the Annotation Viewer with selected folder"""
        if not self.video_folder:
            messagebox.showwarning(
                "No Folder Selected",
                "Please select a video folder first using the Browse button."
            )
            return
        
        try:
            script_path = Path(__file__).parent / "src" / "main.py"
            if not script_path.exists():
                messagebox.showerror(
                    "File Not Found",
                    f"Annotation viewer not found at:\n{script_path}"
                )
                return
            
            # Launch as separate process with folder argument
            process = subprocess.Popen(
                [sys.executable, str(script_path), str(self.video_folder)],
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
            )
            self.processes.append(("Annotation Viewer", process))
            
            self.status_label.config(text=f"Annotation Viewer launched: {self.video_folder.name}")
        except Exception as e:
            messagebox.showerror("Launch Error", f"Failed to launch Annotation Viewer:\n{str(e)}")
    
    def _launch_export(self):
        """Launch video export with folder selection"""
        # Select video folder to export
        folder = filedialog.askdirectory(
            title="Select Video Folder to Export (containing frames/, state.pkl)",
            mustexist=True
        )
        
        if not folder:
            return
        
        folder_path = Path(folder)
        
        # Validate folder structure
        frames_dir = folder_path / "frames"
        if not frames_dir.exists():
            messagebox.showerror(
                "Invalid Folder",
                f"The selected folder does not contain a 'frames' subdirectory.\n\n" +
                f"Selected: {folder_path}"
            )
            return
        
        # Check for state.pkl or ocr.pkl/sam3.pkl
        state_file = folder_path / "state.pkl"
        ocr_file = folder_path / "ocr.pkl"
        sam3_file = folder_path / "sam3.pkl"
        
        if not state_file.exists() and not (ocr_file.exists() or sam3_file.exists()):
            messagebox.showerror(
                "No Annotations Found",
                f"The selected folder does not contain state.pkl, ocr.pkl, or sam3.pkl.\n\n" +
                f"Selected: {folder_path}"
            )
            return
        
        # Ask for output file path
        output_path = filedialog.asksaveasfilename(
            title="Export Anonymized Video",
            initialdir=str(folder_path.parent),
            initialfile=f"{folder_path.name}_anonymized.mp4",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("AVI Video", "*.avi"), ("All Files", "*.*")]
        )
        
        if not output_path:
            return
        
        # Get blur strength from spinbox and ensure it's odd
        blur_strength = self.blur_strength_var.get()
        if blur_strength % 2 == 0:
            blur_strength += 1
            self.blur_strength_var.set(blur_strength)
        
        # Show confirmation
        response = messagebox.askyesno(
            "Export Video",
            f"Export video with the following settings?\n\n" +
            f"Input folder: {folder_path.name}\n" +
            f"Output: {Path(output_path).name}\n" +
            f"Blur strength: {blur_strength}\n\n" +
            f"This may take several minutes and will run in a new window.",
            icon='question'
        )
        
        if not response:
            return
        
        try:
            # Prepare export script path
            export_script = Path(__file__).parent / "export_video.py"
            
            if not export_script.exists():
                messagebox.showerror(
                    "Export Script Not Found",
                    f"Export script not found at:\n{export_script}"
                )
                return
            
            # Launch export in new console window
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(export_script),
                    "--video_dir", str(folder_path),
                    "--output", output_path,
                    "--blur_strength", str(blur_strength)
                ],
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
            )
            self.processes.append(("Video Export", process))
            
            self.status_label.config(text=f"Exporting: {folder_path.name}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to start export:\n{str(e)}")
    
    def _on_close(self):
        """Handle window close event"""
        # Check if any processes are still running
        running = [name for name, proc in self.processes if proc.poll() is None]
        
        if running:
            response = messagebox.askyesno(
                "Applications Running",
                f"The following applications are still running:\n" +
                f"{', '.join(running)}\n\n" +
                f"Close launcher anyway?",
                icon='warning'
            )
            if not response:
                return
        
        self.root.destroy()


def main():
    root = tk.Tk()
    app = TiTLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
