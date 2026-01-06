"""
Main Window - primary application window with controls and canvas (Tkinter version).
"""
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from ui.canvas_widget_tk import CanvasWidget
from models.annotation_manager import AnnotationManager


class MainWindow:
    """Main application window."""
    
    def __init__(self, root: tk.Tk, data_dir: Path):
        """
        Initialize the main window.
        
        Args:
            root: Tkinter root window
            data_dir: Path to data directory
        """
        self.root = root
        self.data_dir = Path(data_dir)
        self.frames_dir = self.data_dir / "frames"
        
        # Initialize managers
        self.annotation_manager = AnnotationManager(self.data_dir)
        
        # Current frame position
        self.current_position = 0
        self._slider_updating = False  # Flag to prevent recursion
        
        # Setup window
        self.root.title("TiT Video Annotation Viewer")
        self.root.geometry("1200x800")
        
        # Setup UI
        self._create_menu()
        self._create_widgets()
        self._bind_shortcuts()
        
        # Load data
        self._load_data()
    
    def _create_menu(self):
        """Create menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save State (Ctrl+S)", command=self._save_state)
        file_menu.add_command(label="Export Visibility", command=self._export_visibility)
        file_menu.add_separator()
        file_menu.add_command(label="Quit (Ctrl+Q)", command=self._quit)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Toggle Hidden Preview (H)", command=self._toggle_hidden_preview)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
    
    def _create_widgets(self):
        """Create UI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas
        self.canvas = CanvasWidget(
            main_frame, 
            self.frames_dir,
            on_annotation_clicked=self._on_annotation_clicked
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Info frame
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.frame_label = ttk.Label(info_frame, text="Frame: 0 / 0")
        self.frame_label.pack(side=tk.LEFT)
        
        self.stats_label = ttk.Label(info_frame, text="OCR: 0 | SAM3: 0 | Visible: 0 | Hidden: 0")
        self.stats_label.pack(side=tk.RIGHT)
        
        # Slider frame
        slider_frame = ttk.Frame(main_frame)
        slider_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.prev_button = ttk.Button(slider_frame, text="◀ Previous", command=self._go_to_previous_frame)
        self.prev_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.slider = ttk.Scale(
            slider_frame, 
            from_=0, 
            to=0, 
            orient=tk.HORIZONTAL,
            command=self._on_slider_changed
        )
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.next_button = ttk.Button(slider_frame, text="Next ▶", command=self._go_to_next_frame)
        self.next_button.pack(side=tk.LEFT, padx=(5, 0))
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        self.toggle_preview_button = ttk.Button(
            button_frame, 
            text="Toggle Hidden Preview (H)", 
            command=self._toggle_hidden_preview
        )
        self.toggle_preview_button.pack(side=tk.LEFT)
        
        self.save_button = ttk.Button(button_frame, text="Save State (S)", command=self._save_state)
        self.save_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        self.export_button = ttk.Button(button_frame, text="Export Visibility", command=self._export_visibility)
        self.export_button.pack(side=tk.RIGHT)
        
        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _bind_shortcuts(self):
        """Bind keyboard shortcuts."""
        self.root.bind('<Left>', lambda e: self._go_to_previous_frame())
        self.root.bind('<Right>', lambda e: self._go_to_next_frame())
        self.root.bind('<Up>', lambda e: self._jump_frames(-10))
        self.root.bind('<Down>', lambda e: self._jump_frames(10))
        self.root.bind('<Home>', lambda e: self._go_to_position(0))
        self.root.bind('<End>', lambda e: self._go_to_position(self.annotation_manager.get_frame_count() - 1))
        self.root.bind('<h>', lambda e: self._toggle_hidden_preview())
        self.root.bind('<H>', lambda e: self._toggle_hidden_preview())
        self.root.bind('<s>', lambda e: self._save_state())
        self.root.bind('<S>', lambda e: self._save_state())
        self.root.bind('<Control-s>', lambda e: self._save_state())
        self.root.bind('<Control-S>', lambda e: self._save_state())
        self.root.bind('<Control-q>', lambda e: self._quit())
        self.root.bind('<Control-Q>', lambda e: self._quit())
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
    
    def _load_data(self):
        """Load annotation data and first frame."""
        # Check if frames directory exists
        if not self.frames_dir.exists():
            messagebox.showerror(
                "Error",
                f"Frames directory not found: {self.frames_dir}\n\n"
                "Please ensure your data is organized as:\n"
                "data/frames/  (containing 0000.jpg, 0001.jpg, ...)\n"
                "data/annotations.pkl or data/ocr.pkl + data/sam3.pkl"
            )
            return
        
        # Load annotations
        if not self.annotation_manager.load_annotations():
            messagebox.showwarning(
                "Warning",
                f"Failed to load annotations from {self.data_dir}\n\n"
                "Please ensure you have either:\n"
                "- data/annotations.pkl (combined)\n"
                "- data/ocr.pkl and/or data/sam3.pkl (separate)"
            )
            return
        
        # Setup slider
        frame_count = self.annotation_manager.get_frame_count()
        if frame_count > 0:
            self.slider.config(to=frame_count - 1)
            self._go_to_position(0)
            self._update_status(f"Loaded {frame_count} frames")
        else:
            self._update_status("No frames loaded")
    
    def _go_to_position(self, position: int):
        """Go to specific position in frame list."""
        frame_count = self.annotation_manager.get_frame_count()
        if frame_count == 0:
            return
        
        # Clamp position
        position = max(0, min(position, frame_count - 1))
        self.current_position = position
        
        # Update slider without triggering callback
        self._slider_updating = True
        self.slider.set(position)
        self._slider_updating = False
        
        # Load frame
        frame_idx = self.annotation_manager.get_frame_index(position)
        if frame_idx is not None:
            annotations = self.annotation_manager.get_frame_annotations(frame_idx)
            self.canvas.load_frame(frame_idx, annotations)
            self._update_labels(frame_idx)
    
    def _go_to_next_frame(self):
        """Go to next frame."""
        self._go_to_position(self.current_position + 1)
    
    def _go_to_previous_frame(self):
        """Go to previous frame."""
        self._go_to_position(self.current_position - 1)
    
    def _jump_frames(self, delta: int):
        """Jump multiple frames."""
        self._go_to_position(self.current_position + delta)
    
    def _on_slider_changed(self, value):
        """Handle slider value change."""
        if not self._slider_updating:  # Prevent recursion
            self._go_to_position(int(float(value)))
    
    def _on_annotation_clicked(self, annotation_id: int, frame_idx: int):
        """Handle annotation click."""
        # Toggle annotation
        new_state = self.annotation_manager.toggle_annotation(annotation_id, frame_idx)
        
        # Reload frame to update display
        annotations = self.annotation_manager.get_frame_annotations(frame_idx)
        self.canvas.load_frame(frame_idx, annotations)
        self._update_labels(frame_idx)
        
        # Show status
        state_str = "visible" if new_state else "hidden"
        self._update_status(f"Annotation {annotation_id} is now {state_str}", timeout=2000)
    
    def _toggle_hidden_preview(self):
        """Toggle preview mode for hidden annotations."""
        self.canvas.toggle_hidden_preview()
        state = "enabled" if self.canvas.show_hidden_preview else "disabled"
        self._update_status(f"Hidden preview {state}", timeout=2000)
    
    def _save_state(self):
        """Save current state."""
        if self.annotation_manager.save_state():
            messagebox.showinfo("Success", "State saved successfully!")
            self._update_status("State saved", timeout=3000)
        else:
            messagebox.showerror("Error", "Failed to save state")
    
    def _export_visibility(self):
        """Export visibility state."""
        filepath = filedialog.asksaveasfilename(
            title="Export Visibility State",
            initialdir=str(self.data_dir),
            initialfile="visibility.pkl",
            defaultextension=".pkl",
            filetypes=[("Pickle Files", "*.pkl"), ("All Files", "*.*")]
        )
        
        if filepath:
            if self.annotation_manager.export_visibility_state(Path(filepath)):
                messagebox.showinfo("Success", f"Visibility state exported to:\n{filepath}")
                self._update_status("Visibility exported", timeout=3000)
            else:
                messagebox.showerror("Error", "Failed to export visibility state")
    
    def _update_labels(self, frame_idx: int):
        """Update info labels."""
        # Frame label
        frame_count = self.annotation_manager.get_frame_count()
        self.frame_label.config(
            text=f"Frame: {self.current_position + 1} / {frame_count} (Index: {frame_idx})"
        )
        
        # Stats label
        stats = self.annotation_manager.get_statistics(frame_idx)
        self.stats_label.config(
            text=f"OCR: {stats['ocr']} | SAM3: {stats['sam3']} | "
                 f"Visible: {stats['visible']} | Hidden: {stats['hidden']}"
        )
    
    def _update_status(self, message: str, timeout: int = 0):
        """Update status bar."""
        self.status_bar.config(text=message)
        
        if timeout > 0:
            self.root.after(timeout, lambda: self.status_bar.config(text="Ready"))
    
    def _show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About TiT Annotation Viewer",
            "TiT Video Annotation Viewer\n\n"
            "Interactive viewer for OCR and SAM3 annotations on video frames.\n\n"
            "Controls:\n"
            "• Click on visible annotation to hide it\n"
            "• Hover over hidden annotation area and click to show it\n"
            "• Use arrow keys to navigate frames\n"
            "• Press H to toggle hidden preview mode\n"
            "• Press S to save state\n\n"
            "Version 1.0 - January 2026"
        )
    
    def _quit(self):
        """Handle quit."""
        # Check for unsaved changes
        if self.annotation_manager.modified:
            result = messagebox.askyesnocancel(
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save before closing?"
            )
            
            if result is None:  # Cancel
                return
            elif result:  # Yes, save
                self._save_state()
        
        self.root.quit()
