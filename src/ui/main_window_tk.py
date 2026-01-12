"""
Main Window - primary application window with controls and canvas (Tkinter version).
"""
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from ui.canvas_widget_tk import CanvasWidget
from models.annotation_manager import AnnotationManager
from models.transition_manager import TransitionManager


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
        self.transition_manager = TransitionManager(self.data_dir)
        
        # Current frame position
        self.current_position = 0
        self._slider_updating = False  # Flag to prevent recursion
        
        # Transition marking state
        self.transition_start_frame = None  # Frame where transition marking started
        
        # Mouse wheel scrolling acceleration
        self._last_scroll_time = 0
        self._scroll_speed = 3  # Start at 3 frames per scroll for faster initial movement
        
        # Preview mode
        self.preview_mode = False
        
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
        file_menu.add_command(label="Export Anonymized Video...", command=self._export_video)
        file_menu.add_separator()
        file_menu.add_command(label="Quit (Ctrl+Q)", command=self._quit)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Toggle Hidden Preview (H)", command=self._toggle_hidden_preview)
        view_menu.add_command(label="Toggle Blur Preview (B)", command=self._toggle_blur_preview)
        
        # Transitions menu
        transitions_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Transitions", menu=transitions_menu)
        transitions_menu.add_command(label="Mark Transition Start (T)", command=self._mark_transition_start)
        transitions_menu.add_command(label="Mark Transition End (E)", command=self._mark_transition_end)
        transitions_menu.add_command(label="Remove Current Transition (R)", command=self._remove_current_transition)
        transitions_menu.add_separator()
        transitions_menu.add_command(label="Save Transitions", command=self._save_transitions)
        transitions_menu.add_command(label="List All Transitions", command=self._list_transitions)
        
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
        
        # Left side buttons (transitions)
        left_buttons = ttk.Frame(button_frame)
        left_buttons.pack(side=tk.LEFT)
        
        self.transition_start_button = ttk.Button(
            left_buttons,
            text="Mark Start (T)",
            command=self._mark_transition_start
        )
        self.transition_start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.transition_end_button = ttk.Button(
            left_buttons,
            text="Mark End (E)",
            command=self._mark_transition_end
        )
        self.transition_end_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.remove_transition_button = ttk.Button(
            left_buttons,
            text="Remove (R)",
            command=self._remove_current_transition
        )
        self.remove_transition_button.pack(side=tk.LEFT)
        
        # Right side buttons
        self.toggle_preview_button = ttk.Button(
            button_frame, 
            text="Toggle Hidden Preview (H)", 
            command=self._toggle_hidden_preview
        )
        self.toggle_preview_button.pack(side=tk.LEFT, padx=(20, 0))
        
        self.preview_blur_button = ttk.Button(
            button_frame,
            text="Preview Blur (B)",
            command=self._toggle_blur_preview
        )
        self.preview_blur_button.pack(side=tk.LEFT, padx=(5, 0))
        
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
        self.root.bind('<b>', lambda e: self._toggle_blur_preview())
        self.root.bind('<B>', lambda e: self._toggle_blur_preview())
        self.root.bind('<s>', lambda e: self._save_state())
        self.root.bind('<S>', lambda e: self._save_state())
        self.root.bind('<Control-s>', lambda e: self._save_state())
        self.root.bind('<Control-S>', lambda e: self._save_state())
        self.root.bind('<Control-q>', lambda e: self._quit())
        self.root.bind('<Control-Q>', lambda e: self._quit())
        self.root.bind('<t>', lambda e: self._mark_transition_start())
        self.root.bind('<T>', lambda e: self._mark_transition_start())
        self.root.bind('<e>', lambda e: self._mark_transition_end())
        self.root.bind('<E>', lambda e: self._mark_transition_end())
        self.root.bind('<r>', lambda e: self._remove_current_transition())
        self.root.bind('<R>', lambda e: self._remove_current_transition())
        
        # Mouse wheel scrolling
        self.root.bind('<MouseWheel>', self._on_mouse_wheel)  # Windows/Mac
        self.root.bind('<Button-4>', lambda e: self._go_to_previous_frame())  # Linux scroll up
        self.root.bind('<Button-5>', lambda e: self._go_to_next_frame())  # Linux scroll down
        
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
        
        # Load transitions
        self.transition_manager.load_transitions()  # Non-fatal if file doesn't exist
        
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
            is_transition = self.transition_manager.is_in_transition(frame_idx)
            self.canvas.load_frame(frame_idx, annotations, is_transition)
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
    
    def _on_mouse_wheel(self, event):
        """Handle mouse wheel scrolling with acceleration."""
        import time
        
        current_time = time.time()
        time_since_last_scroll = current_time - self._last_scroll_time
        
        # If scrolling quickly (within 300ms), increase speed
        if time_since_last_scroll < 0.3:
            # Accelerate faster: increase by 3 frames each time, up to 30 frames per scroll
            self._scroll_speed = min(self._scroll_speed + 3, 30)
        else:
            # Reset to base speed if paused
            self._scroll_speed = 3
        
        self._last_scroll_time = current_time
        
        # event.delta is positive for scroll up, negative for scroll down
        if event.delta > 0:
            self._jump_frames(-self._scroll_speed)
        else:
            self._jump_frames(self._scroll_speed)
    
    def _on_annotation_clicked(self, annotation_id: int, frame_idx: int, is_right_click: bool = False):
        """Handle annotation click."""
        # Toggle annotation or parent box based on mouse button
        if is_right_click:
            new_state = self.annotation_manager.toggle_parent_box(annotation_id, frame_idx)
        else:
            new_state = self.annotation_manager.toggle_annotation(annotation_id, frame_idx)
        
        # Reload frame to update display
        annotations = self.annotation_manager.get_frame_annotations(frame_idx)
        is_transition = self.transition_manager.is_in_transition(frame_idx)
        self.canvas.load_frame(frame_idx, annotations, is_transition)
        self._update_labels(frame_idx)
        
        # Show status
        state_str = "visible" if new_state else "hidden"
        action = "Parent box" if is_right_click else "Annotation"
        self._update_status(f"{action} {annotation_id} is now {state_str}", timeout=2000)
    
    def _toggle_hidden_preview(self):
        """Toggle preview mode for hidden annotations."""
        self.canvas.toggle_hidden_preview()
        state = "enabled" if self.canvas.show_hidden_preview else "disabled"
        self._update_status(f"Hidden preview {state}", timeout=2000)
    
    def _toggle_blur_preview(self):
        """Toggle blur preview mode."""
        self.preview_mode = not self.preview_mode
        self.canvas.set_preview_mode(self.preview_mode)
        state = "ON" if self.preview_mode else "OFF"
        self._update_status(f"Blur preview {state}", timeout=2000)
        
        # Reload current frame to apply/remove preview
        frame_idx = self.annotation_manager.frame_indices[self.current_position]
        annotations = self.annotation_manager.get_frame_annotations(frame_idx)
        is_transition = self.transition_manager.is_in_transition(frame_idx)
        self.canvas.load_frame(frame_idx, annotations, is_transition)
        self._update_labels(frame_idx)
    
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
    
    def _export_video(self):
        """Export anonymized video with blurring applied."""
        from tkinter import simpledialog
        import subprocess
        import sys
        
        # Ask for output file path
        output_path = filedialog.asksaveasfilename(
            title="Export Anonymized Video",
            initialdir=str(self.data_dir.parent),
            initialfile=f"{self.data_dir.name}_anonymized.mp4",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("AVI Video", "*.avi"), ("All Files", "*.*")]
        )
        
        if not output_path:
            return
        
        # Ask for blur strength
        blur_strength = simpledialog.askinteger(
            "Blur Strength",
            "Enter blur kernel size (must be odd number):",
            initialvalue=51,
            minvalue=3,
            maxvalue=201
        )
        
        if blur_strength is None:
            return
        
        # Make sure it's odd
        if blur_strength % 2 == 0:
            blur_strength += 1
        
        # Save current state before exporting
        self._save_state()
        
        # Prepare export script path
        export_script = Path(__file__).parent.parent.parent / "export_video.py"
        
        if not export_script.exists():
            messagebox.showerror(
                "Export Script Not Found",
                f"Export script not found at:\n{export_script}"
            )
            return
        
        # Show info dialog
        response = messagebox.askyesno(
            "Export Video",
            f"This will export an anonymized video with:\n\n" +
            f"• OCR boxes blurred (visible ones only)\n" +
            f"• SAM3 masks blurred (visible ones only)\n" +
            f"• Transition frames fully blurred\n" +
            f"• Blur strength: {blur_strength}\n\n" +
            f"Output: {Path(output_path).name}\n\n" +
            f"This may take several minutes. Continue?",
            icon='question'
        )
        
        if not response:
            return
        
        try:
            # Run export script and capture output
            self._update_status("Exporting video... please wait")
            
            # Run in subprocess and capture output
            result = subprocess.run(
                [
                    sys.executable,
                    str(export_script),
                    "--video_dir", str(self.data_dir),
                    "--output", output_path,
                    "--blur_strength", str(blur_strength)
                ],
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            
            if result.returncode == 0:
                messagebox.showinfo(
                    "Export Complete",
                    f"Video exported successfully!\n\n" +
                    f"Output: {output_path}"
                )
                self._update_status("Video export completed", timeout=3000)
            else:
                # Show error details
                error_msg = result.stderr if result.stderr else result.stdout
                messagebox.showerror(
                    "Export Failed",
                    f"Export failed with error:\n\n{error_msg[:500]}"
                )
                self._update_status("Export failed", timeout=3000)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            messagebox.showerror(
                "Export Error", 
                f"Failed to start export:\n{str(e)}\n\n{error_details[:500]}"
            )
            self._update_status("Export failed", timeout=3000)
    
    def _update_labels(self, frame_idx: int):
        """Update info labels."""
        # Frame label with transition indicator
        frame_count = self.annotation_manager.get_frame_count()
        transition_status = ""
        if self.transition_manager.is_in_transition(frame_idx):
            transition = self.transition_manager.get_transition_at_frame(frame_idx)
            transition_status = f" [TRANSITION {transition[0]}-{transition[1]}]"
        elif self.transition_start_frame is not None:
            transition_status = f" [Marking from {self.transition_start_frame}]"
        
        self.frame_label.config(
            text=f"Frame: {self.current_position + 1} / {frame_count} (Index: {frame_idx}){transition_status}"
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
        if self.annotation_manager.modified or self.transition_manager.modified:
            result = messagebox.askyesnocancel(
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save before closing?"
            )
            
            if result is None:  # Cancel
                return
            elif result:  # Yes, save
                self._save_state()
                if self.transition_manager.modified:
                    self._save_transitions()
        
        self.root.quit()
    
    def _mark_transition_start(self):
        """Mark current frame as transition start."""
        self.transition_start_frame = self.annotation_manager.frame_indices[self.current_position]
        self._update_labels(self.annotation_manager.frame_indices[self.current_position])
        self._update_status(f"Marked frame {self.transition_start_frame} as transition start. Navigate to end frame and press 'E'.", timeout=5000)
    
    def _mark_transition_end(self):
        """Mark current frame as transition end and create the range."""
        if self.transition_start_frame is None:
            messagebox.showwarning("No Start Frame", "Please mark a transition start frame first (press 'T')")
            return
        
        end_frame = self.annotation_manager.frame_indices[self.current_position]
        
        if end_frame < self.transition_start_frame:
            messagebox.showerror("Invalid Range", f"End frame ({end_frame}) must be >= start frame ({self.transition_start_frame})")
            return
        
        # Add the transition
        if self.transition_manager.add_transition(self.transition_start_frame, end_frame):
            self._update_status(f"Added transition range: {self.transition_start_frame}-{end_frame}", timeout=3000)
            self.transition_start_frame = None
            self._update_labels(end_frame)
        else:
            messagebox.showerror("Error", f"Failed to add transition range {self.transition_start_frame}-{end_frame}")
    
    def _remove_current_transition(self):
        """Remove the transition range containing the current frame."""
        current_frame = self.annotation_manager.frame_indices[self.current_position]
        
        if self.transition_manager.remove_transition_at_frame(current_frame):
            self._update_status(f"Removed transition at frame {current_frame}", timeout=3000)
            self._update_labels(current_frame)
            # Reload the frame to remove the transition visual indicator
            self._go_to_position(self.current_position)
        else:
            messagebox.showinfo("No Transition", f"Frame {current_frame} is not in any transition range")
    
    def _save_transitions(self):
        """Save transition ranges."""
        if self.transition_manager.save_transitions():
            messagebox.showinfo("Success", "Transitions saved successfully!")
            self._update_status("Transitions saved", timeout=3000)
        else:
            messagebox.showerror("Error", "Failed to save transitions")
    
    def _list_transitions(self):
        """Show a dialog listing all transitions."""
        transitions = self.transition_manager.get_all_transitions()
        
        if not transitions:
            messagebox.showinfo("Transitions", "No transitions defined")
            return
        
        # Create a simple list dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Transition Ranges")
        dialog.geometry("400x400")
        
        # Add scrollable listbox
        frame = ttk.Frame(dialog)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(frame, yscrollcommand=scrollbar.set)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        for i, (start, end) in enumerate(transitions, 1):
            duration = end - start + 1
            listbox.insert(tk.END, f"{i}. Frames {start}-{end} ({duration} frames)")
        
        # Add close button
        close_button = ttk.Button(dialog, text="Close", command=dialog.destroy)
        close_button.pack(pady=(0, 10))
