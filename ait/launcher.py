"""
AiT Application Launcher - Unified entry point for all tools

Launch the Video Processor (batch processing) or Annotation Viewer
from a single interface. Both can run simultaneously.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import subprocess
import sys
from pathlib import Path

from ait.export_queue import (
    ExportQueueController,
    ExportQueueError,
    validate_processed_folder,
)


class AiTLauncher:
    # How often (ms) the Tk event loop polls the export queue for progress.
    POLL_INTERVAL_MS = 400

    def __init__(self, root):
        self.root = root
        self.root.title("AiT Application Launcher")
        self.root.geometry("640x860")
        self.root.minsize(560, 700)

        # Track running processes (separately launched processor/viewer tools)
        self.processes = []

        # Selected video folder for annotation viewer
        self.video_folder = None

        # Sequential export queue (runs one export_video subprocess at a time)
        self.queue_controller = ExportQueueController()
        self._poll_job = None

        self._create_ui()

        # Reflect the (empty) initial queue state in the controls.
        self._refresh_queue()

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
            text="AiT Video Anonymization Suite",
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
        
        # === VIDEO EXPORT QUEUE SECTION ===
        export_frame = ttk.LabelFrame(main_frame, text="Export Queue", padding="15")
        export_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        ttk.Label(
            export_frame,
            text="Queue processed video folders and export them one after "
                 "another.\n" +
                 "Each job records its blur strength when added. Exports run "
                 "one at a time.",
            justify=tk.LEFT,
            foreground="gray"
        ).pack(anchor=tk.W, pady=(0, 10))

        # Blur strength setting (captured per job when Add is pressed)
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
            blur_frame,
            text="Add to Queue...",
            command=self._add_to_queue
        ).pack(side=tk.RIGHT)

        # Queue table showing each job's source, output, blur, and status
        columns = ("source", "output", "blur", "status")
        self.queue_tree = ttk.Treeview(
            export_frame, columns=columns, show="headings", height=6
        )
        self.queue_tree.heading("source", text="Source Folder")
        self.queue_tree.heading("output", text="Output File")
        self.queue_tree.heading("blur", text="Blur")
        self.queue_tree.heading("status", text="Status")
        self.queue_tree.column("source", width=180, anchor=tk.W)
        self.queue_tree.column("output", width=180, anchor=tk.W)
        self.queue_tree.column("blur", width=45, anchor=tk.CENTER)
        self.queue_tree.column("status", width=85, anchor=tk.W)
        self.queue_tree.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        # Queue control buttons
        controls = ttk.Frame(export_frame)
        controls.pack(fill=tk.X)

        self.start_queue_button = ttk.Button(
            controls, text="Start Queue", command=self._start_queue
        )
        self.start_queue_button.pack(side=tk.LEFT, padx=(0, 5))

        self.cancel_active_button = ttk.Button(
            controls, text="Cancel Active", command=self._cancel_active,
            state=tk.DISABLED
        )
        self.cancel_active_button.pack(side=tk.LEFT, padx=(0, 5))

        self.remove_button = ttk.Button(
            controls, text="Remove Selected", command=self._remove_selected
        )
        self.remove_button.pack(side=tk.LEFT, padx=(0, 5))

        self.clear_pending_button = ttk.Button(
            controls, text="Clear Pending", command=self._clear_pending
        )
        self.clear_pending_button.pack(side=tk.LEFT, padx=(0, 5))

        self.queue_status_label = ttk.Label(
            export_frame, text="Queue empty", foreground="gray"
        )
        self.queue_status_label.pack(anchor=tk.W, pady=(8, 0))

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
            # Launch as separate process via module
            process = subprocess.Popen(
                [sys.executable, "-m", "ait.video_processor_gui"],
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
            # Launch as separate process via module
            process = subprocess.Popen(
                [sys.executable, "-m", "ait.viewer.main", str(self.video_folder)],
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
            )
            self.processes.append(("Annotation Viewer", process))
            
            self.status_label.config(text=f"Annotation Viewer launched: {self.video_folder.name}")
        except Exception as e:
            messagebox.showerror("Launch Error", f"Failed to launch Annotation Viewer:\n{str(e)}")
    
    # === EXPORT QUEUE ===

    def _add_to_queue(self):
        """Select a processed folder + output path and enqueue an export job."""
        folder = filedialog.askdirectory(
            title="Select Processed Video Folder "
                  "(containing frames/ and state.pkl/ocr.pkl/sam3.pkl)",
            mustexist=True
        )
        if not folder:
            return

        folder_path = Path(folder)

        # Validate early so the user gets specific feedback before choosing an
        # output file. The controller re-validates on add.
        ok, reason = validate_processed_folder(folder_path)
        if not ok:
            messagebox.showerror(
                "Invalid Folder",
                f"{reason}\n\nSelected: {folder_path}"
            )
            return

        # Ask for an explicit output path so the destination is predictable.
        output_path = filedialog.asksaveasfilename(
            title="Choose Export Output File",
            initialdir=str(folder_path.parent),
            initialfile=f"{folder_path.name}_anonymized.mp4",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("AVI Video", "*.avi"), ("All Files", "*.*")]
        )
        if not output_path:
            return

        # Capture blur strength now (odd), stored on the job.
        blur_strength = self.blur_strength_var.get()
        if blur_strength % 2 == 0:
            blur_strength += 1
            self.blur_strength_var.set(blur_strength)

        try:
            self.queue_controller.add(folder_path, output_path, blur_strength)
        except ExportQueueError as e:
            messagebox.showerror("Cannot Add to Queue", str(e))
            return

        self._refresh_queue()
        self.status_label.config(text=f"Queued: {folder_path.name}")

    def _remove_selected(self):
        """Remove the selected pending job from the queue."""
        item = self._selected_item()
        if item is None:
            messagebox.showinfo("No Selection", "Select a queued job to remove.")
            return
        if not self.queue_controller.remove(item):
            messagebox.showwarning(
                "Cannot Remove",
                "Only pending jobs can be removed. Use 'Cancel Active' to stop "
                "a running export."
            )
            return
        self._refresh_queue()

    def _clear_pending(self):
        """Remove every pending job, leaving running/finished ones intact."""
        removed = self.queue_controller.clear_pending()
        self._refresh_queue()
        self.status_label.config(text=f"Cleared {removed} pending job(s)")

    def _start_queue(self):
        """Start sequential export of the pending jobs."""
        if self.queue_controller.is_active():
            return
        if not self.queue_controller.has_pending():
            messagebox.showinfo(
                "Nothing to Export", "Add at least one job to the queue first."
            )
            return
        self.queue_controller.start()
        self._start_polling()
        self._refresh_queue()
        self.status_label.config(text="Export queue started")

    def _cancel_active(self):
        """Cancel the currently running export; pending jobs continue."""
        if not self.queue_controller.is_active():
            return
        confirm = messagebox.askyesno(
            "Cancel Active Export",
            "Stop the export that is currently running?\n\n"
            "Remaining pending jobs will continue.",
            icon='warning'
        )
        if not confirm:
            return
        self.queue_controller.cancel_active(continue_pending=True)
        self._refresh_queue()
        self.status_label.config(
            text="Cancelling active export; the next job will start after it exits"
        )

    def _start_polling(self):
        """Ensure the Tk after-based poll loop is running."""
        if self._poll_job is None:
            self._poll_job = self.root.after(
                self.POLL_INTERVAL_MS, self._poll_queue
            )

    def _stop_polling(self):
        """Cancel the scheduled poll callback, if any."""
        if self._poll_job is not None:
            try:
                self.root.after_cancel(self._poll_job)
            except Exception:
                pass
            self._poll_job = None

    def _poll_queue(self):
        """Non-blocking poll: advance the queue and refresh the UI."""
        self._poll_job = None
        self.queue_controller.poll()
        self._refresh_queue()
        if self.queue_controller.is_active():
            # Keep polling while an export subprocess is running.
            self._poll_job = self.root.after(
                self.POLL_INTERVAL_MS, self._poll_queue
            )

    def _selected_item(self):
        """Return the ExportItem for the selected tree row, or None."""
        selection = self.queue_tree.selection()
        if not selection:
            return None
        iid = selection[0]
        for item in self.queue_controller.items:
            if str(item._id) == iid:
                return item
        return None

    def _refresh_queue(self):
        """Rebuild the queue table and update the control states."""
        selected = set(self.queue_tree.selection())
        self.queue_tree.delete(*self.queue_tree.get_children())
        for item in self.queue_controller.items:
            iid = str(item._id)
            self.queue_tree.insert(
                "", tk.END, iid=iid,
                values=(
                    str(item.source_dir),
                    str(item.output_path),
                    item.blur_strength,
                    item.error if item.status == "failed" and item.error else item.status,
                )
            )
            if iid in selected and self.queue_tree.exists(iid):
                self.queue_tree.selection_add(iid)
        self._update_queue_controls()

    def _update_queue_controls(self):
        """Keep the queue buttons/status coherent with controller state."""
        active = self.queue_controller.is_active()
        has_pending = self.queue_controller.has_pending()

        self.start_queue_button.config(
            state=tk.DISABLED if (active or not has_pending) else tk.NORMAL
        )
        self.cancel_active_button.config(
            state=tk.NORMAL if active else tk.DISABLED
        )

        if self.queue_controller.items:
            self.queue_status_label.config(
                text=self.queue_controller.summary_text(), foreground="black"
            )
        else:
            self.queue_status_label.config(text="Queue empty", foreground="gray")

    def _on_close(self):
        """Handle window close event"""
        # Separately launched processor/viewer processes still running.
        running = [name for name, proc in self.processes if proc.poll() is None]
        queue_busy = (
            self.queue_controller.is_active()
            or self.queue_controller.has_pending()
        )

        warnings = list(running)
        if queue_busy:
            warnings.append("Export Queue")

        if warnings:
            response = messagebox.askyesno(
                "Applications Running",
                "The following are still running or pending:\n" +
                f"{', '.join(warnings)}\n\n" +
                "Close launcher anyway? Any active export will be stopped and "
                "pending exports cancelled.",
                icon='warning'
            )
            if not response:
                return

        # Tear down the export queue so no subprocess or hidden pending queue
        # is left orphaned after the window is destroyed.
        self._stop_polling()
        self.queue_controller.shutdown()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = AiTLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
