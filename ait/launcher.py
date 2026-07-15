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
    discover_pipeline_folders,
)
from ait.config import get_last_browse_dir, set_last_browse_dir


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

        # Last mask choice made in the export dialog; used as the default for
        # folders the annotation viewer never recorded a choice for.
        self._last_mask_choice = None

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

        # Queue table showing each job's source, output, blur, masks, delete
        # flag, and status
        columns = ("source", "output", "blur", "masks", "del", "status")
        self.queue_tree = ttk.Treeview(
            export_frame, columns=columns, show="headings", height=6
        )
        self.queue_tree.heading("source", text="Source Folder")
        self.queue_tree.heading("output", text="Output File")
        self.queue_tree.heading("blur", text="Blur")
        self.queue_tree.heading("masks", text="Masks")
        self.queue_tree.heading("del", text="Del")
        self.queue_tree.heading("status", text="Status")
        self.queue_tree.column("source", width=150, anchor=tk.W)
        self.queue_tree.column("output", width=150, anchor=tk.W)
        self.queue_tree.column("blur", width=40, anchor=tk.CENTER)
        self.queue_tree.column("masks", width=65, anchor=tk.CENTER)
        self.queue_tree.column("del", width=35, anchor=tk.CENTER)
        self.queue_tree.column("status", width=80, anchor=tk.W)
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

    def _default_mask_choice(self, folder_path, has_original, has_circular):
        """Default masks selection for the export dialog.

        Priority: the choice recorded by the annotation viewer for this video
        (mask_choice.txt), then whichever variant is the only one available,
        then the last choice made in this launcher session, then original.
        """
        choice_file = folder_path / "mask_choice.txt"
        if choice_file.exists():
            try:
                choice = choice_file.read_text().strip().lower()
            except OSError:
                choice = ""
            if choice == "circular" and has_circular:
                return "circular"
            if choice == "original" and has_original:
                return "original"
        if has_circular and not has_original:
            return "circular"
        if has_original and not has_circular:
            return "original"
        if self._last_mask_choice in ("original", "circular"):
            return self._last_mask_choice
        return "original"

    @staticmethod
    def _default_output_path(folder_path):
        """Default export destination: sibling of the pipeline folder.

        Placed right next to the folder that holds the pipeline files, named
        ``<folder>_anonymized.mp4``.
        """
        return folder_path.parent / f"{folder_path.name}_anonymized.mp4"

    def _ask_export_options(self, folder_path):
        """Modal dialog: output file, SAM3 masks, and delete-after for one job.

        Returns ``(output_path, sam3_source, delete_source)`` or ``None`` if
        cancelled. ``sam3_source`` is ``"auto"`` when reviewed state.pkl masks
        apply.
        """
        state_exists = (folder_path / "state.pkl").exists()
        has_original = (folder_path / "sam3.pkl").exists()
        has_circular = (folder_path / "sam3_circular.pkl").exists()

        dialog = tk.Toplevel(self.root)
        dialog.title("Choose Export Output File")
        dialog.transient(self.root)
        dialog.resizable(False, False)

        frame = ttk.Frame(dialog, padding="15")
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frame, text=f"Source: {folder_path.name}", foreground="gray"
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))

        # Output file row (pre-filled with the default so no navigation needed)
        ttk.Label(frame, text="Output file:").grid(row=1, column=0, sticky=tk.W)
        output_var = tk.StringVar(value=str(self._default_output_path(folder_path)))
        output_entry = ttk.Entry(frame, textvariable=output_var, width=48)
        output_entry.grid(row=1, column=1, sticky=tk.EW, padx=5)

        def browse_output():
            selected = filedialog.asksaveasfilename(
                parent=dialog,
                title="Choose Export Output File",
                initialdir=str(Path(output_var.get()).parent),
                initialfile=Path(output_var.get()).name,
                defaultextension=".mp4",
                filetypes=[("MP4 Video", "*.mp4"), ("AVI Video", "*.avi"),
                           ("All Files", "*.*")]
            )
            if selected:
                output_var.set(selected)

        ttk.Button(frame, text="Browse...", command=browse_output).grid(
            row=1, column=2
        )

        # SAM3 mask choice row
        ttk.Label(frame, text="SAM3 masks:").grid(
            row=2, column=0, sticky=tk.W, pady=(10, 0)
        )
        mask_var = tk.StringVar(
            value=self._default_mask_choice(folder_path, has_original, has_circular)
        )
        mask_frame = ttk.Frame(frame)
        mask_frame.grid(row=2, column=1, columnspan=2, sticky=tk.W, pady=(10, 0))

        original_radio = ttk.Radiobutton(
            mask_frame, text="Original", variable=mask_var, value="original"
        )
        original_radio.pack(side=tk.LEFT, padx=(0, 10))
        circular_radio = ttk.Radiobutton(
            mask_frame, text="Circular", variable=mask_var, value="circular"
        )
        circular_radio.pack(side=tk.LEFT)

        note = None
        if state_exists:
            # Reviewed annotations already contain the masks chosen in the
            # viewer; the export ignores the mask option in that case.
            original_radio.config(state=tk.DISABLED)
            circular_radio.config(state=tk.DISABLED)
            note = ("Masks reviewed in the annotation viewer (state.pkl) "
                    "will be used.")
        else:
            if not has_original:
                original_radio.config(state=tk.DISABLED)
            if not has_circular:
                circular_radio.config(state=tk.DISABLED)
            if (folder_path / "mask_choice.txt").exists():
                note = "Pre-selected from your annotation viewer choice."
        if note:
            ttk.Label(frame, text=note, foreground="gray").grid(
                row=3, column=1, columnspan=2, sticky=tk.W, pady=(4, 0)
            )

        # Delete-after-export option (destructive; off by default)
        delete_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame,
            text="Delete pipeline folder after successful export",
            variable=delete_var,
        ).grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=(12, 0))

        # Buttons
        result = {}
        buttons = ttk.Frame(frame)
        buttons.grid(row=5, column=0, columnspan=3, sticky=tk.E, pady=(15, 0))

        def on_add():
            output = output_var.get().strip()
            if not output:
                messagebox.showerror(
                    "Missing Output", "Choose an output file.", parent=dialog
                )
                return
            result["output"] = output
            result["masks"] = "auto" if state_exists else mask_var.get()
            result["delete"] = delete_var.get()
            dialog.destroy()

        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(
            side=tk.RIGHT, padx=(5, 0)
        )
        ttk.Button(buttons, text="Add to Queue", command=on_add).pack(
            side=tk.RIGHT
        )

        frame.columnconfigure(1, weight=1)
        dialog.bind("<Return>", lambda e: on_add())
        dialog.bind("<Escape>", lambda e: dialog.destroy())
        dialog.grab_set()
        output_entry.focus_set()
        self.root.wait_window(dialog)

        if "output" not in result:
            return None
        if not state_exists:
            self._last_mask_choice = result["masks"]
        return result["output"], result["masks"], result["delete"]

    def _ask_batch_export_options(self, folders):
        """Modal dialog to enqueue several discovered pipeline folders at once.

        Output paths default to each folder's sibling ``<name>_anonymized.mp4``
        so no per-video navigation is needed. Returns a list of
        ``(folder, output_path, sam3_source, delete_source)`` tuples, or
        ``None`` if cancelled.
        """
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Multiple Exports to Queue")
        dialog.transient(self.root)

        frame = ttk.Frame(dialog, padding="15")
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frame,
            text=f"Found {len(folders)} processed folder(s). Each exports to "
                 f"<name>_anonymized.mp4 next to its folder.",
            foreground="gray",
            wraplength=520,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 10))

        # Scrollable checklist of discovered folders
        list_container = ttk.Frame(frame)
        list_container.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(list_container, height=180, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            list_container, orient=tk.VERTICAL, command=canvas.yview
        )
        inner = ttk.Frame(canvas)
        inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        check_vars = []
        for folder in folders:
            var = tk.BooleanVar(value=True)
            check_vars.append(var)
            ttk.Checkbutton(inner, text=str(folder), variable=var).pack(
                anchor=tk.W, pady=1
            )

        # Global SAM3 mask choice for the whole batch
        mask_frame = ttk.Frame(frame)
        mask_frame.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(mask_frame, text="SAM3 masks:").pack(side=tk.LEFT, padx=(0, 8))
        mask_var = tk.StringVar(value="auto")
        for text, value in (("Auto (per folder)", "auto"),
                            ("Original", "original"),
                            ("Circular", "circular")):
            ttk.Radiobutton(
                mask_frame, text=text, variable=mask_var, value=value
            ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(
            frame,
            text="Auto follows each folder's annotation-viewer choice, else "
                 "reviewed state.pkl masks, else original.",
            foreground="gray",
            wraplength=520,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(4, 0))

        # Global delete-after option (destructive; off by default)
        delete_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame,
            text="Delete each pipeline folder after its successful export",
            variable=delete_var,
        ).pack(anchor=tk.W, pady=(12, 0))

        result = {}
        buttons = ttk.Frame(frame)
        buttons.pack(fill=tk.X, pady=(15, 0))

        def on_add():
            chosen = [
                folder for folder, var in zip(folders, check_vars) if var.get()
            ]
            if not chosen:
                messagebox.showinfo(
                    "Nothing Selected",
                    "Select at least one folder to add.",
                    parent=dialog,
                )
                return
            result["folders"] = chosen
            result["masks"] = mask_var.get()
            result["delete"] = delete_var.get()
            dialog.destroy()

        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(
            side=tk.RIGHT, padx=(5, 0)
        )
        ttk.Button(buttons, text="Add Selected to Queue", command=on_add).pack(
            side=tk.RIGHT
        )

        dialog.bind("<Escape>", lambda e: dialog.destroy())
        dialog.grab_set()
        self.root.wait_window(dialog)

        if "folders" not in result:
            return None
        if result["masks"] in ("original", "circular"):
            self._last_mask_choice = result["masks"]
        return [
            (
                folder,
                self._default_output_path(folder),
                result["masks"],
                result["delete"],
            )
            for folder in result["folders"]
        ]

    def _add_to_queue(self):
        """Select folder(s) and enqueue one or many export jobs.

        The chosen folder may be a single processed-video folder or a parent
        holding many (e.g. a mirrored output tree); all pipeline folders found
        underneath are offered for batch enqueue.
        """
        folder = filedialog.askdirectory(
            title="Select a Processed Folder or a Parent of Several",
            mustexist=True,
            initialdir=get_last_browse_dir(),
        )
        if not folder:
            return

        folder_path = Path(folder)
        set_last_browse_dir(folder_path)

        pipeline_folders = discover_pipeline_folders(folder_path)
        if not pipeline_folders:
            messagebox.showerror(
                "No Processed Folders",
                "No folder containing frames/ and state.pkl/ocr.pkl/sam3.pkl "
                f"was found in:\n\n{folder_path}"
            )
            return

        # Capture blur strength now (odd), shared by every job added here.
        blur_strength = self.blur_strength_var.get()
        if blur_strength % 2 == 0:
            blur_strength += 1
            self.blur_strength_var.set(blur_strength)

        if len(pipeline_folders) == 1:
            options = self._ask_export_options(pipeline_folders[0])
            if options is None:
                return
            output_path, sam3_source, delete_source = options
            jobs = [(pipeline_folders[0], output_path, sam3_source, delete_source)]
        else:
            jobs = self._ask_batch_export_options(pipeline_folders)
            if jobs is None:
                return

        added, errors = 0, []
        for source, output_path, sam3_source, delete_source in jobs:
            try:
                self.queue_controller.add(
                    source, output_path, blur_strength,
                    sam3_source=sam3_source,
                    delete_source_after=delete_source,
                )
                added += 1
            except ExportQueueError as e:
                errors.append(f"{Path(source).name}: {e}")

        self._refresh_queue()
        if errors:
            messagebox.showwarning(
                "Some Jobs Not Added",
                f"Added {added} job(s). Skipped {len(errors)}:\n\n"
                + "\n".join(errors)
            )
        self.status_label.config(text=f"Queued {added} job(s)")

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
            masks_display = (
                "reviewed" if item.sam3_source == "auto" else item.sam3_source
            )
            self.queue_tree.insert(
                "", tk.END, iid=iid,
                values=(
                    str(item.source_dir),
                    str(item.output_path),
                    item.blur_strength,
                    masks_display,
                    "yes" if item.delete_source_after else "",
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
