"""
Video Processor GUI - Frontend for batch video processing pipeline

This GUI allows users to:
- Select video files or folders
- Configure processing parameters
- Run the complete pipeline (frame extraction, OCR, SAM3, transitions)
- Monitor progress
- View results
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import threading
import queue
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from process_videos import process_single_video, process_multiple_videos


class VideoProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Processor - Batch Processing")
        self.root.geometry("900x700")
        
        # State
        self.video_paths = []
        self.output_dir = None
        self.dict_path = None
        self.processing = False
        self.log_queue = queue.Queue()
        
        # Create UI
        self._create_ui()
        
        # Start log monitor
        self._monitor_logs()
    
    def _create_ui(self):
        """Create the main UI layout"""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)  # Log area gets extra space
        
        # === INPUT SECTION ===
        input_frame = ttk.LabelFrame(main_frame, text="Input", padding="10")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)
        
        # Video selection
        ttk.Label(input_frame, text="Videos:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.video_label = ttk.Label(input_frame, text="No videos selected", 
                                     foreground="gray")
        self.video_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        
        video_buttons = ttk.Frame(input_frame)
        video_buttons.grid(row=0, column=2, sticky=tk.E, padx=(10, 0))
        ttk.Button(video_buttons, text="Select File(s)", 
                  command=self._select_video_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(video_buttons, text="Select Folder", 
                  command=self._select_video_folder).pack(side=tk.LEFT, padx=2)
        
        # Output directory
        ttk.Label(input_frame, text="Output Dir:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.output_label = ttk.Label(input_frame, text="No output directory selected", 
                                      foreground="gray")
        self.output_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        ttk.Button(input_frame, text="Browse", 
                  command=self._select_output_dir).grid(row=1, column=2, sticky=tk.E, padx=(10, 0))
        
        # Dictionary file
        ttk.Label(input_frame, text="Names Dict:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.dict_label = ttk.Label(input_frame, text="No dictionary selected", 
                                    foreground="gray")
        self.dict_label.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        ttk.Button(input_frame, text="Browse", 
                  command=self._select_dict_file).grid(row=2, column=2, sticky=tk.E, padx=(10, 0))
        
        # === PARAMETERS SECTION ===
        params_frame = ttk.LabelFrame(main_frame, text="Processing Parameters", padding="10")
        params_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        params_frame.columnconfigure(1, weight=1)
        params_frame.columnconfigure(3, weight=1)
        
        # Frame step
        ttk.Label(params_frame, text="Frame Step:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.frame_step_var = tk.StringVar(value="1")
        ttk.Entry(params_frame, textvariable=self.frame_step_var, width=10).grid(
            row=0, column=1, sticky=tk.W, padx=(10, 20), pady=2)
        
        # OCR languages
        ttk.Label(params_frame, text="OCR Languages:").grid(row=0, column=2, sticky=tk.W, pady=2)
        self.ocr_langs_var = tk.StringVar(value="en de")
        ttk.Entry(params_frame, textvariable=self.ocr_langs_var, width=20).grid(
            row=0, column=3, sticky=tk.W, padx=(10, 0), pady=2)
        
        # SAM3 batch size
        ttk.Label(params_frame, text="SAM3 Batch:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.sam3_batch_var = tk.StringVar(value="2")
        ttk.Entry(params_frame, textvariable=self.sam3_batch_var, width=10).grid(
            row=1, column=1, sticky=tk.W, padx=(10, 20), pady=2)
        
        # SAM3 device
        ttk.Label(params_frame, text="SAM3 Device:").grid(row=1, column=2, sticky=tk.W, pady=2)
        self.sam3_device_var = tk.StringVar(value="cuda")
        device_combo = ttk.Combobox(params_frame, textvariable=self.sam3_device_var, 
                                    values=["cuda", "cpu"], state="readonly", width=10)
        device_combo.grid(row=1, column=3, sticky=tk.W, padx=(10, 0), pady=2)
        
        # SAM3 prompt
        ttk.Label(params_frame, text="SAM3 Prompt:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.sam3_prompt_var = tk.StringVar(value="profile image, profile picture")
        ttk.Entry(params_frame, textvariable=self.sam3_prompt_var, width=40).grid(
            row=2, column=1, columnspan=3, sticky=(tk.W, tk.E), padx=(10, 0), pady=2)
        device_combo.grid(row=3, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        
        # === OPTIONS SECTION ===
        options_frame = ttk.LabelFrame(main_frame, text="Processing Options", padding="10")
        options_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.skip_frames_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Skip frame extraction (use existing)", 
                       variable=self.skip_frames_var).pack(side=tk.LEFT, padx=(0, 15))
        
        self.skip_ocr_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Skip OCR", 
                       variable=self.skip_ocr_var).pack(side=tk.LEFT, padx=(0, 15))
        
        self.skip_sam3_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Skip SAM3", 
                       variable=self.skip_sam3_var).pack(side=tk.LEFT, padx=(0, 15))
        
        self.skip_transitions_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Skip transitions", 
                       variable=self.skip_transitions_var).pack(side=tk.LEFT)
        
        # === CONTROL SECTION ===
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.start_button = ttk.Button(control_frame, text="Start Processing", 
                                       command=self._start_processing, state=tk.DISABLED)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(control_frame, text="Stop", 
                                      command=self._stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_log_button = ttk.Button(control_frame, text="Clear Log", 
                                           command=self._clear_log)
        self.clear_log_button.pack(side=tk.LEFT)
        
        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        
        # === LOG SECTION ===
        log_frame = ttk.LabelFrame(main_frame, text="Processing Log", padding="5")
        log_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=20, wrap=tk.WORD, 
                                                  state=tk.DISABLED, font=("Consolas", 9))
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        self.status_label = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
    
    def _select_video_files(self):
        """Select individual video files"""
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm"),
            ("All files", "*.*")
        ]
        files = filedialog.askopenfilenames(title="Select Video Files", filetypes=filetypes)
        if files:
            # Handle both tuple and space-separated string (Windows quirk)
            if isinstance(files, str):
                # Single string returned - split by spaces but handle paths with spaces
                files = self.root.tk.splitlist(files)
            
            self.video_paths = [Path(f) for f in files]
            print(f"Selected {len(self.video_paths)} video file(s):")
            for vp in self.video_paths:
                print(f"  - {vp}")
            self._update_video_label()
            self._check_ready()
    
    def _select_video_folder(self):
        """Select folder containing videos"""
        folder = filedialog.askdirectory(title="Select Video Folder")
        if folder:
            folder_path = Path(folder)
            # Find all video files
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
            self.video_paths = []
            for ext in video_extensions:
                self.video_paths.extend(folder_path.glob(f'*{ext}'))
                self.video_paths.extend(folder_path.glob(f'*{ext.upper()}'))
            
            if self.video_paths:
                self._update_video_label()
                self._check_ready()
            else:
                messagebox.showwarning("No Videos", "No video files found in the selected folder")
    
    def _select_output_dir(self):
        """Select output directory"""
        folder = filedialog.askdirectory(title="Select Output Directory")
        if folder:
            self.output_dir = Path(folder)
            self.output_label.config(text=str(self.output_dir), foreground="black")
            self._check_ready()
    
    def _select_dict_file(self):
        """Select dictionary JSON file"""
        filetypes = [("JSON files", "*.json"), ("All files", "*.*")]
        file = filedialog.askopenfilename(title="Select Names Dictionary", filetypes=filetypes)
        if file:
            self.dict_path = Path(file)
            self.dict_label.config(text=self.dict_path.name, foreground="black")
            self._check_ready()
    
    def _update_video_label(self):
        """Update the video label with count and first filename"""
        if not self.video_paths:
            self.video_label.config(text="No videos selected", foreground="gray")
        elif len(self.video_paths) == 1:
            self.video_label.config(text=self.video_paths[0].name, foreground="black")
        else:
            self.video_label.config(
                text=f"{len(self.video_paths)} videos (first: {self.video_paths[0].name})",
                foreground="black"
            )
    
    def _check_ready(self):
        """Check if all requirements are met to enable start button"""
        if self.video_paths and self.output_dir and self.dict_path and not self.processing:
            self.start_button.config(state=tk.NORMAL)
        else:
            self.start_button.config(state=tk.DISABLED)
    
    def _start_processing(self):
        """Start the processing pipeline in a background thread"""
        if self.processing:
            return
        
        self.processing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress.start()
        self._log("Starting video processing pipeline...")
        
        # Collect parameters
        try:
            frame_step = int(self.frame_step_var.get())
            sam3_batch_size = int(self.sam3_batch_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Frame step and SAM3 batch must be integers")
            self._stop_processing()
            return
        
        ocr_languages = self.ocr_langs_var.get().split()
        sam3_prompt = self.sam3_prompt_var.get()
        sam3_device = self.sam3_device_var.get()
        
        # Start processing in background thread
        thread = threading.Thread(
            target=self._processing_thread,
            args=(frame_step, ocr_languages, sam3_batch_size, 
                  sam3_prompt, sam3_device),
            daemon=True
        )
        thread.start()
    
    def _processing_thread(self, frame_step, ocr_languages, 
                          sam3_batch_size, sam3_prompt, sam3_device):
        """Background thread for processing videos"""
        # Redirect stdout to log
        import io
        
        class LogCapture(io.StringIO):
            def __init__(self, log_queue):
                super().__init__()
                self.log_queue = log_queue
            
            def write(self, message):
                if message.strip():
                    self.log_queue.put(message)
                return len(message)
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = LogCapture(self.log_queue)
        sys.stderr = LogCapture(self.log_queue)
        
        try:
            if len(self.video_paths) == 1:
                process_single_video(
                    video_path=self.video_paths[0],
                    output_base_dir=self.output_dir,
                    dict_path=self.dict_path,
                    ocr_languages=ocr_languages,
                    sam3_prompt=sam3_prompt,
                    sam3_batch_size=sam3_batch_size,
                    sam3_device=sam3_device,
                    frame_step=frame_step,
                    extract_frames=not self.skip_frames_var.get(),
                    run_ocr=not self.skip_ocr_var.get(),
                    run_sam3=not self.skip_sam3_var.get(),
                    run_transitions=not self.skip_transitions_var.get()
                )
            else:
                process_multiple_videos(
                    video_paths=self.video_paths,
                    output_base_dir=self.output_dir,
                    dict_path=self.dict_path,
                    ocr_languages=ocr_languages,
                    sam3_prompt=sam3_prompt,
                    sam3_batch_size=sam3_batch_size,
                    sam3_device=sam3_device,
                    frame_step=frame_step,
                    extract_frames=not self.skip_frames_var.get(),
                    run_ocr=not self.skip_ocr_var.get(),
                    run_sam3=not self.skip_sam3_var.get(),
                    run_transitions=not self.skip_transitions_var.get()
                )
            
            self.log_queue.put("\n✓ Processing complete!")
        except Exception as e:
            self.log_queue.put(f"\n✗ Error: {str(e)}")
            import traceback
            self.log_queue.put(traceback.format_exc())
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            self.log_queue.put("__DONE__")
    
    def _stop_processing(self):
        """Stop the processing (note: this just updates UI, doesn't actually stop the thread)"""
        self.processing = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress.stop()
        self._check_ready()
    
    def _clear_log(self):
        """Clear the log text"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def _log(self, message):
        """Add a message to the log"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def _monitor_logs(self):
        """Monitor the log queue and update the UI"""
        try:
            while True:
                message = self.log_queue.get_nowait()
                if message == "__DONE__":
                    self._stop_processing()
                else:
                    self._log(message.rstrip())
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self._monitor_logs)


def main():
    root = tk.Tk()
    app = VideoProcessorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
