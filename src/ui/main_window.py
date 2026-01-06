"""
Main Window - primary application window with controls and canvas.
"""
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QSlider, QLabel, QPushButton, QStatusBar, QMessageBox,
    QFileDialog
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut, QAction

from .canvas_widget import CanvasWidget
from ..models.annotation_manager import AnnotationManager


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self, data_dir: Path):
        """
        Initialize the main window.
        
        Args:
            data_dir: Path to data directory
        """
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.frames_dir = self.data_dir / "frames"
        
        # Initialize managers
        self.annotation_manager = AnnotationManager(self.data_dir)
        
        # Current frame position
        self.current_position = 0
        
        # Setup UI
        self.setWindowTitle("TiT Video Annotation Viewer")
        self.setGeometry(100, 100, 1200, 800)
        
        self._create_widgets()
        self._create_layout()
        self._create_menu_bar()
        self._create_shortcuts()
        self._create_status_bar()
        
        # Load data
        self._load_data()
    
    def _create_widgets(self):
        """Create UI widgets."""
        # Canvas
        self.canvas = CanvasWidget(self.frames_dir)
        self.canvas.annotation_clicked.connect(self._on_annotation_clicked)
        
        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(100)
        self.slider.valueChanged.connect(self._on_slider_changed)
        
        # Labels
        self.frame_label = QLabel("Frame: 0 / 0")
        self.stats_label = QLabel("OCR: 0 | SAM3: 0 | Visible: 0 | Hidden: 0")
        
        # Buttons
        self.prev_button = QPushButton("◀ Previous")
        self.prev_button.clicked.connect(self._go_to_previous_frame)
        
        self.next_button = QPushButton("Next ▶")
        self.next_button.clicked.connect(self._go_to_next_frame)
        
        self.toggle_preview_button = QPushButton("Toggle Hidden Preview (H)")
        self.toggle_preview_button.clicked.connect(self._toggle_hidden_preview)
        
        self.save_button = QPushButton("Save State (S)")
        self.save_button.clicked.connect(self._save_state)
        
        self.export_button = QPushButton("Export Visibility")
        self.export_button.clicked.connect(self._export_visibility)
    
    def _create_layout(self):
        """Create and set layout."""
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Main layout
        layout = QVBoxLayout(main_widget)
        
        # Add canvas
        layout.addWidget(self.canvas, stretch=1)
        
        # Info row
        info_layout = QHBoxLayout()
        info_layout.addWidget(self.frame_label)
        info_layout.addStretch()
        info_layout.addWidget(self.stats_label)
        layout.addLayout(info_layout)
        
        # Slider row
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.prev_button)
        slider_layout.addWidget(self.slider, stretch=1)
        slider_layout.addWidget(self.next_button)
        layout.addLayout(slider_layout)
        
        # Button row
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.toggle_preview_button)
        button_layout.addStretch()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.export_button)
        layout.addLayout(button_layout)
    
    def _create_menu_bar(self):
        """Create menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        save_action = QAction("&Save State", self)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self._save_state)
        file_menu.addAction(save_action)
        
        export_action = QAction("&Export Visibility", self)
        export_action.triggered.connect(self._export_visibility)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        toggle_hidden_action = QAction("Toggle &Hidden Preview", self)
        toggle_hidden_action.setShortcut(QKeySequence("H"))
        toggle_hidden_action.triggered.connect(self._toggle_hidden_preview)
        view_menu.addAction(toggle_hidden_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _create_shortcuts(self):
        """Create keyboard shortcuts."""
        # Arrow keys for navigation
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, self._go_to_previous_frame)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, self._go_to_next_frame)
        QShortcut(QKeySequence(Qt.Key.Key_Up), self, lambda: self._jump_frames(-10))
        QShortcut(QKeySequence(Qt.Key.Key_Down), self, lambda: self._jump_frames(10))
        
        # Home/End
        QShortcut(QKeySequence(Qt.Key.Key_Home), self, lambda: self._go_to_position(0))
        QShortcut(QKeySequence(Qt.Key.Key_End), self, 
                 lambda: self._go_to_position(self.annotation_manager.get_frame_count() - 1))
    
    def _create_status_bar(self):
        """Create status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def _load_data(self):
        """Load annotation data and first frame."""
        # Check if frames directory exists
        if not self.frames_dir.exists():
            QMessageBox.critical(
                self,
                "Error",
                f"Frames directory not found: {self.frames_dir}\n\n"
                "Please ensure your data is organized as:\n"
                "data/frames/  (containing 0000.jpg, 0001.jpg, ...)\n"
                "data/annotations.pkl or data/ocr.pkl + data/sam3.pkl"
            )
            return
        
        # Load annotations
        if not self.annotation_manager.load_annotations():
            QMessageBox.warning(
                self,
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
            self.slider.setMaximum(frame_count - 1)
            self._go_to_position(0)
            self.status_bar.showMessage(f"Loaded {frame_count} frames")
        else:
            self.status_bar.showMessage("No frames loaded")
    
    def _go_to_position(self, position: int):
        """
        Go to specific position in frame list.
        
        Args:
            position: Position index (0-based)
        """
        frame_count = self.annotation_manager.get_frame_count()
        if frame_count == 0:
            return
        
        # Clamp position
        position = max(0, min(position, frame_count - 1))
        self.current_position = position
        
        # Update slider without triggering signal
        self.slider.blockSignals(True)
        self.slider.setValue(position)
        self.slider.blockSignals(False)
        
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
    
    def _on_slider_changed(self, value: int):
        """Handle slider value change."""
        self._go_to_position(value)
    
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
        self.status_bar.showMessage(f"Annotation {annotation_id} is now {state_str}", 2000)
    
    def _toggle_hidden_preview(self):
        """Toggle preview mode for hidden annotations."""
        self.canvas.toggle_hidden_preview()
        state = "enabled" if self.canvas.show_hidden_preview else "disabled"
        self.status_bar.showMessage(f"Hidden preview {state}", 2000)
    
    def _save_state(self):
        """Save current state."""
        if self.annotation_manager.save_state():
            QMessageBox.information(self, "Success", "State saved successfully!")
            self.status_bar.showMessage("State saved", 3000)
        else:
            QMessageBox.critical(self, "Error", "Failed to save state")
    
    def _export_visibility(self):
        """Export visibility state."""
        # Ask for file location
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Visibility State",
            str(self.data_dir / "visibility.pkl"),
            "Pickle Files (*.pkl);;All Files (*)"
        )
        
        if filepath:
            if self.annotation_manager.export_visibility_state(Path(filepath)):
                QMessageBox.information(self, "Success", 
                                      f"Visibility state exported to:\n{filepath}")
                self.status_bar.showMessage("Visibility exported", 3000)
            else:
                QMessageBox.critical(self, "Error", "Failed to export visibility state")
    
    def _update_labels(self, frame_idx: int):
        """Update info labels."""
        # Frame label
        frame_count = self.annotation_manager.get_frame_count()
        self.frame_label.setText(
            f"Frame: {self.current_position + 1} / {frame_count} (Index: {frame_idx})"
        )
        
        # Stats label
        stats = self.annotation_manager.get_statistics(frame_idx)
        self.stats_label.setText(
            f"OCR: {stats['ocr']} | SAM3: {stats['sam3']} | "
            f"Visible: {stats['visible']} | Hidden: {stats['hidden']}"
        )
    
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About TiT Annotation Viewer",
            "<h3>TiT Video Annotation Viewer</h3>"
            "<p>Interactive viewer for OCR and SAM3 annotations on video frames.</p>"
            "<p><b>Controls:</b></p>"
            "<ul>"
            "<li>Click on visible annotation to hide it</li>"
            "<li>Hover over hidden annotation area and click to show it</li>"
            "<li>Use arrow keys to navigate frames</li>"
            "<li>Press H to toggle hidden preview mode</li>"
            "<li>Press S to save state</li>"
            "</ul>"
            "<p>Version 1.0 - January 2026</p>"
        )
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Check for unsaved changes
        if self.annotation_manager.modified:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save before closing?",
                QMessageBox.StandardButton.Save | 
                QMessageBox.StandardButton.Discard | 
                QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.Save:
                self._save_state()
                event.accept()
            elif reply == QMessageBox.StandardButton.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
