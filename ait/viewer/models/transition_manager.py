"""
Transition Manager - handles loading, saving, and managing transition ranges.
"""
from pathlib import Path
from typing import List, Tuple, Optional


class TransitionManager:
    """Manages frame transition ranges (frames to be blurred)."""
    
    def __init__(self, data_dir: Path):
        """
        Initialize the transition manager.
        
        Args:
            data_dir: Path to data directory containing transitions.txt
        """
        self.data_dir = Path(data_dir)
        self.transitions: List[Tuple[int, int]] = []  # List of (start_frame, end_frame) tuples
        self.modified = False
        
    def load_transitions(self) -> bool:
        """
        Load transitions from transitions.txt file.
        
        Returns:
            True if successful, False otherwise
        """
        transitions_path = self.data_dir / "transitions.txt"
        
        if not transitions_path.exists():
            print(f"No transitions file found at {transitions_path}")
            self.transitions = []
            return False
        
        try:
            self.transitions = []
            with open(transitions_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse start,end
                    parts = line.split(',')
                    if len(parts) == 2:
                        try:
                            start = int(parts[0].strip())
                            end = int(parts[1].strip())
                            if start <= end:
                                self.transitions.append((start, end))
                            else:
                                print(f"Warning: Invalid range {start},{end} (start > end)")
                        except ValueError:
                            print(f"Warning: Could not parse line: {line}")
            
            self.transitions.sort()  # Sort by start frame
            print(f"Loaded {len(self.transitions)} transition ranges")
            return True
            
        except Exception as e:
            print(f"Error loading transitions: {e}")
            return False
    
    def save_transitions(self) -> bool:
        """
        Save transitions to transitions.txt file.
        
        Returns:
            True if successful, False otherwise
        """
        transitions_path = self.data_dir / "transitions.txt"
        
        try:
            with open(transitions_path, 'w') as f:
                f.write("# Scene Transition Ranges\n")
                f.write("# Format: start_frame,end_frame\n")
                f.write("# You can manually edit this file to add missing transitions or remove false positives\n")
                f.write("# Lines starting with # are comments and will be ignored\n\n")
                
                for start, end in sorted(self.transitions):
                    f.write(f"{start},{end}\n")
            
            print(f"Saved {len(self.transitions)} transition ranges to {transitions_path}")
            self.modified = False
            return True
            
        except Exception as e:
            print(f"Error saving transitions: {e}")
            return False
    
    def is_in_transition(self, frame_idx: int) -> bool:
        """
        Check if a frame is within any transition range.
        
        Args:
            frame_idx: Frame index to check
            
        Returns:
            True if frame is in a transition range
        """
        for start, end in self.transitions:
            if start <= frame_idx <= end:
                return True
        return False
    
    def get_transition_at_frame(self, frame_idx: int) -> Optional[Tuple[int, int]]:
        """
        Get the transition range containing the given frame.
        
        Args:
            frame_idx: Frame index to check
            
        Returns:
            (start, end) tuple or None if not in any transition
        """
        for start, end in self.transitions:
            if start <= frame_idx <= end:
                return (start, end)
        return None
    
    def add_transition(self, start_frame: int, end_frame: int) -> bool:
        """
        Add a new transition range.
        
        Args:
            start_frame: Start frame (inclusive)
            end_frame: End frame (inclusive)
            
        Returns:
            True if added successfully
        """
        if start_frame > end_frame:
            print(f"Error: Start frame ({start_frame}) > end frame ({end_frame})")
            return False
        
        # Check for overlaps
        for existing_start, existing_end in self.transitions:
            if (start_frame <= existing_end and end_frame >= existing_start):
                print(f"Warning: Range {start_frame}-{end_frame} overlaps with {existing_start}-{existing_end}")
                # We'll allow it but warn the user
        
        self.transitions.append((start_frame, end_frame))
        self.transitions.sort()
        self.modified = True
        return True
    
    def remove_transition(self, start_frame: int, end_frame: int) -> bool:
        """
        Remove a transition range.
        
        Args:
            start_frame: Start frame of range to remove
            end_frame: End frame of range to remove
            
        Returns:
            True if removed successfully
        """
        try:
            self.transitions.remove((start_frame, end_frame))
            self.modified = True
            return True
        except ValueError:
            print(f"Transition range {start_frame}-{end_frame} not found")
            return False
    
    def remove_transition_at_frame(self, frame_idx: int) -> bool:
        """
        Remove the transition range containing the given frame.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            True if a transition was removed
        """
        transition = self.get_transition_at_frame(frame_idx)
        if transition:
            return self.remove_transition(*transition)
        return False
    
    def get_all_transitions(self) -> List[Tuple[int, int]]:
        """
        Get all transition ranges.
        
        Returns:
            List of (start, end) tuples
        """
        return list(self.transitions)
