import pandas as pd
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from eye_tracker import EyeData, GestureType


class DataLogger:
    """Handles logging of eye tracking data to CSV files."""
    
    def __init__(self, output_dir: str = "data"):
        """
        Initialize the data logger.
        
        Args:
            output_dir: Directory to save log files
        """
        self.output_dir = output_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = None
        self.data_buffer = []
        self.buffer_size = 100  # Flush to file every 100 records
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize logging
        self._initialize_logging()
    
    def _initialize_logging(self):
        """Initialize the CSV log file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.output_dir, f"eye_tracking_session_{timestamp}.csv")
        
        # Create DataFrame with column headers
        columns = [
            'timestamp',
            'session_id',
            'frame_number',
            'left_ear',
            'right_ear',
            'left_gaze_x',
            'left_gaze_y',
            'right_gaze_x',
            'right_gaze_y',
            'combined_gaze_x',
            'combined_gaze_y',
            'is_blinking',
            'gesture',
            'gesture_duration',
            'fps'
        ]
        
        # Create empty CSV file with headers
        df = pd.DataFrame(columns=columns)
        df.to_csv(self.log_file, index=False)
        
        print(f"Data logging initialized: {self.log_file}")
    
    def log_eye_data(self, eye_data: EyeData, frame_number: int, 
                    fps: float = 0.0, gesture_duration: float = 0.0) -> None:
        """
        Log eye tracking data for a single frame.
        
        Args:
            eye_data: Eye tracking data
            frame_number: Current frame number
            fps: Current FPS
            gesture_duration: Duration of current gesture (if any)
        """
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'session_id': self.session_id,
            'frame_number': frame_number,
            'left_ear': eye_data.left_ear,
            'right_ear': eye_data.right_ear,
            'left_gaze_x': eye_data.left_gaze[0],
            'left_gaze_y': eye_data.left_gaze[1],
            'right_gaze_x': eye_data.right_gaze[0],
            'right_gaze_y': eye_data.right_gaze[1],
            'combined_gaze_x': eye_data.combined_gaze[0],
            'combined_gaze_y': eye_data.combined_gaze[1],
            'is_blinking': eye_data.is_blinking,
            'gesture': eye_data.gesture.value,
            'gesture_duration': gesture_duration,
            'fps': fps
        }
        
        self.data_buffer.append(log_entry)
        
        # Flush buffer if it's full
        if len(self.data_buffer) >= self.buffer_size:
            self._flush_buffer()
    
    def log_no_face_detected(self, frame_number: int, fps: float = 0.0) -> None:
        """
        Log when no face is detected.
        
        Args:
            frame_number: Current frame number
            fps: Current FPS
        """
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'session_id': self.session_id,
            'frame_number': frame_number,
            'left_ear': None,
            'right_ear': None,
            'left_gaze_x': None,
            'left_gaze_y': None,
            'right_gaze_x': None,
            'right_gaze_y': None,
            'combined_gaze_x': None,
            'combined_gaze_y': None,
            'is_blinking': False,
            'gesture': GestureType.NONE.value,
            'gesture_duration': 0.0,
            'fps': fps
        }
        
        self.data_buffer.append(log_entry)
        
        # Flush buffer if it's full
        if len(self.data_buffer) >= self.buffer_size:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """Flush the data buffer to the CSV file."""
        if not self.data_buffer:
            return
        
        # Convert buffer to DataFrame and append to CSV
        df = pd.DataFrame(self.data_buffer)
        df.to_csv(self.log_file, mode='a', header=False, index=False)
        
        # Clear buffer
        self.data_buffer.clear()
    
    def log_session_summary(self, total_frames: int, total_blinks: int, 
                           gesture_counts: Dict[str, int], avg_fps: float) -> None:
        """
        Log a summary of the tracking session.
        
        Args:
            total_frames: Total number of frames processed
            total_blinks: Total number of blinks detected
            gesture_counts: Dictionary of gesture counts
            avg_fps: Average FPS during session
        """
        summary_file = os.path.join(self.output_dir, f"session_summary_{self.session_id}.txt")
        
        with open(summary_file, 'w') as f:
            f.write(f"Eye Tracking Session Summary\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Frames: {total_frames}\n")
            f.write(f"Total Blinks: {total_blinks}\n")
            f.write(f"Average FPS: {avg_fps:.2f}\n")
            f.write(f"\nGesture Counts:\n")
            
            for gesture, count in gesture_counts.items():
                f.write(f"  {gesture}: {count}\n")
            
            f.write(f"\nData File: {self.log_file}\n")
        
        print(f"Session summary saved: {summary_file}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get basic statistics for the current session.
        
        Returns:
            Dictionary with session statistics
        """
        if not os.path.exists(self.log_file):
            return {}
        
        try:
            df = pd.read_csv(self.log_file)
            
            stats = {
                'total_frames': len(df),
                'frames_with_face': len(df[df['left_ear'].notna()]),
                'total_blinks': df['is_blinking'].sum() if 'is_blinking' in df.columns else 0,
                'avg_fps': df['fps'].mean() if 'fps' in df.columns else 0,
                'gesture_counts': df['gesture'].value_counts().to_dict() if 'gesture' in df.columns else {}
            }
            
            return stats
            
        except Exception as e:
            print(f"Error reading session stats: {e}")
            return {}
    
    def export_to_excel(self, output_file: Optional[str] = None) -> str:
        """
        Export the session data to an Excel file.
        
        Args:
            output_file: Optional custom output filename
            
        Returns:
            Path to the exported Excel file
        """
        if not os.path.exists(self.log_file):
            raise FileNotFoundError("No log file found to export")
        
        if output_file is None:
            output_file = os.path.join(self.output_dir, f"eye_tracking_data_{self.session_id}.xlsx")
        
        try:
            df = pd.read_csv(self.log_file)
            df.to_excel(output_file, index=False)
            print(f"Data exported to Excel: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"Error exporting to Excel: {e}")
            raise
    
    def close(self):
        """Close the logger and flush any remaining data."""
        self._flush_buffer()
        print(f"Data logging completed. Session data saved to: {self.log_file}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
