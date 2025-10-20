import cv2
import os
from typing import Optional, Tuple


class InputHandler:
    """Handles video input from webcam or video files."""
    
    def __init__(self, source: str = "webcam", video_path: Optional[str] = None):
        """
        Initialize input handler.
        
        Args:
            source: "webcam" or "video"
            video_path: Path to video file (required if source is "video")
        """
        self.source = source
        self.video_path = video_path
        self.cap = None
        self.frame_width = 0
        self.frame_height = 0
        
    def initialize(self) -> bool:
        """
        Initialize the video source.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.source == "webcam":
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    print("Error: Could not open webcam")
                    return False
            elif self.source == "video":
                if not self.video_path or not os.path.exists(self.video_path):
                    print(f"Error: Video file not found: {self.video_path}")
                    return False
                self.cap = cv2.VideoCapture(self.video_path)
                if not self.cap.isOpened():
                    print(f"Error: Could not open video file: {self.video_path}")
                    return False
            else:
                print(f"Error: Invalid source '{self.source}'. Use 'webcam' or 'video'")
                return False
                
            # Get frame dimensions
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Initialized {self.source} source: {self.frame_width}x{self.frame_height}")
            return True
            
        except Exception as e:
            print(f"Error initializing input handler: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """
        Read the next frame from the video source.
        
        Returns:
            Tuple of (success, frame) where frame is None if unsuccessful
        """
        if self.cap is None:
            return False, None
            
        ret, frame = self.cap.read()
        if not ret:
            if self.source == "video":
                print("End of video file reached")
            return False, None
            
        return True, frame
    
    def get_frame_dimensions(self) -> Tuple[int, int]:
        """Get frame width and height."""
        return self.frame_width, self.frame_height
    
    def get_fps(self) -> float:
        """Get the FPS of the video source."""
        if self.cap is None:
            return 0.0
        return self.cap.get(cv2.CAP_PROP_FPS)
    
    def is_video_file(self) -> bool:
        """Check if the source is a video file."""
        return self.source == "video"
    
    def release(self):
        """Release the video capture resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
