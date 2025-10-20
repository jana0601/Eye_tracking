import cv2
import numpy as np
from typing import List, Tuple, Optional
from eye_tracker import EyeData, GestureType


class Visualizer:
    """Handles visualization of eye tracking data."""
    
    def __init__(self, frame_width: int, frame_height: int):
        """
        Initialize the visualizer.
        
        Args:
            frame_width: Frame width
            frame_height: Frame height
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Colors (BGR format)
        self.colors = {
            'landmark': (0, 255, 0),      # Green
            'gaze_line': (255, 0, 0),     # Blue
            'gaze_point': (0, 0, 255),    # Red
            'blink': (0, 255, 255),       # Yellow
            'gesture': (255, 0, 255),     # Magenta
            'text': (255, 255, 255),      # White
            'background': (0, 0, 0)       # Black
        }
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        
    def draw_landmarks(self, frame: np.ndarray, eye_data: EyeData) -> np.ndarray:
        """
        Draw eye landmarks on the frame.
        
        Args:
            frame: Input frame
            eye_data: Eye tracking data
            
        Returns:
            Frame with landmarks drawn
        """
        if eye_data.landmarks is None:
            return frame
        
        # Draw left eye landmarks with enhanced visibility
        for i in range(33, 161):  # Left eye region
            if i < len(eye_data.landmarks):
                cv2.circle(frame, eye_data.landmarks[i], 2, self.colors['landmark'], -1)
                cv2.circle(frame, eye_data.landmarks[i], 3, (255, 255, 255), 1)
        
        # Draw right eye landmarks with enhanced visibility
        for i in range(362, 398):  # Right eye region
            if i < len(eye_data.landmarks):
                cv2.circle(frame, eye_data.landmarks[i], 2, self.colors['landmark'], -1)
                cv2.circle(frame, eye_data.landmarks[i], 3, (255, 255, 255), 1)
        
        return frame
    
    def draw_gaze(self, frame: np.ndarray, eye_data: EyeData) -> np.ndarray:
        """
        Draw gaze direction indicators.
        
        Args:
            frame: Input frame
            eye_data: Eye tracking data
            
        Returns:
            Frame with gaze indicators
        """
        # Draw gaze crosshair with enhanced styling
        gaze_x = int(eye_data.combined_gaze[0] * self.frame_width)
        gaze_y = int(eye_data.combined_gaze[1] * self.frame_height)
        
        # Draw outer circle for better visibility
        cv2.circle(frame, (gaze_x, gaze_y), 25, self.colors['gaze_line'], 2)
        
        # Draw crosshair with thicker lines
        cv2.line(frame, (gaze_x - 30, gaze_y), (gaze_x + 30, gaze_y), self.colors['gaze_line'], 3)
        cv2.line(frame, (gaze_x, gaze_y - 30), (gaze_x, gaze_y + 30), self.colors['gaze_line'], 3)
        
        # Draw center point with gradient effect
        cv2.circle(frame, (gaze_x, gaze_y), 8, self.colors['gaze_point'], -1)
        cv2.circle(frame, (gaze_x, gaze_y), 5, (255, 255, 255), -1)
        
        # Add gaze coordinates text
        coord_text = f"({eye_data.combined_gaze[0]:.2f}, {eye_data.combined_gaze[1]:.2f})"
        cv2.putText(frame, coord_text, (gaze_x + 35, gaze_y - 10), self.font, 0.5, 
                   self.colors['text'], 1)
        
        return frame
    
    def draw_gaze_history(self, frame: np.ndarray, gaze_history: List[Tuple[float, float]]) -> np.ndarray:
        """
        Draw gaze history trail.
        
        Args:
            frame: Input frame
            gaze_history: List of recent gaze positions
            
        Returns:
            Frame with gaze history trail
        """
        if len(gaze_history) < 2:
            return frame
        
        # Draw trail with fading opacity
        for i in range(1, len(gaze_history)):
            alpha = i / len(gaze_history)
            thickness = max(1, int(3 * alpha))
            
            pt1 = (
                int(gaze_history[i-1][0] * self.frame_width),
                int(gaze_history[i-1][1] * self.frame_height)
            )
            pt2 = (
                int(gaze_history[i][0] * self.frame_width),
                int(gaze_history[i][1] * self.frame_height)
            )
            
            cv2.line(frame, pt1, pt2, self.colors['gaze_line'], thickness)
        
        return frame
    
    def draw_metrics(self, frame: np.ndarray, eye_data: EyeData, fps: float = 0.0, mask_only: bool = False) -> np.ndarray:
        """
        Draw tracking metrics on the frame.
        
        Args:
            frame: Input frame
            eye_data: Eye tracking data
            fps: Current FPS
            mask_only: Whether in mask-only mode (affects overlay blending)
            
        Returns:
            Frame with metrics overlay
        """
        # Create metrics overlay with better styling
        overlay = frame.copy()
        
        # Semi-transparent background with rounded corners effect - compact size
        cv2.rectangle(overlay, (10, 10), (290, 200), self.colors['background'], -1)
        if mask_only:
            # In mask-only mode, don't blend - just draw solid background
            cv2.rectangle(frame, (10, 10), (290, 200), self.colors['background'], -1)
        else:
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Enhanced font settings - compact
        font_scale_large = 0.7
        font_scale_medium = 0.5
        font_scale_small = 0.4
        thickness_large = 2
        thickness_medium = 1
        thickness_small = 1
        
        # Title
        cv2.putText(frame, "Eye Tracking Metrics", (20, 35), self.font, font_scale_large, 
                   self.colors['text'], thickness_large)
        
        # Metrics with better formatting - compact spacing
        y_offset = 55
        line_height = 25
        
        metrics = [
            f"FPS: {fps:.1f}",
            f"Left EAR: {eye_data.left_ear:.3f}",
            f"Right EAR: {eye_data.right_ear:.3f}",
            f"Gaze: ({eye_data.combined_gaze[0]:.3f}, {eye_data.combined_gaze[1]:.3f})",
            f"Blinking: {'Yes' if eye_data.is_blinking else 'No'}",
            f"Gesture: {eye_data.gesture.value.replace('_', ' ').title()}"
        ]
        
        for i, metric in enumerate(metrics):
            # Use different font sizes for different types of info
            if i == 0:  # FPS
                font_scale = font_scale_medium
                thickness = thickness_medium
                color = (0, 255, 255)  # Cyan for FPS
            elif i < 3:  # EAR values
                font_scale = font_scale_small
                thickness = thickness_small
                color = (255, 255, 0)  # Yellow for EAR
            else:  # Other metrics
                font_scale = font_scale_small
                thickness = thickness_small
                color = self.colors['text']
            
            cv2.putText(frame, metric, (20, y_offset), self.font, font_scale, 
                       color, thickness)
            y_offset += line_height
        
        return frame
    
    def draw_gesture_indicator(self, frame: np.ndarray, gesture: GestureType) -> np.ndarray:
        """
        Draw gesture-specific visual indicators.
        
        Args:
            frame: Input frame
            gesture: Detected gesture
            
        Returns:
            Frame with gesture indicators
        """
        if gesture == GestureType.NONE:
            return frame
        
        # Draw gesture indicator in top-right corner
        indicator_text = f"GESTURE: {gesture.value.upper()}"
        text_size = cv2.getTextSize(indicator_text, self.font, self.font_scale * 1.0, self.font_thickness)[0]
        
        # Background for gesture indicator
        x = self.frame_width - text_size[0] - 20
        y = 50
        cv2.rectangle(frame, (x - 10, y - text_size[1] - 10), 
                     (x + text_size[0] + 10, y + 10), self.colors['gesture'], -1)
        
        # Gesture text
        cv2.putText(frame, indicator_text, (x, y), self.font, self.font_scale * 1.0, 
                   self.colors['text'], self.font_thickness)
        
        return frame
    
    def draw_blink_indicator(self, frame: np.ndarray, is_blinking: bool) -> np.ndarray:
        """
        Draw blink indicator.
        
        Args:
            frame: Input frame
            is_blinking: Whether eyes are currently blinking
            
        Returns:
            Frame with blink indicator
        """
        if is_blinking:
            # Draw blinking indicator - bigger circle
            cv2.circle(frame, (self.frame_width - 50, 100), 35, self.colors['blink'], -1)
            cv2.putText(frame, "BLINK", (self.frame_width - 80, 110), self.font, 
                       self.font_scale, self.colors['text'], self.font_thickness)
        
        return frame
    
    def create_debug_view(self, frame: np.ndarray, eye_data: EyeData) -> np.ndarray:
        """
        Create a debug view showing isolated eye regions.
        
        Args:
            frame: Input frame
            eye_data: Eye tracking data
            
        Returns:
            Debug view frame
        """
        if eye_data.landmarks is None:
            return frame
        
        # Create debug frame
        debug_frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # Draw face mesh landmarks
        for i, landmark in enumerate(eye_data.landmarks):
            if i in range(33, 161) or i in range(362, 398):  # Eye regions
                cv2.circle(debug_frame, landmark, 2, self.colors['landmark'], -1)
        
        # Draw gaze indicators
        debug_frame = self.draw_gaze(debug_frame, eye_data)
        
        return debug_frame
    
    def draw_instructions(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw control instructions on the frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with instructions
        """
        instructions = [
            "Controls:",
            "Q - Quit",
            "S - Save screenshot",
            "R - Reset metrics",
            "D - Toggle debug view"
        ]
        
        y_start = self.frame_height - len(instructions) * 25 - 10
        
        for i, instruction in enumerate(instructions):
            y = y_start + i * 25
            cv2.putText(frame, instruction, (10, y), self.font, self.font_scale, 
                       self.colors['text'], self.font_thickness)
        
        return frame
    
    def visualize_frame(self, frame: np.ndarray, eye_data: Optional[EyeData], 
                       gaze_history: List[Tuple[float, float]], fps: float = 0.0,
                       show_debug: bool = False, mask_only: bool = False) -> np.ndarray:
        """
        Main visualization function that combines all visual elements.
        
        Args:
            frame: Input frame
            eye_data: Eye tracking data
            gaze_history: Recent gaze positions
            fps: Current FPS
            show_debug: Whether to show debug view
            mask_only: Whether to show only landmarks/overlays on black background
            
        Returns:
            Fully visualized frame
        """
        if eye_data is None:
            # No face detected - show basic frame with instructions
            frame = self.draw_instructions(frame)
            cv2.putText(frame, "No face detected", (self.frame_width // 2 - 100, 
                       self.frame_height // 2), self.font, self.font_scale * 1.5, 
                       self.colors['text'], self.font_thickness)
            return frame
        
        if show_debug:
            frame = self.create_debug_view(frame, eye_data)
        else:
            # Draw all visual elements (works for both normal and mask-only modes)
            frame = self.draw_landmarks(frame, eye_data)
            frame = self.draw_gaze(frame, eye_data)
            frame = self.draw_gaze_history(frame, gaze_history)
            frame = self.draw_blink_indicator(frame, eye_data.is_blinking)
            frame = self.draw_gesture_indicator(frame, eye_data.gesture)
        
        # Always draw metrics and instructions
        frame = self.draw_metrics(frame, eye_data, fps, mask_only)
        frame = self.draw_instructions(frame)
        
        return frame
