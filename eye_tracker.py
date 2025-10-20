import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class GestureType(Enum):
    """Eye gesture types."""
    NONE = "none"
    LEFT_WINK = "left_wink"
    RIGHT_WINK = "right_wink"
    DOUBLE_BLINK = "double_blink"
    SUSTAINED_GAZE = "sustained_gaze"


@dataclass
class EyeData:
    """Container for eye tracking data."""
    left_ear: float
    right_ear: float
    left_gaze: Tuple[float, float]
    right_gaze: Tuple[float, float]
    combined_gaze: Tuple[float, float]
    is_blinking: bool
    gesture: GestureType
    landmarks: Optional[List[Tuple[int, int]]] = None


class EyeTracker:
    """MediaPipe-based eye tracking system."""
    
    # MediaPipe face mesh landmark indices for eyes
    LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    
    # Key eye landmark indices for EAR calculation
    LEFT_EAR_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EAR_INDICES = [362, 385, 387, 263, 373, 380]
    
    def __init__(self, ear_threshold: float = 0.25, wink_duration: Tuple[float, float] = (0.2, 0.5)):
        """
        Initialize the eye tracker.
        
        Args:
            ear_threshold: Threshold for blink detection (lower = more sensitive)
            wink_duration: Min and max duration for wink detection in seconds
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize MediaPipe Face Mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.ear_threshold = ear_threshold
        self.wink_duration = wink_duration
        
        # Gesture detection state
        self.left_eye_closed_time = 0
        self.right_eye_closed_time = 0
        self.last_blink_time = 0
        self.blink_count = 0
        self.gaze_history = []
        self.gaze_history_max = 30  # frames
        
        # Frame dimensions (set when first frame is processed)
        self.frame_width = 0
        self.frame_height = 0
        
    def set_frame_dimensions(self, width: int, height: int):
        """Set frame dimensions for coordinate normalization."""
        self.frame_width = width
        self.frame_height = height
    
    def calculate_ear(self, landmarks: List[Tuple[int, int]], eye_indices: List[int]) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection.
        
        Args:
            landmarks: List of landmark coordinates
            eye_indices: Indices for EAR calculation points
            
        Returns:
            EAR value
        """
        if len(landmarks) < max(eye_indices) + 1:
            return 0.0
            
        # Extract eye landmark points
        p1 = landmarks[eye_indices[0]]  # Top
        p2 = landmarks[eye_indices[1]]  # Bottom
        p3 = landmarks[eye_indices[2]]  # Left
        p4 = landmarks[eye_indices[3]]  # Right
        p5 = landmarks[eye_indices[4]]  # Inner corner
        p6 = landmarks[eye_indices[5]]  # Outer corner
        
        # Calculate distances
        vertical_1 = np.linalg.norm(np.array(p2) - np.array(p6))
        vertical_2 = np.linalg.norm(np.array(p3) - np.array(p5))
        horizontal = np.linalg.norm(np.array(p1) - np.array(p4))
        
        # Calculate EAR
        if horizontal == 0:
            return 0.0
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def calculate_gaze(self, landmarks: List[Tuple[int, int]], eye_indices: List[int]) -> Tuple[float, float]:
        """
        Calculate gaze direction for an eye.
        
        Args:
            landmarks: List of landmark coordinates
            eye_indices: Eye landmark indices
            
        Returns:
            Normalized gaze coordinates (x, y) in range [0, 1]
        """
        if len(landmarks) < max(eye_indices) + 1:
            return (0.5, 0.5)
        
        # Get eye center and iris position
        eye_center = landmarks[eye_indices[0]]  # Use first landmark as reference
        iris_center = landmarks[eye_indices[1]]  # Use second landmark as iris
        
        # Calculate relative position
        if self.frame_width > 0 and self.frame_height > 0:
            gaze_x = iris_center[0] / self.frame_width
            gaze_y = iris_center[1] / self.frame_height
        else:
            gaze_x = 0.5
            gaze_y = 0.5
            
        return (gaze_x, gaze_y)
    
    def detect_gesture(self, left_ear: float, right_ear: float, current_time: float) -> GestureType:
        """
        Detect eye gestures based on EAR values and timing.
        
        Args:
            left_ear: Left eye EAR value
            right_ear: Right eye EAR value
            current_time: Current timestamp
            
        Returns:
            Detected gesture type
        """
        left_closed = left_ear < self.ear_threshold
        right_closed = right_ear < self.ear_threshold
        
        # Update eye closed times
        if left_closed:
            if self.left_eye_closed_time == 0:
                self.left_eye_closed_time = current_time
        else:
            self.left_eye_closed_time = 0
            
        if right_closed:
            if self.right_eye_closed_time == 0:
                self.right_eye_closed_time = current_time
        else:
            self.right_eye_closed_time = 0
        
        # Detect gestures
        if left_closed and not right_closed:
            if self.left_eye_closed_time > 0:
                duration = current_time - self.left_eye_closed_time
                if self.wink_duration[0] <= duration <= self.wink_duration[1]:
                    return GestureType.LEFT_WINK
                    
        elif right_closed and not left_closed:
            if self.right_eye_closed_time > 0:
                duration = current_time - self.right_eye_closed_time
                if self.wink_duration[0] <= duration <= self.wink_duration[1]:
                    return GestureType.RIGHT_WINK
                    
        elif left_closed and right_closed:
            # Both eyes closed - check for double blink
            if current_time - self.last_blink_time < 0.5:  # Within 0.5 seconds
                self.blink_count += 1
                if self.blink_count >= 2:
                    self.blink_count = 0
                    return GestureType.DOUBLE_BLINK
            else:
                self.blink_count = 1
            self.last_blink_time = current_time
        
        return GestureType.NONE
    
    def process_frame(self, frame: np.ndarray, timestamp: float = 0.0) -> Optional[EyeData]:
        """
        Process a single frame for eye tracking.
        
        Args:
            frame: Input frame
            timestamp: Frame timestamp
            
        Returns:
            EyeData object with tracking results, or None if no face detected
        """
        if self.frame_width == 0 or self.frame_height == 0:
            self.set_frame_dimensions(frame.shape[1], frame.shape[0])
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get the first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert landmarks to pixel coordinates
        landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * self.frame_width)
            y = int(landmark.y * self.frame_height)
            landmarks.append((x, y))
        
        # Calculate EAR for both eyes
        left_ear = self.calculate_ear(landmarks, self.LEFT_EAR_INDICES)
        right_ear = self.calculate_ear(landmarks, self.RIGHT_EAR_INDICES)
        
        # Calculate gaze directions
        left_gaze = self.calculate_gaze(landmarks, self.LEFT_EYE_INDICES)
        right_gaze = self.calculate_gaze(landmarks, self.RIGHT_EYE_INDICES)
        
        # Calculate combined gaze (average of both eyes)
        combined_gaze = (
            (left_gaze[0] + right_gaze[0]) / 2,
            (left_gaze[1] + right_gaze[1]) / 2
        )
        
        # Update gaze history
        self.gaze_history.append(combined_gaze)
        if len(self.gaze_history) > self.gaze_history_max:
            self.gaze_history.pop(0)
        
        # Detect blinking
        is_blinking = left_ear < self.ear_threshold or right_ear < self.ear_threshold
        
        # Detect gestures
        gesture = self.detect_gesture(left_ear, right_ear, timestamp)
        
        return EyeData(
            left_ear=left_ear,
            right_ear=right_ear,
            left_gaze=left_gaze,
            right_gaze=right_gaze,
            combined_gaze=combined_gaze,
            is_blinking=is_blinking,
            gesture=gesture,
            landmarks=landmarks
        )
    
    def get_gaze_history(self) -> List[Tuple[float, float]]:
        """Get recent gaze history."""
        return self.gaze_history.copy()
    
    def reset_metrics(self):
        """Reset tracking metrics."""
        self.left_eye_closed_time = 0
        self.right_eye_closed_time = 0
        self.last_blink_time = 0
        self.blink_count = 0
        self.gaze_history.clear()
    
    def release(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
