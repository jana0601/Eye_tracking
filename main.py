#!/usr/bin/env python3
"""
MediaPipe Eye Tracking System
Main application that integrates all modules for real-time eye tracking.
"""

import cv2
import time
import argparse
import sys
from typing import Optional

from input_handler import InputHandler
from eye_tracker import EyeTracker, GestureType
from visualizer import Visualizer
from data_logger import DataLogger


class EyeTrackingApp:
    """Main application class for eye tracking system."""
    
    def __init__(self, source: str = "webcam", video_path: Optional[str] = None):
        """
        Initialize the eye tracking application.
        
        Args:
            source: Input source ("webcam" or "video")
            video_path: Path to video file (required if source is "video")
        """
        self.source = source
        self.video_path = video_path
        
        # Initialize components
        self.input_handler = InputHandler(source, video_path)
        self.eye_tracker = EyeTracker()
        self.visualizer = None  # Will be initialized after getting frame dimensions
        self.data_logger = DataLogger()
        
        # Application state
        self.running = False
        self.frame_count = 0
        self.total_blinks = 0
        self.gesture_counts = {gesture.value: 0 for gesture in GestureType}
        self.fps_history = []
        self.show_debug = False
        
        # Performance tracking
        self.start_time = 0
        self.last_fps_time = 0
        self.fps_counter = 0
        
    def initialize(self) -> bool:
        """
        Initialize all components.
        
        Returns:
            True if successful, False otherwise
        """
        print("Initializing Eye Tracking System...")
        
        # Initialize input handler
        if not self.input_handler.initialize():
            print("Failed to initialize input handler")
            return False
        
        # Get frame dimensions
        width, height = self.input_handler.get_frame_dimensions()
        self.visualizer = Visualizer(width, height)
        
        # Set frame dimensions in eye tracker
        self.eye_tracker.set_frame_dimensions(width, height)
        
        print("Eye Tracking System initialized successfully!")
        print(f"Source: {self.source}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {self.input_handler.get_fps():.1f}")
        
        return True
    
    def process_frame(self, frame) -> Optional[dict]:
        """
        Process a single frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary with processing results
        """
        self.frame_count += 1
        current_time = time.time()
        
        # Calculate FPS
        if self.last_fps_time == 0:
            self.last_fps_time = current_time
        else:
            fps = 1.0 / (current_time - self.last_fps_time)
            self.fps_history.append(fps)
            if len(self.fps_history) > 30:  # Keep last 30 FPS measurements
                self.fps_history.pop(0)
            self.last_fps_time = current_time
        
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        
        # Process frame with eye tracker
        eye_data = self.eye_tracker.process_frame(frame, current_time)
        
        if eye_data is not None:
            # Update blink count
            if eye_data.is_blinking:
                self.total_blinks += 1
            
            # Update gesture counts
            if eye_data.gesture != GestureType.NONE:
                self.gesture_counts[eye_data.gesture.value] += 1
            
            # Log eye data
            self.data_logger.log_eye_data(eye_data, self.frame_count, avg_fps)
            
            return {
                'eye_data': eye_data,
                'fps': avg_fps,
                'frame_count': self.frame_count
            }
        else:
            # No face detected
            self.data_logger.log_no_face_detected(self.frame_count, avg_fps)
            return {
                'eye_data': None,
                'fps': avg_fps,
                'frame_count': self.frame_count
            }
    
    def handle_keyboard_input(self, key: int) -> bool:
        """
        Handle keyboard input.
        
        Args:
            key: Key code from cv2.waitKey()
            
        Returns:
            True to continue, False to quit
        """
        if key == ord('q') or key == ord('Q'):
            print("Quit requested")
            return False
        elif key == ord('s') or key == ord('S'):
            # Save screenshot
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, self.current_frame)
            print(f"Screenshot saved: {filename}")
        elif key == ord('r') or key == ord('R'):
            # Reset metrics
            self.eye_tracker.reset_metrics()
            self.total_blinks = 0
            self.gesture_counts = {gesture.value: 0 for gesture in GestureType}
            self.fps_history.clear()
            print("Metrics reset")
        elif key == ord('d') or key == ord('D'):
            # Toggle debug view
            self.show_debug = not self.show_debug
            print(f"Debug view: {'ON' if self.show_debug else 'OFF'}")
        
        return True
    
    def run(self):
        """Main application loop."""
        if not self.initialize():
            return
        
        print("\nStarting eye tracking...")
        print("Controls:")
        print("  Q - Quit")
        print("  S - Save screenshot")
        print("  R - Reset metrics")
        print("  D - Toggle debug view")
        print("\nPress Q to quit...")
        
        self.running = True
        self.start_time = time.time()
        
        try:
            while self.running:
                # Read frame
                success, frame = self.input_handler.read_frame()
                if not success:
                    if self.input_handler.is_video_file():
                        print("End of video file reached")
                    break
                
                self.current_frame = frame.copy()
                
                # Process frame
                result = self.process_frame(frame)
                
                if result:
                    # Visualize frame
                    gaze_history = self.eye_tracker.get_gaze_history()
                    visualized_frame = self.visualizer.visualize_frame(
                        frame, 
                        result['eye_data'], 
                        gaze_history, 
                        result['fps'],
                        self.show_debug
                    )
                    
                    # Display frame
                    cv2.imshow('Eye Tracking System', visualized_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if not self.handle_keyboard_input(key):
                        break
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error during execution: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources and save session data."""
        print("\nCleaning up...")
        
        # Calculate session statistics
        session_duration = time.time() - self.start_time
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        
        # Log session summary
        self.data_logger.log_session_summary(
            self.frame_count,
            self.total_blinks,
            self.gesture_counts,
            avg_fps
        )
        
        # Print session summary
        print(f"\nSession Summary:")
        print(f"  Duration: {session_duration:.1f} seconds")
        print(f"  Total Frames: {self.frame_count}")
        print(f"  Average FPS: {avg_fps:.1f}")
        print(f"  Total Blinks: {self.total_blinks}")
        print(f"  Gesture Counts:")
        for gesture, count in self.gesture_counts.items():
            if count > 0:
                print(f"    {gesture}: {count}")
        
        # Cleanup resources
        self.input_handler.release()
        self.eye_tracker.release()
        self.data_logger.close()
        cv2.destroyAllWindows()
        
        print("Cleanup completed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='MediaPipe Eye Tracking System')
    parser.add_argument('--source', choices=['webcam', 'video'], default='webcam',
                       help='Input source: webcam or video file')
    parser.add_argument('--path', type=str, help='Path to video file (required if source is video)')
    parser.add_argument('--ear-threshold', type=float, default=0.25,
                       help='EAR threshold for blink detection (default: 0.25)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.source == 'video' and not args.path:
        print("Error: --path is required when source is 'video'")
        sys.exit(1)
    
    # Create and run application
    app = EyeTrackingApp(args.source, args.path)
    app.eye_tracker.ear_threshold = args.ear_threshold
    
    try:
        app.run()
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
