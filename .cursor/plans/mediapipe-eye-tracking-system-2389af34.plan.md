<!-- 2389af34-75a4-4aec-b0b1-c412e8baa59a 4ea26608-adf6-4329-a766-a18669473a70 -->
# MediaPipe Eye Tracking System

## Architecture Overview

Create a modular Python application using MediaPipe's Face Mesh solution to detect 468 facial landmarks, focusing on eye regions (landmarks 33, 133, 159, 145, 362, 263, 386, 374 for eye corners and key points). The system will process both live webcam feeds and video files.

## Core Components

### 1. Eye Tracker Module (`eye_tracker.py`)

- Initialize MediaPipe Face Mesh with refined eye landmarks
- Detect and extract eye landmarks (left and right eye separately)
- Calculate Eye Aspect Ratio (EAR) for blink detection using vertical/horizontal eye distances
- Compute gaze direction using iris positions and eye centers
- Implement eye gesture recognition (winks, rapid blinks, sustained gaze)

### 2. Data Logger Module (`data_logger.py`)

- Log timestamped eye tracking data to CSV files
- Record: timestamp, gaze coordinates (x, y), blink events, eye aspect ratios, gesture events
- Support session-based logging with unique filenames

### 3. Visualization Module (`visualizer.py`)

- Draw eye landmarks on video frames
- Overlay gaze direction vectors/crosshairs on screen
- Display real-time metrics (blink count, gaze position, detected gestures)
- Create debug view showing isolated eye regions

### 4. Input Handler (`input_handler.py`)

- Abstract video source (webcam or file)
- Handle frame capture and preprocessing
- Manage camera initialization and video file validation

### 5. Main Application (`main.py`)

- Command-line interface for mode selection (webcam/video file)
- Integrate all modules
- Control flow for processing and display
- Keyboard controls (q: quit, s: save screenshot, r: reset metrics)

## Key Algorithms

**Blink Detection**: EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||) where p1-p6 are eye landmark points. Threshold: EAR < 0.25 indicates closed eye.

**Gaze Estimation**: Calculate iris center relative to eye boundaries, map to screen coordinates using perspective transformation.

**Gesture Recognition**: Pattern matching on temporal sequences (e.g., left wink = left eye closed for 0.2-0.5s while right eye open).

## Dependencies

- opencv-python (cv2): Video capture and display
- mediapipe: Face mesh and eye landmark detection
- numpy: Numerical computations
- pandas: Data logging and CSV operations

## File Structure

```
eye_tracking/
├── main.py
├── eye_tracker.py
├── visualizer.py
├── data_logger.py
├── input_handler.py
├── requirements.txt
├── README.md
└── data/ (created at runtime for logs)
```

## Usage Flow

1. User runs `python main.py --source webcam` or `python main.py --source video --path input.mp4`
2. System initializes MediaPipe and selected input source
3. Process frames in loop: detect face → extract eye landmarks → compute metrics → visualize → log data
4. Save session data on exit

### To-dos

- [ ] Create requirements.txt with opencv-python, mediapipe, numpy, and pandas
- [ ] Implement input_handler.py for webcam and video file abstraction
- [ ] Build eye_tracker.py with landmark detection, EAR calculation, and gaze estimation
- [ ] Add blink detection and eye gesture recognition to eye_tracker.py
- [ ] Create visualizer.py with landmark overlays, gaze indicators, and metrics display
- [ ] Implement data_logger.py for CSV logging with timestamps and all metrics
- [ ] Build main.py with CLI, integration of all modules, and control flow
- [ ] Create README.md with setup instructions, usage examples, and feature documentation