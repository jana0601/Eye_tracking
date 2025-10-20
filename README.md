# MediaPipe Eye Tracking System

A comprehensive Python-based eye tracking system using MediaPipe's Face Mesh solution. This system provides real-time gaze detection, blink detection, eye gesture recognition, visual overlays, and data logging capabilities.

## Features

- **Real-time Eye Tracking**: Track gaze direction using MediaPipe's 468 facial landmarks
- **Blink Detection**: Detect blinks using Eye Aspect Ratio (EAR) calculation
- **Eye Gesture Recognition**: Identify winks, double blinks, and sustained gaze patterns
- **Visual Overlays**: Real-time visualization with landmarks, gaze indicators, and metrics
- **Data Logging**: Comprehensive CSV logging with timestamps and all tracking metrics
- **Multiple Input Sources**: Support for both webcam and video file inputs
- **Debug Mode**: Toggle debug view for detailed landmark visualization

## Installation

### Prerequisites

- Python 3.7 or higher
- Webcam (for real-time tracking)
- OpenCV-compatible camera drivers

### Setup

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

- `opencv-python==4.8.1.78`: Video capture and display
- `mediapipe==0.10.7`: Face mesh and eye landmark detection
- `numpy==1.24.3`: Numerical computations
- `pandas==2.0.3`: Data logging and CSV operations

## Usage

### Basic Usage

#### Webcam Mode (Default)
```bash
python main.py
```

#### Video File Mode
```bash
python main.py --source video --path path/to/your/video.mp4
```

### Command Line Options

- `--source`: Input source (`webcam` or `video`)
- `--path`: Path to video file (required when source is `video`)
- `--ear-threshold`: EAR threshold for blink detection (default: 0.25, lower = more sensitive)

### Examples

```bash
# Use webcam with default settings
python main.py

# Use webcam with custom blink sensitivity
python main.py --ear-threshold 0.2

# Process a video file
python main.py --source video --path sample_video.mp4

# Process video with custom settings
python main.py --source video --path sample_video.mp4 --ear-threshold 0.3
```

## GUI Interface

The system includes a modern desktop GUI application built with Tkinter:

### Starting the GUI
```bash
python gui_tk.py
```

### GUI Features

- **Real-time Controls**: Start/stop tracking, adjust EAR threshold
- **Input Source Selection**: Switch between webcam and video file
- **Visual Display Options**: 
  - Show/hide video feed
  - Mask-only mode (shows only landmarks and overlays on black background)
- **Status Monitoring**: Real-time status updates and mode indicators
- **Modern Interface**: High-DPI scaling, improved fonts, and responsive design

### GUI Controls

- **Start**: Begin eye tracking
- **Close Camera**: Stop tracking and release camera
- **Exit App**: Close the entire application
- **Show Video**: Toggle video feed visibility
- **Mask Only**: Show only detection results on black background
- **EAR Threshold**: Adjustable sensitivity slider for blink detection

### Web Application

The system also includes a Flask-based web interface:

```bash
python web_app.py
```

Then open your browser to `http://localhost:5000`

#### Web App Features
- **Browser-based interface**: Access from any device on the network
- **Real-time video stream**: MJPEG streaming with overlays
- **Remote control**: Start/stop tracking from web interface
- **Easy disconnection**: ESC key or disconnect button to stop camera

### Interface Screenshots

The system provides multiple interface options:

- **Desktop GUI**: Modern Tkinter-based interface with real-time controls
- **Web Interface**: Browser-based interface accessible from any device
- **Command Line**: Traditional CLI for automated processing

See `web_interface3.jpg` for an example of the web interface in action.

## Controls

### Current Controls

The GUI and Web interfaces provide button-based controls for all functionality.

### Future Implementation - Keyboard Controls

*Note: Keyboard controls are planned for future implementation*

The following keyboard shortcuts are planned for the command-line interface:

- **Q**: Quit the application
- **S**: Save a screenshot
- **R**: Reset all metrics and counters
- **D**: Toggle debug view (shows detailed landmark visualization)

## Output Files

The system creates several output files in the `data/` directory:

### Data Files
- `eye_tracking_session_YYYYMMDD_HHMMSS.csv`: Detailed tracking data
- `session_summary_YYYYMMDD_HHMMSS.txt`: Session summary statistics

### Screenshots
- `screenshot_YYYYMMDD_HHMMSS.jpg`: Saved screenshots (when pressing 'S')

## Data Format

The CSV log file contains the following columns:

- `timestamp`: ISO format timestamp
- `session_id`: Unique session identifier
- `frame_number`: Sequential frame number
- `left_ear`: Left eye aspect ratio
- `right_ear`: Right eye aspect ratio
- `left_gaze_x`, `left_gaze_y`: Left eye gaze coordinates (normalized 0-1)
- `right_gaze_x`, `right_gaze_y`: Right eye gaze coordinates (normalized 0-1)
- `combined_gaze_x`, `combined_gaze_y`: Combined gaze coordinates (normalized 0-1)
- `is_blinking`: Boolean indicating if eyes are currently blinking
- `gesture`: Detected gesture type (`none`, `left_wink`, `right_wink`, `double_blink`, `sustained_gaze`)
- `gesture_duration`: Duration of current gesture in seconds
- `fps`: Current frames per second

## Technical Details

### Eye Aspect Ratio (EAR)
The system uses EAR for blink detection:
```
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
```
Where p1-p6 are specific eye landmark points. A threshold of 0.25 is used by default.

### Gaze Estimation
Gaze direction is calculated by:
1. Extracting iris position relative to eye boundaries
2. Normalizing coordinates to screen space (0-1 range)
3. Combining left and right eye data for overall gaze direction

### Gesture Recognition
- **Left Wink**: Left eye closed for 0.2-0.5 seconds while right eye open
- **Right Wink**: Right eye closed for 0.2-0.5 seconds while left eye open
- **Double Blink**: Two blinks within 0.5 seconds
- **Sustained Gaze**: Gaze held in same region for extended period

## Architecture

The system is modular with the following components:

- `main.py`: Main application and control flow
- `input_handler.py`: Video source abstraction (webcam/video file)
- `eye_tracker.py`: Core eye tracking algorithms and MediaPipe integration
- `visualizer.py`: Real-time visualization and overlay rendering
- `data_logger.py`: CSV logging and session management

## Troubleshooting

### Common Issues

1. **"Could not open webcam"**
   - Ensure webcam is connected and not used by another application
   - Check camera permissions
   - Try different camera index: modify `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

2. **Poor tracking accuracy**
   - Ensure good lighting conditions
   - Keep face centered in frame
   - Adjust `--ear-threshold` parameter for better sensitivity

3. **Low FPS performance**
   - Close other applications using the camera
   - Reduce video resolution if possible
   - Ensure adequate system resources

4. **"No face detected"**
   - Ensure face is visible and well-lit
   - Check that face is not too close or too far from camera
   - Verify MediaPipe installation

### Performance Tips

- Use good lighting for better landmark detection
- Keep face centered and at appropriate distance
- Close unnecessary applications to free up system resources
- For video files, ensure they have clear, well-lit faces

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Acknowledgments

- MediaPipe team for the excellent face mesh solution
- OpenCV community for computer vision tools
- Contributors to the Python scientific computing ecosystem
