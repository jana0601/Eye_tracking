import threading
import time
from typing import Optional, List, Tuple

import cv2
import numpy as np
from flask import Flask, Response, render_template, request, redirect, url_for

from input_handler import InputHandler
from eye_tracker import EyeTracker
from visualizer import Visualizer
from data_logger import DataLogger


app = Flask(__name__)


class StreamState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.input_handler: Optional[InputHandler] = None
        self.eye_tracker: Optional[EyeTracker] = None
        self.visualizer: Optional[Visualizer] = None
        self.data_logger: Optional[DataLogger] = None
        self.frame_bgr = None
        # No privacy toggle; always render standard overlays
        self.source = 'webcam'
        self.video_path: Optional[str] = None
        self.avg_fps = 0.0

    def start(self, source: str = 'webcam', video_path: Optional[str] = None, ear_threshold: float = 0.25):
        with self.lock:
            if self.running:
                return
            self.source = source
            self.video_path = video_path
            self.running = True
            self.thread = threading.Thread(target=self._run_loop, args=(ear_threshold,), daemon=True)
            self.thread.start()

    def stop(self):
        with self.lock:
            self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self._cleanup()

    def _cleanup(self):
        try:
            if self.input_handler:
                self.input_handler.release()
            if self.eye_tracker:
                self.eye_tracker.release()
            if self.data_logger:
                self.data_logger.close()
        except Exception:
            pass
        self.input_handler = None
        self.eye_tracker = None
        self.visualizer = None
        self.data_logger = None
        self.frame_bgr = None

    # Removed privacy mask function; standard rendering only

    def _run_loop(self, ear_threshold: float):
        try:
            self.input_handler = InputHandler(self.source, self.video_path)
            if not self.input_handler.initialize():
                self.running = False
                return
            width, height = self.input_handler.get_frame_dimensions()
            self.eye_tracker = EyeTracker(ear_threshold=ear_threshold)
            self.eye_tracker.set_frame_dimensions(width, height)
            self.visualizer = Visualizer(width, height)
            self.data_logger = DataLogger()

            fps_list = []
            last_time = 0.0
            frame_idx = 0

            while self.running:
                ok, frame = self.input_handler.read_frame()
                if not ok:
                    break
                frame_idx += 1
                now = time.time()
                if last_time != 0:
                    fps = 1.0 / (now - last_time)
                    fps_list.append(fps)
                    if len(fps_list) > 30:
                        fps_list.pop(0)
                last_time = now
                self.avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0.0

                eye_data = self.eye_tracker.process_frame(frame, now)
                if eye_data is not None:
                    self.data_logger.log_eye_data(eye_data, frame_idx, self.avg_fps)
                    # Apply privacy mask to raw frame first, then draw overlays so they stay sharp
                    vis = self.visualizer.visualize_frame(
                        frame, eye_data, self.eye_tracker.get_gaze_history(), self.avg_fps, False
                    )
                else:
                    self.data_logger.log_no_face_detected(frame_idx, self.avg_fps)
                    vis = frame
                    cv2.putText(vis, "No face detected", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                self.frame_bgr = vis

        finally:
            self._cleanup()


stream = StreamState()


@app.route('/')
def index():
    return render_template('index.html', debug=stream.debug, privacy=stream.privacy, running=stream.running, fps=f"{stream.avg_fps:.1f}")


def _jpeg_generator():
    # Start stream on first connection if not running
    if not stream.running:
        stream.start('webcam', None)

    while True:
        frame = stream.frame_bgr
        if frame is None:
            time.sleep(0.01)
            continue
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ret:
            continue
        jpg = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(_jpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/control', methods=['POST'])
def control():
    action = request.form.get('action')
    if action == 'start' and not stream.running:
        source = request.form.get('source', 'webcam')
        path = request.form.get('path') or None
        ear = float(request.form.get('ear', '0.25'))
        stream.start(source, path, ear)
    elif action == 'stop' and stream.running:
        stream.stop()
    elif action == 'set_flags':
        stream.debug = request.form.get('debug') == 'on'
        stream.privacy = request.form.get('privacy') == 'on'
    return redirect(url_for('index'))


if __name__ == '__main__':
    # Run with: python web_app.py, then open http://127.0.0.1:5000/
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)


