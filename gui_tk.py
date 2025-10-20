import threading
import time
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import font as tkfont
from PIL import Image, ImageTk
from typing import Optional, List, Tuple

from input_handler import InputHandler
from eye_tracker import EyeTracker, GestureType
from visualizer import Visualizer
from data_logger import DataLogger


class EyeTrackingGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MediaPipe Eye Tracking GUI")
        self.root.geometry("1280x800")

        # High-DPI scaling and modern theme
        self._init_scaling_and_theme()
        
        # State
        self.running = False
        self.capture_thread: Optional[threading.Thread] = None
        self.source_var = tk.StringVar(value="webcam")
        self.video_path_var = tk.StringVar(value="")
        self.ear_threshold_var = tk.DoubleVar(value=0.25)
        self.status_var = tk.StringVar(value="Idle")
        self.debug_var = tk.BooleanVar(value=False)  # Acts as Privacy Mode toggle
        self.show_video_var = tk.BooleanVar(value=True)  # Controls video display visibility
        self.mask_only_var = tk.BooleanVar(value=False)  # Controls mask-only mode
        
        # Components (initialized on start)
        self.input_handler: Optional[InputHandler] = None
        self.eye_tracker: Optional[EyeTracker] = None
        self.visualizer: Optional[Visualizer] = None
        self.data_logger: Optional[DataLogger] = None
        
        self.frame_label = None
        self.current_frame_bgr = None
        
        self._build_ui()
        
    def _init_scaling_and_theme(self):
        # DPI-aware scaling
        try:
            dpi = self.root.winfo_fpixels('1i')
            scaling = max(1.0, dpi / 96.0)
            self.root.tk.call('tk', 'scaling', scaling)
        except Exception:
            pass

        # Modern ttk theme and fonts
        style = ttk.Style(self.root)
        try:
            style.theme_use('clam')
        except Exception:
            pass
        default_font = tkfont.nametofont('TkDefaultFont')
        default_font.configure(size=11)
        text_font = tkfont.nametofont('TkTextFont')
        text_font.configure(size=11)
        heading_font = tkfont.nametofont('TkHeadingFont')
        heading_font.configure(size=12, weight='bold')
        style.configure('TLabel', padding=2)
        style.configure('TButton', padding=(10, 6))
        style.configure('TRadiobutton', padding=2)
        style.configure('TCheckbutton', padding=2)

    def _build_ui(self):
        controls = ttk.Frame(self.root, padding=10)
        controls.pack(side=tk.TOP, fill=tk.X)
        
        # Source selection
        ttk.Label(controls, text="Source:").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(controls, text="Webcam", variable=self.source_var, value="webcam").grid(row=0, column=1, sticky=tk.W)
        ttk.Radiobutton(controls, text="Video", variable=self.source_var, value="video").grid(row=0, column=2, sticky=tk.W)
        
        # Video path
        ttk.Label(controls, text="Video Path:").grid(row=0, column=3, sticky=tk.W, padx=(20, 0))
        path_entry = ttk.Entry(controls, textvariable=self.video_path_var, width=50)
        path_entry.grid(row=0, column=4, sticky=tk.W)
        ttk.Button(controls, text="Browse", command=self._browse_video).grid(row=0, column=5, sticky=tk.W, padx=(5, 0))
        
        # EAR threshold
        ttk.Label(controls, text="EAR Threshold:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        ear_scale = ttk.Scale(controls, from_=0.1, to=0.4, orient=tk.HORIZONTAL, variable=self.ear_threshold_var)
        ear_scale.grid(row=1, column=1, columnspan=2, sticky=tk.EW, pady=(10, 0))
        ttk.Label(controls, textvariable=self.ear_threshold_var).grid(row=1, column=3, sticky=tk.W, pady=(10, 0))
        
        # Buttons
        ttk.Button(controls, text="Start", command=self.start).grid(row=1, column=4, sticky=tk.W, padx=(10, 0))
        ttk.Button(controls, text="Close Camera", command=self.stop).grid(row=1, column=5, sticky=tk.W)
        ttk.Button(controls, text="Exit App", command=self.close_app).grid(row=1, column=6, sticky=tk.W)
        
        # Video display toggle
        ttk.Checkbutton(controls, text="Show Video", variable=self.show_video_var, command=self.toggle_video_display).grid(row=1, column=7, sticky=tk.W, padx=(15, 0))
        ttk.Checkbutton(controls, text="Mask Only", variable=self.mask_only_var, command=self.toggle_mask_only).grid(row=1, column=8, sticky=tk.W, padx=(15, 0))
        
        # Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W).pack(side=tk.LEFT, padx=10, pady=5)
        
        # Video display
        video_frame = ttk.Frame(self.root, padding=10)
        video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.frame_label = ttk.Label(video_frame)
        self.frame_label.pack(fill=tk.BOTH, expand=True)
        
    def _browse_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv"), ("All Files", "*.*")])
        if path:
            self.video_path_var.set(path)
        
    def start(self):
        if self.running:
            return
        
        # Validate source
        if self.source_var.get() == "video" and not self.video_path_var.get():
            messagebox.showerror("Error", "Please select a video file.")
            return
        
        self.running = True
        self.status_var.set("Starting...")
        
        # Initialize components
        self.input_handler = InputHandler(self.source_var.get(), self.video_path_var.get() or None)
        if not self.input_handler.initialize():
            messagebox.showerror("Error", "Failed to initialize input source.")
            self.running = False
            self.status_var.set("Idle")
            return
        
        width, height = self.input_handler.get_frame_dimensions()
        self.eye_tracker = EyeTracker(ear_threshold=self.ear_threshold_var.get())
        self.eye_tracker.set_frame_dimensions(width, height)
        self.visualizer = Visualizer(width, height)
        self.data_logger = DataLogger()
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.capture_thread.start()
        self.status_var.set("Running")
        
    def stop(self):
        if not self.running:
            return
        self.running = False
        self.status_var.set("Stopping...")
        
        # Wait for thread to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        # Cleanup
        try:
            if self.input_handler:
                self.input_handler.release()
            if self.eye_tracker:
                self.eye_tracker.release()
            if self.data_logger:
                self.data_logger.close()
        except Exception:
            pass
        
        self.status_var.set("Idle")
    
    def toggle_video_display(self):
        """Toggle video display visibility."""
        if self.show_video_var.get():
            self.frame_label.pack(fill=tk.BOTH, expand=True)
        else:
            self.frame_label.pack_forget()
    
    def toggle_mask_only(self):
        """Toggle mask-only mode."""
        if self.mask_only_var.get():
            self.status_var.set("Mask-only mode enabled")
        else:
            self.status_var.set("Normal mode")
    
    def close_app(self):
        """Close the entire application."""
        self.stop()
        self.root.quit()
        self.root.destroy()
        
    def _update_frame(self, frame_bgr):
        # Only update if video display is enabled
        if not self.show_video_var.get():
            return
            
        # Convert BGR to RGB for display
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        # Resize to fit label while preserving aspect with high-quality filter
        label_w = self.frame_label.winfo_width() or image.width
        label_h = self.frame_label.winfo_height() or image.height
        if label_w > 0 and label_h > 0:
            src_w, src_h = image.size
            scale = min(label_w / src_w, label_h / src_h)
            new_w = max(1, int(src_w * scale))
            new_h = max(1, int(src_h * scale))
            image = image.resize((new_w, new_h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image=image)
        
        # Keep reference
        self.frame_label.photo = photo
        self.frame_label.configure(image=photo)
        
    def _run_loop(self):
        fps_list = []
        last_time = 0.0
        frame_count = 0
        
        try:
            while self.running:
                success, frame = self.input_handler.read_frame()
                if not success:
                    if self.input_handler.is_video_file():
                        self.status_var.set("End of video reached")
                    break
                
                frame_count += 1
                now = time.time()
                if last_time != 0:
                    fps = 1.0 / (now - last_time)
                    fps_list.append(fps)
                    if len(fps_list) > 30:
                        fps_list.pop(0)
                last_time = now
                avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0.0
                
                eye_data = self.eye_tracker.process_frame(frame, now)
                if eye_data is not None:
                    # Log
                    self.data_logger.log_eye_data(eye_data, frame_count, avg_fps)
                    
                    # Create base frame for visualization
                    if self.mask_only_var.get():
                        # Mask-only mode: black background with landmarks and overlays
                        base_frame = np.zeros_like(frame)
                    else:
                        # Normal mode: original frame
                        base_frame = frame
                    
                    # Visualize standard overlays (no privacy blur)
                    gaze_history = self.eye_tracker.get_gaze_history()
                    vis_frame = self.visualizer.visualize_frame(
                        base_frame,
                        eye_data,
                        gaze_history,
                        avg_fps,
                        False,
                        self.mask_only_var.get()
                    )
                else:
                    self.data_logger.log_no_face_detected(frame_count, avg_fps)
                    if self.mask_only_var.get():
                        vis_frame = np.zeros_like(frame)
                        cv2.putText(vis_frame, "No face detected", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    else:
                        vis_frame = frame
                        cv2.putText(vis_frame, "No face detected", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                
                self.current_frame_bgr = vis_frame
                self.root.after(0, self._update_frame, vis_frame.copy())
                
            self.stop()
        except Exception as e:
            self.status_var.set(f"Error: {e}")
            self.stop()

    def _apply_privacy_mask(self, frame_bgr, landmarks: Optional[List[Tuple[int, int]]]):
        # If landmarks are unavailable, just blur entire face region is not possible; blur whole frame
        if not landmarks or len(landmarks) == 0:
            return cv2.GaussianBlur(frame_bgr, (31, 31), 0)

        h, w = frame_bgr.shape[:2]

        # Build blurred background
        blurred = cv2.GaussianBlur(frame_bgr, (51, 51), 0)

        # Compute eye bounding boxes from known indices
        def bbox_from_indices(indices: List[int]):
            pts = [(x, y) for i, (x, y) in enumerate(landmarks) if i in indices]
            if not pts:
                return None
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            # Add margin
            mx = int(0.25 * (x2 - x1 + 1))
            my = int(0.5 * (y2 - y1 + 1))
            x1 = max(0, x1 - mx)
            y1 = max(0, y1 - my)
            x2 = min(w - 1, x2 + mx)
            y2 = min(h - 1, y2 + my)
            return x1, y1, x2, y2

        left_box = bbox_from_indices(EyeTracker.LEFT_EYE_INDICES)
        right_box = bbox_from_indices(EyeTracker.RIGHT_EYE_INDICES)

        # Start with blurred frame, then paste original eye regions back
        output = blurred.copy()
        for box in [left_box, right_box]:
            if box is None:
                continue
            x1, y1, x2, y2 = box
            output[y1:y2, x1:x2] = frame_bgr[y1:y2, x1:x2]

        return output


def main():
    root = tk.Tk()
    app = EyeTrackingGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.close_app)
    root.mainloop()


if __name__ == "__main__":
    main()


