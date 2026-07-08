# utils/camera_handler.py
import cv2
import threading
import time
import os
import sys
import numpy as np

# Try importing mvsdk from the parent/root directory if it exists
try:
    import mvsdk
except ImportError:
    # Try adding the project root to sys.path to locate mvsdk.py
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        import mvsdk
    except ImportError:
        mvsdk = None

class StandardCameraHandler:
    def __init__(self, camera_id=0, resolution=(1280, 720)):
        self.camera_id = camera_id
        self.resolution = resolution
        self.cap = None
        self.running = False
        self.frame = None
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        try:
            if isinstance(self.camera_id, str):
                self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_FFMPEG)
            else:
                self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_MSMF)
        except Exception as e:
            print(f"[CameraHandler] Driver Error: {e}")
            self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"[CameraHandler] Error: Could not open camera {self.camera_id}")
            return False
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return True

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def set_exposure(self, val):
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_EXPOSURE, float(val))

    def set_gain(self, val):
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_GAIN, float(val))

    def read(self):
        with self.lock:
            return self.frame is not None, self.frame

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()

class MindVisionCameraHandler:
    def __init__(self, camera_index=0, slice_height=200, canvas_multiplier=10, enhance=True, target_res=(640, 480)):
        self.camera_index = camera_index
        self.slice_height = slice_height
        self.canvas_multiplier = canvas_multiplier
        self.enhance = enhance
        self.target_res = target_res
        
        self.hCamera = 0
        self.pFrameBuffer = 0
        self.running = False
        self.frame = None
        self.lock = threading.Lock()
        self.thread = None
        
        self.width = 0
        self.height = 0
        self.canvas = None
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def start(self):
        if mvsdk is None:
            print("[CameraHandler] Error: mvsdk module not found. Cannot start MindVision Camera.")
            return False
        try:
            DevList = mvsdk.CameraEnumerateDevice()
            nDev = len(DevList)
            if nDev < 1:
                print("[CameraHandler] Error: No MindVision camera found!")
                return False

            if self.camera_index >= nDev:
                print(f"[CameraHandler] Error: Camera index {self.camera_index} is out of range ({nDev} available).")
                return False

            DevInfo = DevList[self.camera_index]
            print(f"[CameraHandler] Initializing MindVision Camera {self.camera_index}: {DevInfo.GetFriendlyName()}")

            self.hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
            
            # Capability & resolution configuration
            cap = mvsdk.CameraGetCapability(self.hCamera)
            res = mvsdk.CameraGetImageResolution(self.hCamera)
            res.iIndex = 0xff  # Custom ROI
            res.iHeight = self.slice_height
            res.iHeightFOV = self.slice_height
            mvsdk.CameraSetImageResolution(self.hCamera, res)

            # Retrieve verified resolution
            res = mvsdk.CameraGetImageResolution(self.hCamera)
            self.width = res.iWidth
            self.height = res.iHeight
            print(f"[CameraHandler] MindVision Slice Resolution: {self.width} x {self.height}")

            # Output BGR8 format from ISP
            mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

            # Freerun mode
            mvsdk.CameraSetTriggerMode(self.hCamera, 0)

            # Manual Exposure
            mvsdk.CameraSetAeState(self.hCamera, 0)
            try:
                min_exp, max_exp, step_exp = mvsdk.CameraGetExposureTimeRange(self.hCamera)
            except Exception:
                min_exp, max_exp, step_exp = 3.5, 5000.0, 0.1
            default_exp = min(max(100.0, min_exp), max_exp)
            mvsdk.CameraSetExposureTime(self.hCamera, default_exp)

            # Gain
            min_gain, max_gain = cap.sExposeDesc.uiAnalogGainMin, cap.sExposeDesc.uiAnalogGainMax
            default_gain = min(max(16, min_gain), max_gain)
            mvsdk.CameraSetAnalogGain(self.hCamera, default_gain)

            mvsdk.CameraPlay(self.hCamera)

            # Allocate ISP buffer
            FrameBufferSize = self.width * self.height * 3
            self.pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

            # Allocate rolling canvas
            canvas_height = self.height * self.canvas_multiplier
            self.canvas = np.zeros((canvas_height, self.width, 3), dtype=np.uint8)

            self.running = True
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
            return True
        except Exception as e:
            print(f"[CameraHandler] Failed to initialize MindVision camera: {e}")
            self.stop()
            return False

    def _update(self):
        while self.running:
            try:
                # Retrieve frame from camera (timeout = 200 ms)
                pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera, 200)
                mvsdk.CameraImageProcess(self.hCamera, pRawData, self.pFrameBuffer, FrameHead)
                mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)
                
                # Retrieve data from buffer
                frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.pFrameBuffer)
                raw_frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = raw_frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 3))

                # Image enhancement (CLAHE + Sharpening)
                if self.enhance:
                    gray = frame[:, :, 0]
                    gray_enhanced = self.clahe.apply(gray)
                    blur = cv2.GaussianBlur(gray_enhanced, (0, 0), 1.5)
                    sharpened = cv2.addWeighted(gray_enhanced, 1.6, blur, -0.6, 0)
                    frame = cv2.merge([sharpened, sharpened, sharpened])

                # Stitch to rolling canvas
                h_slice = FrameHead.iHeight
                if h_slice > 0:
                    canvas_height = self.canvas.shape[0]
                    if h_slice > canvas_height:
                        h_slice = canvas_height
                        frame = frame[-h_slice:, ...]
                    self.canvas[:-h_slice, ...] = self.canvas[h_slice:, ...]
                    self.canvas[-h_slice:, ...] = frame

                # Create output display frame (letterbox/preserve aspect ratio)
                canvas_h, canvas_w = self.canvas.shape[:2]
                target_w, target_h = self.target_res
                
                scale = min(target_w / canvas_w, target_h / canvas_h)
                nw = int(canvas_w * scale)
                nh = int(canvas_h * scale)
                
                resized = cv2.resize(self.canvas, (nw, nh), interpolation=cv2.INTER_LINEAR)
                
                out_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                y_offset = (target_h - nh) // 2
                x_offset = (target_w - nw) // 2
                out_frame[y_offset:y_offset+nh, x_offset:x_offset+nw] = resized
                
                # Draw neon cyan scanning entry line at the bottom of the stitched fabric
                entry_y = y_offset + nh - 2
                cv2.line(out_frame, (x_offset, entry_y), (x_offset + nw, entry_y), (0, 255, 255), 2)
                cv2.putText(out_frame, "LIVE SCANNING ENTRY", (x_offset + 10, entry_y - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

                with self.lock:
                    self.frame = out_frame
            except mvsdk.CameraException as e:
                if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                    print(f"[CameraHandler] MindVision grab error ({e.error_code}): {e.message}")
                    time.sleep(0.05)
            except Exception as e:
                print(f"[CameraHandler] MindVision error: {e}")
                time.sleep(0.05)

    def set_exposure(self, val):
        if self.hCamera != 0 and mvsdk is not None:
            try:
                min_exp, max_exp, step_exp = mvsdk.CameraGetExposureTimeRange(self.hCamera)
            except Exception:
                min_exp, max_exp = 3.5, 5000.0
            exposure = min(max(float(val), min_exp), max_exp)
            mvsdk.CameraSetExposureTime(self.hCamera, exposure)
            print(f"[MindVision] Exposure set to {exposure:.2f} ms")

    def set_gain(self, val):
        if self.hCamera != 0 and mvsdk is not None:
            try:
                cap = mvsdk.CameraGetCapability(self.hCamera)
                min_gain, max_gain = cap.sExposeDesc.uiAnalogGainMin, cap.sExposeDesc.uiAnalogGainMax
            except Exception:
                min_gain, max_gain = 16, 64
            gain = int(min(max(val, min_gain), max_gain))
            mvsdk.CameraSetAnalogGain(self.hCamera, gain)
            print(f"[MindVision] Gain set to {gain}")

    def set_slice_height(self, val):
        val = int(val)
        if val != self.slice_height:
            self.slice_height = val
            print(f"[MindVision] Slice height updated to {val}. Re-initializing camera stream.")
            if self.running:
                def restart():
                    self.stop()
                    self.start()
                threading.Thread(target=restart, daemon=True).start()

    def set_enhance(self, enhance):
        self.enhance = bool(enhance)
        print(f"[MindVision] Enhancements (CLAHE/Sharpen) set to {self.enhance}")

    def read(self):
        with self.lock:
            return self.frame is not None, self.frame

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.hCamera != 0:
            try:
                mvsdk.CameraStop(self.hCamera)
                mvsdk.CameraUnInit(self.hCamera)
            except Exception:
                pass
            self.hCamera = 0
        if self.pFrameBuffer != 0:
            try:
                mvsdk.CameraAlignFree(self.pFrameBuffer)
            except Exception:
                pass
            self.pFrameBuffer = 0

class CameraHandler:
    def __init__(self, camera_id=0, resolution=(1280, 720), use_mindvision=None):
        self.camera_id = camera_id
        self.resolution = resolution
        self.use_mindvision = use_mindvision
        self.handler = None
        self._init_handler()

    def _init_handler(self):
        is_mv = False
        if self.use_mindvision is True or self.camera_id == 'mindvision':
            is_mv = True
        elif self.use_mindvision is None:
            is_mv = (self.camera_id == 'mindvision')

        if is_mv:
            try:
                import mvsdk
                devices = mvsdk.CameraEnumerateDevice()
                if len(devices) > 0:
                    idx = 0
                    if isinstance(self.camera_id, int):
                        idx = self.camera_id
                    
                    try:
                        from utils.config_live import LIVE_CONFIG
                        slice_height = LIVE_CONFIG.get('mindvision_slice_height', 200)
                        canvas_mult = LIVE_CONFIG.get('mindvision_canvas_multiplier', 10)
                        enhance = LIVE_CONFIG.get('mindvision_enhance', True)
                    except ImportError:
                        slice_height = 200
                        canvas_mult = 10
                        enhance = True

                    self.handler = MindVisionCameraHandler(
                        camera_index=idx,
                        slice_height=slice_height,
                        canvas_multiplier=canvas_mult,
                        enhance=enhance,
                        target_res=self.resolution
                    )
                    print(f"[CameraHandler] Initialized MindVision Line-Scan Camera {idx}")
                    return
                else:
                    print("[CameraHandler] MindVision requested but 0 devices found. Falling back to webcam...")
            except Exception as e:
                print(f"[CameraHandler] Failed to load mvsdk: {e}. Falling back to webcam...")

        idx = 0
        if isinstance(self.camera_id, int):
            idx = self.camera_id
        self.handler = StandardCameraHandler(idx, self.resolution)
        print(f"[CameraHandler] Initialized Standard Webcam {idx}")

    def start(self):
        return self.handler.start()

    def read(self):
        return self.handler.read()

    def stop(self):
        return self.handler.stop()

    def set_exposure(self, val):
        if hasattr(self.handler, 'set_exposure'):
            self.handler.set_exposure(val)

    def set_gain(self, val):
        if hasattr(self.handler, 'set_gain'):
            self.handler.set_gain(val)

    def set_slice_height(self, val):
        if hasattr(self.handler, 'set_slice_height'):
            self.handler.set_slice_height(val)

    def set_enhance(self, val):
        if hasattr(self.handler, 'set_enhance'):
            self.handler.set_enhance(val)
