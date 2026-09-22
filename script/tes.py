import customtkinter as ctk
import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import datetime
import os
from tkinter import filedialog, messagebox
import threading
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import platform
import re
import gc
import sys
from scipy.spatial import distance as dist
from collections import OrderedDict

# ==================== SERIAL COMMUNICATION ====================
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("⚠️ PySerial tidak terinstall. Install dengan: pip install pyserial")

# ==================== CONFIGURATION ====================
tcl_dir = os.path.join(sys.base_prefix, 'tcl', 'tcl8.6')
tk_dir = os.path.join(sys.base_prefix, 'tcl', 'tk8.6')
os.environ['TCL_LIBRARY'] = tcl_dir
os.environ['TK_LIBRARY'] = tk_dir

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

# Constants
LOG_FILE = "log_deteksi.csv"
CONFIG_FILE = "app_config.json"
REJECTED_IMAGES_FOLDER = "rejected_images"
MODEL_PATH = Path("models/best.pt")
TARGET_FPS = 30

# Detection Zone Configuration
DETECTION_ZONE_X = 320
DETECTION_ZONE_TOLERANCE = 50
DECISION_COOLDOWN = 2.0

EMOJI = {
    'egg': '🥚', 'settings': '⚙️', 'chart': '📊', 'check': '✅', 
    'cross': '❌', 'camera': '📹', 'brain': '🧠', 'lightning': '⚡',
    'clock': '⏱️', 'bell': '🔔', 'save': '💾', 'folder': '📂',
    'refresh': '🔄', 'upload': '📤', 'play': '▶️', 'stop': '🛑',
    'search': '🔍', 'down': '⬇️', 'usb': '🔌', 'target': '🎯'
}

os.makedirs(REJECTED_IMAGES_FOLDER, exist_ok=True)
os.makedirs("models", exist_ok=True)


# ==================== CENTROID TRACKER ====================
class CentroidTracker:
    """Track objects menggunakan centroid mereka"""
    def __init__(self, maxDisappeared=30, maxDistance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance
        
    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
    
    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
    
    def update(self, rects):
        """Update tracker dengan deteksi baru"""
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects
        
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
        
        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            usedRows = set()
            usedCols = set()
            
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                
                if D[row, col] > self.maxDistance:
                    continue
                
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            
            unusedRows = set(range(D.shape[0])).difference(usedRows)
            unusedCols = set(range(D.shape[1])).difference(usedCols)
            
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
        
        return self.objects


# ==================== SERIAL MANAGER ====================
class SerialManager:
    """Manage serial communication dengan mikrokontroler"""
    def __init__(self):
        self.serial_port = None
        self.connected = False
        self.lock = threading.Lock()
        
    def list_ports(self):
        """List available serial ports"""
        if not SERIAL_AVAILABLE:
            return []
        ports = serial.tools.list_ports.comports()
        return [(port.device, port.description) for port in ports]
    
    def connect(self, port, baudrate=9600):
        """Connect ke serial port"""
        try:
            with self.lock:
                if self.connected:
                    self.disconnect()
                
                self.serial_port = serial.Serial(port, baudrate, timeout=1)
                time.sleep(2)
                self.connected = True
                print(f"✅ Connected to {port} at {baudrate} baud")
                return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect dari serial port"""
        try:
            with self.lock:
                if self.serial_port and self.connected:
                    self.serial_port.close()
                    self.connected = False
                    print("🔌 Serial disconnected")
                    return True
        except Exception as e:
            print(f"Error disconnecting: {e}")
        return False
    
    def send_decision(self, decision):
        """Kirim keputusan ke mikrokontroler"""
        if not self.connected or not self.serial_port:
            return False
        
        try:
            with self.lock:
                command = 'A' if decision == "ACCEPT" else 'R'
                self.serial_port.write(command.encode())
                self.serial_port.flush()
                print(f"📤 Sent to Arduino: {command} ({decision})")
                return True
        except Exception as e:
            print(f"Error sending data: {e}")
            return False


# ==================== CONFIG CLASS ====================
class Config:
    """Configuration manager"""
    def __init__(self):
        self.ip_camera = "192.168.1.6"
        self.min_conf = 0.3
        self.aktifkan_log = True
        self.device = "cpu"
        self.alert_enabled = True
        self.alert_sound_enabled = True
        self.serial_port = ""
        self.serial_baudrate = 9600
        self.detection_zone_x = DETECTION_ZONE_X
        self.detection_zone_tolerance = DETECTION_ZONE_TOLERANCE
        self.decision_cooldown = DECISION_COOLDOWN
    
    def load(self):
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    for key in self.__dict__:
                        if key in data:
                            setattr(self, key, data[key])
                print(f"✅ Configuration loaded")
        except Exception as e:
            print(f"Could not load configuration: {e}")
    
    def save(self):
        try:
            data = self.__dict__.copy()
            data["last_saved"] = datetime.datetime.now().isoformat()
            with open(CONFIG_FILE, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"✅ Configuration saved")
            return True
        except Exception as e:
            print(f"Could not save configuration: {e}")
            return False


# ==================== MODEL MANAGER ====================
class ModelManager:
    """YOLO model manager"""
    def __init__(self):
        self.model = None
        self.cuda_available = False
    
    def load(self):
        try:
            model_file = MODEL_PATH
            if not model_file.exists():
                messagebox.showwarning("Model Not Found", 
                    f"Model tidak ditemukan di:\n{MODEL_PATH.absolute()}\n\n"
                    "Silakan pilih file model YOLO (.pt)")
                
                model_path = filedialog.askopenfilename(
                    title="Pilih File Model YOLO",
                    filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
                )
                if not model_path:
                    messagebox.showerror("Error", "Model tidak dipilih. Aplikasi akan ditutup.")
                    return False
                model_file = Path(model_path)
            
            print(f"Loading model from: {model_file.absolute()}")
            self.model = YOLO(str(model_file))
            print(f"✅ Model loaded successfully")
            print(f"✅ Detected classes: {self.model.names}")
            
            import torch
            self.cuda_available = torch.cuda.is_available()
            if self.cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"✅ CUDA available: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                print(f"ℹ️ CUDA not available, using CPU only")
            
            return True
        except Exception as e:
            messagebox.showerror("Error Loading Model",
                f"Gagal memuat model YOLO:\n{str(e)}")
            return False
    
    def predict(self, image, conf, device):
        return self.model.predict(image, verbose=False, device=device, conf=conf, half=False)


# ==================== CAMERA MANAGER ====================
class CameraManager:
    """Camera detection and management"""
    @staticmethod
    def detect_cameras():
        available = []
        print("Detecting available cameras...")
        
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                backend = cap.getBackendName()
                ret, _ = cap.read()
                
                if ret:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    camera_name = f"📹 {'Default' if i == 0 else f'Camera {i}'} ({width}x{height})"
                    
                    available.append({
                        "index": i,
                        "name": camera_name,
                        "backend": backend,
                        "resolution": f"{width}x{height}"
                    })
                    print(f"✅ Found: {camera_name} [{backend}]")
                
                cap.release()
            time.sleep(0.1)
        
        if not available:
            print("⚠️ No cameras detected")
            available.append({
                "index": 0,
                "name": "📹 Default Camera (Not detected)",
                "backend": "unknown",
                "resolution": "unknown"
            })
        
        return available
    
    @staticmethod
    def validate_ip(ip):
        pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        if not re.match(pattern, ip):
            return False
        try:
            return all(0 <= int(part) <= 255 for part in ip.split('.'))
        except:
            return False


# ==================== LOG MANAGER ====================
class LogManager:
    """Log management"""
    def __init__(self):
        self.log_deteksi = []
        self.lock = threading.Lock()
    
    def load(self):
        try:
            df = pd.read_csv(LOG_FILE)
            self.log_deteksi = df.to_dict(orient="records")
            print(f"✅ Loaded {len(self.log_deteksi)} log entries")
            return self.log_deteksi
        except FileNotFoundError:
            print("No existing log file found")
            return []
        except Exception as e:
            print(f"Error loading log: {e}")
            return []
    
    def save_append(self, log_data):
        try:
            with self.lock:
                df_new = pd.DataFrame(log_data)
                df_new.to_csv(LOG_FILE, mode='a', index=False, 
                             header=not os.path.exists(LOG_FILE))
        except Exception as e:
            print(f"Error saving log: {e}")
    
    def add_entry(self, entry):
        with self.lock:
            self.log_deteksi.append(entry)
    
    def reset(self):
        with self.lock:
            self.log_deteksi = []
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)


# ==================== MAIN APPLICATION ====================
class EggSorterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title(f"{EMOJI['egg']} Sortir Kualitas Telur Otomatis - Conveyor Mode")
        self.geometry("1600x900")
        
        # Initialize managers
        self.config = Config()
        self.model_manager = ModelManager()
        self.log_manager = LogManager()
        self.serial_manager = SerialManager()
        self.tracker = CentroidTracker(maxDisappeared=30, maxDistance=80)
        
        if not self.model_manager.load():
            self.destroy()
            return
        
        self.init_state()
        self.config.load()
        self.log_manager.load()
        self.rebuild_stats()
        self.setup_ui()
        
        self.bind("<Escape>", lambda e: self.stop_camera_feed() if self.camera_running else None)
        self.bind("<F5>", lambda e: self.refresh_log())
    
    def init_state(self):
        """Initialize application state"""
        self.stats_lock = threading.Lock()
        self.reject_count = 0
        self.accept_count = 0
        self.stop_camera = False
        self.camera_running = False
        self.last_log_time = datetime.datetime.min
        self.camera_source = "local"
        self.available_cameras = []
        self.local_camera_index = 0
        self.fps = 0
        self.inference_time = 0
        self.frame_times = []
        self.last_frame_time = time.time()
        self.chart_data = {"timestamps": [], "accept": [], "reject": []}
        self.max_chart_points = 50
        
        # Tracking state
        self.processed_objects = {}
        self.last_decision_time = 0
        self.current_frame = None
    
    def rebuild_stats(self):
        """Rebuild statistics from log"""
        logs = self.log_manager.log_deteksi
        accept = sum(1 for log in logs if log.get("keputusan") == "ACCEPT")
        reject = sum(1 for log in logs if log.get("keputusan") == "REJECT")
        
        with self.stats_lock:
            self.accept_count = accept
            self.reject_count = reject
        
        temp_accept = temp_reject = 0
        for log in logs[-self.max_chart_points:]:
            if log.get("keputusan") == "ACCEPT":
                temp_accept += 1
            elif log.get("keputusan") == "REJECT":
                temp_reject += 1
            
            self.chart_data["timestamps"].append(log.get("waktu", ""))
            self.chart_data["accept"].append(temp_accept)
            self.chart_data["reject"].append(temp_reject)
    
    def setup_ui(self):
        """Setup main UI with horizontal layout"""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Top bar with statistics
        self.setup_top_bar()
        
        # Main content area
        self.main_frame = ctk.CTkFrame(self, corner_radius=0)
        self.main_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        
        # Tabview for all features
        self.setup_main_tabview()
    
    def setup_top_bar(self):
        """Setup top statistics bar"""
        top_bar = ctk.CTkFrame(self, height=80, corner_radius=0, fg_color="#1a1a1a")
        top_bar.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        top_bar.grid_columnconfigure((0,1,2,3,4,5), weight=1)
        
        # Title
        ctk.CTkLabel(top_bar, text=f"{EMOJI['egg']} Egg Sorter",
                    font=ctk.CTkFont(size=24, weight="bold")).grid(
                        row=0, column=0, padx=20, pady=10, sticky="w")
        
        # Statistics
        stats_frame = ctk.CTkFrame(top_bar, fg_color="transparent")
        stats_frame.grid(row=0, column=1, columnspan=5, padx=20, pady=10, sticky="e")
        
        # ACCEPT stat
        accept_frame = ctk.CTkFrame(stats_frame, fg_color="#1e4d2b", corner_radius=8)
        accept_frame.pack(side="left", padx=5)
        self.top_accept_label = ctk.CTkLabel(accept_frame, 
            text=f"{EMOJI['check']} ACCEPT\n{self.accept_count}",
            font=ctk.CTkFont(size=14, weight="bold"), text_color="#2ecc71")
        self.top_accept_label.pack(padx=20, pady=10)
        
        # REJECT stat
        reject_frame = ctk.CTkFrame(stats_frame, fg_color="#4d1e1e", corner_radius=8)
        reject_frame.pack(side="left", padx=5)
        self.top_reject_label = ctk.CTkLabel(reject_frame,
            text=f"{EMOJI['cross']} REJECT\n{self.reject_count}",
            font=ctk.CTkFont(size=14, weight="bold"), text_color="#e74c3c")
        self.top_reject_label.pack(padx=20, pady=10)
        
        # FPS stat
        perf_frame = ctk.CTkFrame(stats_frame, fg_color="#1e3a4d", corner_radius=8)
        perf_frame.pack(side="left", padx=5)
        self.top_perf_label = ctk.CTkLabel(perf_frame,
            text=f"{EMOJI['lightning']} FPS\n0.0",
            font=ctk.CTkFont(size=14, weight="bold"), text_color="#3498db")
        self.top_perf_label.pack(padx=20, pady=10)
        
        # Tracked stat
        track_frame = ctk.CTkFrame(stats_frame, fg_color="#4d3a1e", corner_radius=8)
        track_frame.pack(side="left", padx=5)
        self.top_tracked_label = ctk.CTkLabel(track_frame,
            text=f"{EMOJI['target']} Tracked\n0",
            font=ctk.CTkFont(size=14, weight="bold"), text_color="#f39c12")
        self.top_tracked_label.pack(padx=20, pady=10)
        
        # Serial status
        serial_frame = ctk.CTkFrame(stats_frame, fg_color="#2b2b2b", corner_radius=8)
        serial_frame.pack(side="left", padx=5)
        self.top_serial_label = ctk.CTkLabel(serial_frame,
            text=f"{EMOJI['usb']} Serial\n⚫ Off",
            font=ctk.CTkFont(size=14, weight="bold"), text_color="gray")
        self.top_serial_label.pack(padx=20, pady=10)
    
    def setup_main_tabview(self):
        """Setup main tabview with all features"""
        self.main_tabview = ctk.CTkTabview(self.main_frame)
        self.main_tabview.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Tabs
        self.setup_detection_tab()
        self.setup_camera_settings_tab()
        self.setup_detection_settings_tab()
        self.setup_serial_tab()
        self.setup_model_settings_tab()
        self.setup_log_tab()
        self.setup_chart_tab()
    
    def setup_detection_tab(self):
        """Setup detection tab"""
        tab = self.main_tabview.add(f"{EMOJI['camera']} Detection")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)
        
        # Control buttons
        control_frame = ctk.CTkFrame(tab, height=80)
        control_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        control_frame.grid_columnconfigure((0,1,2,3), weight=1)
        
        ctk.CTkButton(control_frame, text=f"{EMOJI['upload']} Upload Gambar",
                     command=self.upload_image, height=50, font=ctk.CTkFont(size=14)
                     ).grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        self.start_cam_btn = ctk.CTkButton(control_frame, text=f"{EMOJI['play']} Mulai Kamera",
                                          command=self.start_camera, fg_color="#27ae60", 
                                          hover_color="#229954", height=50, font=ctk.CTkFont(size=14))
        self.start_cam_btn.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        
        self.stop_cam_btn = ctk.CTkButton(control_frame, text=f"{EMOJI['stop']} Stop Kamera",
                                         command=self.stop_camera_feed, fg_color="#e74c3c", 
                                         hover_color="#c0392b", height=50, state="disabled",
                                         font=ctk.CTkFont(size=14))
        self.stop_cam_btn.grid(row=0, column=2, padx=10, pady=10, sticky="ew")
        
        ctk.CTkButton(control_frame, text=f"{EMOJI['refresh']} Reset Data",
                     command=self.reset_data, fg_color="#8e44ad", hover_color="#7d3c98",
                     height=50, font=ctk.CTkFont(size=14)
                     ).grid(row=0, column=3, padx=10, pady=10, sticky="ew")
        
        # Image display
        image_frame = ctk.CTkFrame(tab)
        image_frame.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        image_frame.grid_rowconfigure(0, weight=1)
        image_frame.grid_columnconfigure(0, weight=1)
        
        self.image_label = ctk.CTkLabel(image_frame, text="Tidak ada gambar",
                                       font=ctk.CTkFont(size=16))
        self.image_label.grid(row=0, column=0, sticky="nsew")
        
        # Decision label
        self.keputusan_label = ctk.CTkLabel(tab, text="", 
                                           font=ctk.CTkFont(size=24, weight="bold"))
        self.keputusan_label.grid(row=2, column=0, padx=20, pady=10)
        
        # Info
        info_frame = ctk.CTkFrame(tab, fg_color="#2b2b2b", corner_radius=8)
        info_frame.grid(row=3, column=0, padx=20, pady=(0, 20), sticky="ew")
        
        ctk.CTkLabel(info_frame, text="ℹ️ Conveyor Mode: Telur dihitung 1x saat melewati Detection Zone",
                    font=ctk.CTkFont(size=12), text_color="gray").pack(pady=10)
        ctk.CTkLabel(info_frame, text="📋 Kelas: ✅ ACCEPT (clean, yellow egg) | ❌ REJECT (crack, dirty)",
                    font=ctk.CTkFont(size=11), text_color="gray").pack(pady=(0, 10))
    
    def setup_camera_settings_tab(self):
        """Setup camera settings tab"""
        tab = self.main_tabview.add(f"{EMOJI['camera']} Camera")
        tab.grid_columnconfigure((0,1), weight=1)
        
        # Left column - Camera Source
        left_frame = ctk.CTkFrame(tab)
        left_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        ctk.CTkLabel(left_frame, text="Sumber Kamera", 
                    font=ctk.CTkFont(size=18, weight="bold")).pack(pady=20)
        
        self.camera_source_var = ctk.StringVar(value="local")
        source_menu = ctk.CTkSegmentedButton(left_frame, values=["Webcam", "DroidCam IP"],
                                             variable=self.camera_source_var, 
                                             command=self.change_camera_source, height=40)
        source_menu.pack(pady=10, padx=20, fill="x")
        source_menu.set("Webcam")
        
        # Webcam selection
        self.local_cam_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        self.local_cam_frame.pack(pady=20, padx=20, fill="x")
        
        ctk.CTkLabel(self.local_cam_frame, text="Pilih Kamera Lokal:",
                    font=ctk.CTkFont(size=14)).pack(anchor="w", pady=(0, 10))
        
        cam_frame = ctk.CTkFrame(self.local_cam_frame, fg_color="transparent")
        cam_frame.pack(fill="x")
        
        self.local_cam_var = ctk.StringVar(value="Detecting...")
        self.local_cam_menu = ctk.CTkOptionMenu(cam_frame, values=["Detecting..."],
                                               variable=self.local_cam_var, height=40)
        self.local_cam_menu.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        self.detect_cam_btn = ctk.CTkButton(cam_frame, text=f"{EMOJI['search']}", 
                                           width=50, height=40, command=self.refresh_cameras)
        self.detect_cam_btn.pack(side="left")
        
        threading.Thread(target=self.initial_camera_detection, daemon=True).start()
        
        # IP Camera
        self.ip_cam_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        self.ip_cam_frame.pack(pady=20, padx=20, fill="x")
        
        ctk.CTkLabel(self.ip_cam_frame, text="IP Address DroidCam:",
                    font=ctk.CTkFont(size=14)).pack(anchor="w", pady=(0, 10))
        self.ip_entry = ctk.CTkEntry(self.ip_cam_frame, placeholder_text="192.168.1.6", 
                                     height=40, font=ctk.CTkFont(size=14))
        self.ip_entry.insert(0, self.config.ip_camera)
        self.ip_entry.pack(fill="x")
        
        self.ip_cam_frame.pack_forget()
        
        # Right column - Camera Info
        right_frame = ctk.CTkFrame(tab)
        right_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        ctk.CTkLabel(right_frame, text="Informasi Kamera",
                    font=ctk.CTkFont(size=18, weight="bold")).pack(pady=20)
        
        info_text = """
📹 Webcam: Gunakan kamera lokal (USB/Built-in)
        
📱 DroidCam IP: Gunakan smartphone sebagai kamera
   • Install app DroidCam di Android/iOS
   • Masukkan IP address yang ditampilkan di app
   • Pastikan device dalam jaringan yang sama
   
💡 Tips:
   • Posisikan kamera tegak lurus conveyor
   • Pastikan pencahayaan cukup
   • Jarak ideal: 30-50cm dari telur
   • Hindari backlight
        """
        
        ctk.CTkLabel(right_frame, text=info_text, justify="left",
                    font=ctk.CTkFont(size=12)).pack(pady=20, padx=20, anchor="w")
    
    def setup_detection_settings_tab(self):
        """Setup detection settings tab"""
        tab = self.main_tabview.add(f"{EMOJI['target']} Detection Zone")
        tab.grid_columnconfigure((0,1), weight=1)
        
        # Left - Settings
        left_frame = ctk.CTkFrame(tab)
        left_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        ctk.CTkLabel(left_frame, text="Pengaturan Zona Deteksi",
                    font=ctk.CTkFont(size=18, weight="bold")).pack(pady=20)
        
        # Confidence
        conf_frame = ctk.CTkFrame(left_frame, fg_color="#2b2b2b")
        conf_frame.pack(pady=10, padx=20, fill="x")
        
        ctk.CTkLabel(conf_frame, text="Minimum Confidence:",
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(15, 5))
        
        self.conf_slider = ctk.CTkSlider(conf_frame, from_=0.0, to=1.0, 
                                        command=self.update_conf, height=20)
        self.conf_slider.set(self.config.min_conf)
        self.conf_slider.pack(pady=10, padx=20, fill="x")
        
        self.conf_value = ctk.CTkLabel(conf_frame, text=f"{self.config.min_conf:.2f}",
                                      font=ctk.CTkFont(size=16, weight="bold"))
        self.conf_value.pack(pady=(0, 15))
        
        # Detection Zone X
        zone_x_frame = ctk.CTkFrame(left_frame, fg_color="#2b2b2b")
        zone_x_frame.pack(pady=10, padx=20, fill="x")
        
        ctk.CTkLabel(zone_x_frame, text="Detection Zone X (pixel):",
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(15, 5))
        
        self.zone_x_slider = ctk.CTkSlider(zone_x_frame, from_=100, to=540,
                                          command=self.update_zone_x, height=20)
        self.zone_x_slider.set(self.config.detection_zone_x)
        self.zone_x_slider.pack(pady=10, padx=20, fill="x")
        
        self.zone_x_value = ctk.CTkLabel(zone_x_frame, 
                                        text=f"{self.config.detection_zone_x} px",
                                        font=ctk.CTkFont(size=16, weight="bold"))
        self.zone_x_value.pack(pady=(0, 15))
        
        # Tolerance
        tolerance_frame = ctk.CTkFrame(left_frame, fg_color="#2b2b2b")
        tolerance_frame.pack(pady=10, padx=20, fill="x")
        
        ctk.CTkLabel(tolerance_frame, text="Zone Tolerance (±pixel):",
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(15, 5))
        
        self.tolerance_slider = ctk.CTkSlider(tolerance_frame, from_=10, to=100,
                                             command=self.update_tolerance, height=20)
        self.tolerance_slider.set(self.config.detection_zone_tolerance)
        self.tolerance_slider.pack(pady=10, padx=20, fill="x")
        
        self.tolerance_value = ctk.CTkLabel(tolerance_frame,
                                           text=f"±{self.config.detection_zone_tolerance} px",
                                           font=ctk.CTkFont(size=16, weight="bold"))
        self.tolerance_value.pack(pady=(0, 15))
        
        # Cooldown
        cooldown_frame = ctk.CTkFrame(left_frame, fg_color="#2b2b2b")
        cooldown_frame.pack(pady=10, padx=20, fill="x")
        
        ctk.CTkLabel(cooldown_frame, text="Decision Cooldown (detik):",
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(15, 5))
        
        self.cooldown_slider = ctk.CTkSlider(cooldown_frame, from_=0.5, to=5.0,
                                            command=self.update_cooldown, height=20)
        self.cooldown_slider.set(self.config.decision_cooldown)
        self.cooldown_slider.pack(pady=10, padx=20, fill="x")
        
        self.cooldown_value = ctk.CTkLabel(cooldown_frame,
                                          text=f"{self.config.decision_cooldown:.1f} s",
                                          font=ctk.CTkFont(size=16, weight="bold"))
        self.cooldown_value.pack(pady=(0, 15))
        
        # Right - Info
        right_frame = ctk.CTkFrame(tab)
        right_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        ctk.CTkLabel(right_frame, text="Penjelasan Parameter",
                    font=ctk.CTkFont(size=18, weight="bold")).pack(pady=20)
        
        info_text = """
🎯 Detection Zone X:
   Posisi garis vertikal zona deteksi
   • Default: 320px (tengah frame 640px)
   • Sesuaikan dengan posisi conveyor
   
📏 Zone Tolerance:
   Lebar zona deteksi (kiri-kanan dari garis)
   • Semakin besar = zona lebih lebar
   • Recommended: 40-60px
   
⏱️ Decision Cooldown:
   Jeda waktu minimum antar keputusan
   • Mencegah deteksi ganda
   • Sesuaikan dengan kecepatan conveyor
   • Recommended: 1.5-2.5 detik
   
🧠 Minimum Confidence:
   Threshold keyakinan model
   • Lebih tinggi = lebih yakin tapi kurang sensitif
   • Lebih rendah = lebih sensitif tapi banyak false positive
   • Recommended: 0.25-0.40
        """
        
        ctk.CTkLabel(right_frame, text=info_text, justify="left",
                    font=ctk.CTkFont(size=11)).pack(pady=20, padx=20, anchor="w")
    
    def setup_serial_tab(self):
        """Setup serial communication tab"""
        tab = self.main_tabview.add(f"{EMOJI['usb']} Serial Port")
        tab.grid_columnconfigure((0,1), weight=1)
        
        # Left - Connection
        left_frame = ctk.CTkFrame(tab)
        left_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        ctk.CTkLabel(left_frame, text="Koneksi Arduino/ESP32",
                    font=ctk.CTkFont(size=18, weight="bold")).pack(pady=20)
        
        # Port selection
        port_frame = ctk.CTkFrame(left_frame, fg_color="#2b2b2b")
        port_frame.pack(pady=10, padx=20, fill="x")
        
        ctk.CTkLabel(port_frame, text="Select COM Port:",
                    font=ctk.CTkFont(size=14)).pack(pady=(15, 10))
        
        port_select_frame = ctk.CTkFrame(port_frame, fg_color="transparent")
        port_select_frame.pack(pady=10, padx=20, fill="x")
        
        self.serial_port_var = ctk.StringVar(value=self.config.serial_port or "Not Connected")
        self.serial_port_menu = ctk.CTkOptionMenu(
            port_select_frame,
            values=["Not Connected"],
            variable=self.serial_port_var,
            height=40,
            font=ctk.CTkFont(size=13)
        )
        self.serial_port_menu.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        ctk.CTkButton(port_select_frame, text="🔍 Refresh", width=100, height=40,
                     command=self.refresh_serial_ports).pack(side="left")
        
        # Baudrate
        baud_frame = ctk.CTkFrame(left_frame, fg_color="#2b2b2b")
        baud_frame.pack(pady=10, padx=20, fill="x")
        
        ctk.CTkLabel(baud_frame, text="Baudrate:",
                    font=ctk.CTkFont(size=14)).pack(pady=(15, 10))
        
        self.baudrate_var = ctk.StringVar(value=str(self.config.serial_baudrate))
        ctk.CTkSegmentedButton(baud_frame, values=["9600", "115200"],
                              variable=self.baudrate_var, height=40).pack(
                                  pady=10, padx=20, fill="x")
        
        # Status
        status_frame = ctk.CTkFrame(left_frame, fg_color="#1a1a1a", corner_radius=10)
        status_frame.pack(pady=20, padx=20, fill="x")
        
        self.serial_status_label = ctk.CTkLabel(
            status_frame, text="⚫ Disconnected",
            font=ctk.CTkFont(size=16, weight="bold"), text_color="gray"
        )
        self.serial_status_label.pack(pady=20)
        
        # Connect/Disconnect buttons
        btn_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        btn_frame.pack(pady=10, padx=20, fill="x")
        
        self.serial_connect_btn = ctk.CTkButton(
            btn_frame, text="🔌 Connect", command=self.connect_serial,
            fg_color="#27ae60", hover_color="#229954", height=50,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.serial_connect_btn.pack(side="left", expand=True, fill="x", padx=(0, 10))
        
        self.serial_disconnect_btn = ctk.CTkButton(
            btn_frame, text="⛔ Disconnect", command=self.disconnect_serial,
            fg_color="#e74c3c", hover_color="#c0392b", height=50,
            state="disabled", font=ctk.CTkFont(size=14, weight="bold")
        )
        self.serial_disconnect_btn.pack(side="left", expand=True, fill="x", padx=(10, 0))
        
        # Test buttons
        test_frame = ctk.CTkFrame(left_frame, fg_color="#2b2b2b")
        test_frame.pack(pady=20, padx=20, fill="x")
        
        ctk.CTkLabel(test_frame, text="Test Commands:",
                    font=ctk.CTkFont(size=14)).pack(pady=(15, 10))
        
        test_btn_frame = ctk.CTkFrame(test_frame, fg_color="transparent")
        test_btn_frame.pack(pady=10, padx=20, fill="x")
        
        ctk.CTkButton(test_btn_frame, text="✅ Test ACCEPT", 
                     command=lambda: self.test_serial("ACCEPT"),
                     fg_color="#2ecc71", height=40).pack(
                         side="left", expand=True, fill="x", padx=(0, 10))
        
        ctk.CTkButton(test_btn_frame, text="❌ Test REJECT",
                     command=lambda: self.test_serial("REJECT"),
                     fg_color="#e74c3c", height=40).pack(
                         side="left", expand=True, fill="x", padx=(10, 0))
        
        ctk.CTkLabel(test_frame, text="(Kirim sinyal ke Arduino untuk test)",
                    font=ctk.CTkFont(size=10), text_color="gray").pack(pady=(5, 15))
        
        # Right - Info & Code
        right_frame = ctk.CTkFrame(tab)
        right_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        ctk.CTkLabel(right_frame, text="Arduino Code Example",
                    font=ctk.CTkFont(size=18, weight="bold")).pack(pady=20)
        
        code_text = """
// Arduino Code untuk Egg Sorter
const int servoAcceptPin = 9;
const int servoRejectPin = 10;

void setup() {
  Serial.begin(9600);  // Atau 115200
  pinMode(servoAcceptPin, OUTPUT);
  pinMode(servoRejectPin, OUTPUT);
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    
    if (cmd == 'A') {
      // ACCEPT - Jalur accept
      digitalWrite(servoAcceptPin, HIGH);
      delay(500);
      digitalWrite(servoAcceptPin, LOW);
    }
    else if (cmd == 'R') {
      // REJECT - Jalur reject
      digitalWrite(servoRejectPin, HIGH);
      delay(500);
      digitalWrite(servoRejectPin, LOW);
    }
  }
}
        """
        
        code_box = ctk.CTkTextbox(right_frame, font=ctk.CTkFont(family="Courier", size=11))
        code_box.pack(pady=10, padx=20, fill="both", expand=True)
        code_box.insert("1.0", code_text)
        code_box.configure(state="disabled")
        
        threading.Thread(target=self.initial_serial_detection, daemon=True).start()
    
    def setup_model_settings_tab(self):
        """Setup model and device settings tab"""
        tab = self.main_tabview.add(f"{EMOJI['brain']} Model")
        tab.grid_columnconfigure((0,1), weight=1)
        
        # Left - Device
        left_frame = ctk.CTkFrame(tab)
        left_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        ctk.CTkLabel(left_frame, text="Device Settings",
                    font=ctk.CTkFont(size=18, weight="bold")).pack(pady=20)
        
        # Device selection
        device_frame = ctk.CTkFrame(left_frame, fg_color="#2b2b2b")
        device_frame.pack(pady=10, padx=20, fill="x")
        
        ctk.CTkLabel(device_frame, text="Processing Device:",
                    font=ctk.CTkFont(size=14)).pack(pady=(15, 10))
        
        device_options = ["cpu"]
        if self.model_manager.cuda_available:
            device_options.append("cuda")
        
        self.device_var = ctk.StringVar(value=self.config.device)
        ctk.CTkSegmentedButton(device_frame, values=device_options,
                              variable=self.device_var, command=self.change_device,
                              height=50).pack(pady=10, padx=20, fill="x")
        
        # Device info
        device_info = "CPU Mode"
        if self.model_manager.cuda_available:
            import torch
            device_info = f"GPU: {torch.cuda.get_device_name(0)}"
        
        self.device_info_label = ctk.CTkLabel(device_frame, text=f"ℹ️ {device_info}",
                                             font=ctk.CTkFont(size=12), text_color="gray")
        self.device_info_label.pack(pady=(5, 15))
        
        # Other settings
        settings_frame = ctk.CTkFrame(left_frame, fg_color="#2b2b2b")
        settings_frame.pack(pady=10, padx=20, fill="x")
        
        ctk.CTkLabel(settings_frame, text="Other Settings:",
                    font=ctk.CTkFont(size=14)).pack(pady=(15, 10))
        
        self.log_checkbox = ctk.CTkCheckBox(settings_frame, text="Enable Logging",
                                           command=self.toggle_log,
                                           font=ctk.CTkFont(size=13))
        if self.config.aktifkan_log:
            self.log_checkbox.select()
        self.log_checkbox.pack(pady=10, padx=20, anchor="w")
        
        self.alert_checkbox = ctk.CTkCheckBox(settings_frame, text="Enable Alerts",
                                             command=self.toggle_alert,
                                             font=ctk.CTkFont(size=13))
        if self.config.alert_enabled:
            self.alert_checkbox.select()
        self.alert_checkbox.pack(pady=10, padx=20, anchor="w")
        
        self.sound_checkbox = ctk.CTkCheckBox(settings_frame, text="Enable Sound",
                                             command=self.toggle_sound,
                                             font=ctk.CTkFont(size=13))
        if self.config.alert_sound_enabled:
            self.sound_checkbox.select()
        self.sound_checkbox.pack(pady=(10, 15), padx=20, anchor="w")
        
        # Save/Load config
        config_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        config_frame.pack(pady=20, padx=20, fill="x")
        
        ctk.CTkButton(config_frame, text=f"{EMOJI['save']} Save Config",
                     command=self.save_config_ui, fg_color="#3498db",
                     hover_color="#2980b9", height=45,
                     font=ctk.CTkFont(size=13, weight="bold")
                     ).pack(side="left", expand=True, fill="x", padx=(0, 10))
        
        ctk.CTkButton(config_frame, text=f"{EMOJI['folder']} Load Config",
                     command=self.load_config_ui, fg_color="#9b59b6",
                     hover_color="#8e44ad", height=45,
                     font=ctk.CTkFont(size=13, weight="bold")
                     ).pack(side="left", expand=True, fill="x", padx=(10, 0))
        
        # Right - Model Info
        right_frame = ctk.CTkFrame(tab)
        right_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        ctk.CTkLabel(right_frame, text="Model Information",
                    font=ctk.CTkFont(size=18, weight="bold")).pack(pady=20)
        
        info_frame = ctk.CTkFrame(right_frame, fg_color="#2b2b2b")
        info_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        model_info = f"""
🧠 Model: YOLOv8
📍 Path: {MODEL_PATH}
📊 Classes: {len(self.model_manager.model.names)}

Detected Classes:
{chr(10).join([f"  • {i}: {name}" for i, name in self.model_manager.model.names.items()])}

💻 Device Information:
"""
        
        if self.model_manager.cuda_available:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            model_info += f"""
  ✅ CUDA Available
  🎮 GPU: {gpu_name}
  💾 Memory: {gpu_memory:.1f} GB
  
⚡ Performance Mode: HIGH
            """
        else:
            model_info += """
  ⚠️ CUDA Not Available
  💻 Using: CPU Only
  
⚙️ Performance Mode: STANDARD
            """
        
        ctk.CTkLabel(info_frame, text=model_info, justify="left",
                    font=ctk.CTkFont(size=11)).pack(pady=20, padx=20, anchor="w")
    
    def setup_log_tab(self):
        """Setup log tab"""
        tab = self.main_tabview.add(f"{EMOJI['folder']} Logs")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)
        
        # Control buttons
        control_frame = ctk.CTkFrame(tab)
        control_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        control_frame.grid_columnconfigure((0,1,2), weight=1)
        
        ctk.CTkButton(control_frame, text=f"{EMOJI['refresh']} Refresh Log (F5)",
                     command=self.refresh_log, height=45,
                     font=ctk.CTkFont(size=13)).grid(
                         row=0, column=0, padx=10, pady=10, sticky="ew")
        
        ctk.CTkButton(control_frame, text=f"{EMOJI['down']} Export CSV",
                     command=self.export_log, fg_color="#27ae60",
                     height=45, font=ctk.CTkFont(size=13)).grid(
                         row=0, column=1, padx=10, pady=10, sticky="ew")
        
        ctk.CTkButton(control_frame, text="🗑️ Clear Logs",
                     command=self.clear_logs, fg_color="#e74c3c",
                     height=45, font=ctk.CTkFont(size=13)).grid(
                         row=0, column=2, padx=10, pady=10, sticky="ew")
        
        # Log textbox
        self.log_textbox = ctk.CTkTextbox(tab, wrap="none",
                                         font=ctk.CTkFont(family="Courier", size=10))
        self.log_textbox.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        
        self.refresh_log()
    
    def setup_chart_tab(self):
        """Setup chart tab"""
        tab = self.main_tabview.add(f"{EMOJI['chart']} Chart")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 6), dpi=100, facecolor='#1e1e1e')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#2b2b2b')
        self.ax.set_title('Real-time Detection Statistics', 
                         color='white', fontsize=16, weight='bold')
        self.ax.set_xlabel('Detection Count', color='white', fontsize=12)
        self.ax.set_ylabel('Total Count', color='white', fontsize=12)
        self.ax.tick_params(colors='white', labelsize=10)
        for spine in self.ax.spines.values():
            spine.set_color('white')
        self.ax.grid(True, alpha=0.2, color='gray', linestyle='--')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=tab)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=20)
        
        self.line_accept, = self.ax.plot([], [], 'g-', label='ACCEPT', 
                                         linewidth=3, marker='o', markersize=4)
        self.line_reject, = self.ax.plot([], [], 'r-', label='REJECT',
                                         linewidth=3, marker='o', markersize=4)
        self.ax.legend(loc='upper left', facecolor='#2b2b2b', edgecolor='white',
                      labelcolor='white', fontsize=12, framealpha=0.9)
        
        if self.chart_data["timestamps"]:
            self.update_live_chart()
    
    # ==================== EVENT HANDLERS ====================
    def update_conf(self, value):
        self.config.min_conf = value
        self.conf_value.configure(text=f"{value:.2f}")
    
    def update_zone_x(self, value):
        self.config.detection_zone_x = int(value)
        self.zone_x_value.configure(text=f"{int(value)} px")
    
    def update_tolerance(self, value):
        self.config.detection_zone_tolerance = int(value)
        self.tolerance_value.configure(text=f"±{int(value)} px")
    
    def update_cooldown(self, value):
        self.config.decision_cooldown = value
        self.cooldown_value.configure(text=f"{value:.1f} s")
    
    def toggle_log(self):
        self.config.aktifkan_log = self.log_checkbox.get()
    
    def toggle_alert(self):
        self.config.alert_enabled = self.alert_checkbox.get()
    
    def toggle_sound(self):
        self.config.alert_sound_enabled = self.sound_checkbox.get()
    
    def change_camera_source(self, value):
        if value == "Webcam":
            self.camera_source = "local"
            self.local_cam_frame.pack(pady=20, padx=20, fill="x")
            self.ip_cam_frame.pack_forget()
        else:
            self.camera_source = "ip"
            self.local_cam_frame.pack_forget()
            self.ip_cam_frame.pack(pady=20, padx=20, fill="x")
    
    def change_device(self, value):
        try:
            old_device = self.config.device
            self.config.device = value
            
            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model_manager.predict(test_img, 0.5, value)
            
            if value == "cuda":
                import torch
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.device_info_label.configure(
                    text=f"🚀 {gpu_name} ({gpu_memory:.1f}GB)", 
                    text_color="#2ecc71")
                messagebox.showinfo("Device Changed", 
                    f"Model sekarang menggunakan GPU:\n{gpu_name}")
            else:
                self.device_info_label.configure(text="💻 Using: CPU", text_color="gray")
                messagebox.showinfo("Device Changed", "Model sekarang menggunakan CPU")
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal mengubah device:\n{str(e)}")
            self.device_var.set(old_device)
            self.config.device = old_device
    
    def save_config_ui(self):
        self.config.ip_camera = self.ip_entry.get()
        if self.config.save():
            messagebox.showinfo("Success", "✅ Configuration saved successfully!")
        else:
            messagebox.showerror("Error", "❌ Failed to save configuration!")
    
    def load_config_ui(self):
        self.config.load()
        self.ip_entry.delete(0, "end")
        self.ip_entry.insert(0, self.config.ip_camera)
        self.conf_slider.set(self.config.min_conf)
        self.conf_value.configure(text=f"{self.config.min_conf:.2f}")
        self.zone_x_slider.set(self.config.detection_zone_x)
        self.zone_x_value.configure(text=f"{self.config.detection_zone_x} px")
        self.tolerance_slider.set(self.config.detection_zone_tolerance)
        self.tolerance_value.configure(text=f"±{self.config.detection_zone_tolerance} px")
        self.cooldown_slider.set(self.config.decision_cooldown)
        self.cooldown_value.configure(text=f"{self.config.decision_cooldown:.1f} s")
        
        self.log_checkbox.select() if self.config.aktifkan_log else self.log_checkbox.deselect()
        self.alert_checkbox.select() if self.config.alert_enabled else self.alert_checkbox.deselect()
        self.sound_checkbox.select() if self.config.alert_sound_enabled else self.sound_checkbox.deselect()
        self.device_var.set(self.config.device)
        
        messagebox.showinfo("Success", "✅ Configuration loaded successfully!")
    
    # ==================== SERIAL HANDLERS ====================
    def initial_serial_detection(self):
        time.sleep(0.5)
        self.refresh_serial_ports()
    
    def refresh_serial_ports(self):
        ports = self.serial_manager.list_ports()
        if ports:
            port_list = [f"{port[0]} - {port[1]}" for port in ports]
            self.serial_port_menu.configure(values=port_list)
            if port_list:
                self.serial_port_var.set(port_list[0])
        else:
            self.serial_port_menu.configure(values=["No Ports Found"])
            self.serial_port_var.set("No Ports Found")
    
    def connect_serial(self):
        selected = self.serial_port_var.get()
        if "No Ports" in selected or "Not Connected" in selected:
            messagebox.showwarning("Warning", "Pilih port serial terlebih dahulu!")
            return
        
        port = selected.split(" - ")[0]
        baudrate = int(self.baudrate_var.get())
        
        if self.serial_manager.connect(port, baudrate):
            self.config.serial_port = port
            self.config.serial_baudrate = baudrate
            self.serial_status_label.configure(text="🟢 Connected", text_color="#2ecc71")
            self.top_serial_label.configure(text=f"{EMOJI['usb']} Serial\n🟢 On", text_color="#2ecc71")
            self.serial_connect_btn.configure(state="disabled")
            self.serial_disconnect_btn.configure(state="normal")
            messagebox.showinfo("Success", f"✅ Connected to {port}")
        else:
            messagebox.showerror("Error", f"❌ Failed to connect to {port}")
    
    def disconnect_serial(self):
        if self.serial_manager.disconnect():
            self.serial_status_label.configure(text="⚫ Disconnected", text_color="gray")
            self.top_serial_label.configure(text=f"{EMOJI['usb']} Serial\n⚫ Off", text_color="gray")
            self.serial_connect_btn.configure(state="normal")
            self.serial_disconnect_btn.configure(state="disabled")
            messagebox.showinfo("Info", "Serial disconnected")
    
    def test_serial(self, decision):
        if not self.serial_manager.connected:
            messagebox.showwarning("Warning", "Serial belum terhubung!")
            return
        
        if self.serial_manager.send_decision(decision):
            messagebox.showinfo("Test Success", f"✅ Sent: {decision}")
        else:
            messagebox.showerror("Test Failed", "❌ Failed to send command")
    
    # ==================== CAMERA MANAGEMENT ====================
    def initial_camera_detection(self):
        self.available_cameras = CameraManager.detect_cameras()
        self.after(0, self.update_camera_dropdown)
    
    def refresh_cameras(self):
        self.detect_cam_btn.configure(state="disabled", text="⏳")
        self.local_cam_var.set("Detecting...")
        
        def detect_thread():
            self.available_cameras = CameraManager.detect_cameras()
            self.after(0, self.update_camera_dropdown)
            self.after(0, lambda: self.detect_cam_btn.configure(state="normal", text=f"{EMOJI['search']}"))
            self.after(0, lambda: messagebox.showinfo("Info", 
                f"✅ Ditemukan {len(self.available_cameras)} kamera"))
        
        threading.Thread(target=detect_thread, daemon=True).start()
    
    def update_camera_dropdown(self):
        camera_names = [cam["name"] for cam in self.available_cameras] if self.available_cameras else ["No Camera Detected"]
        self.local_cam_menu.configure(values=camera_names)
        if camera_names:
            self.local_cam_var.set(camera_names[0])
    
    def get_selected_camera_index(self):
        selected_name = self.local_cam_var.get()
        for cam in self.available_cameras:
            if cam["name"] == selected_name:
                return cam["index"]
        return 0
    
    # ==================== DETECTION LOGIC ====================
    def ambil_keputusan(self, labels):
        """Determine decision: ACCEPT or REJECT"""
        reject_classes = ["crack", "dirty"]
        labels_lower = [label.lower() for label in labels]
        return "REJECT" if any(lbl in reject_classes for lbl in labels_lower) else "ACCEPT"
    
    def is_in_detection_zone(self, centroid_x):
        """Check if object is in detection zone"""
        zone_start = self.config.detection_zone_x - self.config.detection_zone_tolerance
        zone_end = self.config.detection_zone_x + self.config.detection_zone_tolerance
        return zone_start <= centroid_x <= zone_end
    
    def should_make_decision(self, object_id, current_time):
        """Check if we should make a decision for this object"""
        if object_id in self.processed_objects:
            return False, None
        
        if current_time - self.last_decision_time < self.config.decision_cooldown:
            return False, None
        
        return True, None
    
    def update_counts(self, keputusan):
        with self.stats_lock:
            if keputusan == "REJECT":
                self.reject_count += 1
            else:
                self.accept_count += 1
    
    def update_stats(self):
        with self.stats_lock:
            accept, reject, fps, inference = self.accept_count, self.reject_count, self.fps, self.inference_time
        
        tracked = len(self.tracker.objects)
        
        self.top_accept_label.configure(text=f"{EMOJI['check']} ACCEPT\n{accept}")
        self.top_reject_label.configure(text=f"{EMOJI['cross']} REJECT\n{reject}")
        self.top_perf_label.configure(text=f"{EMOJI['lightning']} FPS\n{fps:.1f}")
        self.top_tracked_label.configure(text=f"{EMOJI['target']} Tracked\n{tracked}")
    
    def calculate_fps(self):
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        if self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            self.fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
    
    def play_alert_sound(self):
        try:
            system = platform.system()
            if system == "Windows":
                import winsound
                threading.Thread(target=lambda: winsound.Beep(1000, 200), daemon=True).start()
            elif system == "Darwin":
                os.system('afplay /System/Library/Sounds/Ping.aiff &')
            else:
                print('\a')
        except Exception as e:
            print(f"Could not play sound: {e}")
    
    def save_rejected_image(self, image_array, keputusan, object_id):
        if keputusan == "REJECT":
            try:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"REJECT_ID{object_id}_{timestamp}.jpg"
                filepath = os.path.join(REJECTED_IMAGES_FOLDER, filename)
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, image_bgr)
                print(f"✅ Saved rejected image: {filename}")
                return filepath
            except Exception as e:
                print(f"Error saving rejected image: {e}")
        return None
    
    # ==================== IMAGE UPLOAD ====================
    def upload_image(self):
        try:
            file_path = filedialog.askopenfilename(
                title="Pilih Gambar",
                filetypes=[("Image files", "*.jpg *.jpeg *.png")]
            )
            if not file_path:
                return
            
            image = Image.open(file_path).convert("RGB")
            
            start_time = time.time()
            results = self.model_manager.predict(np.array(image), self.config.min_conf, self.config.device)
            self.inference_time = (time.time() - start_time) * 1000
            
            boxes = results[0].boxes
            filtered_boxes = boxes[boxes.conf > self.config.min_conf]
            labels = [self.model_manager.model.names[int(cls)] for cls in filtered_boxes.cls]
            
            result_img = results[0].plot()
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            
            keputusan = self.ambil_keputusan(labels) if labels else "ACCEPT"
            
            self.display_image(result_img)
            color = "#2ecc71" if keputusan == "ACCEPT" else "#e74c3c"
            self.keputusan_label.configure(text=f"{EMOJI['brain']} Keputusan: {keputusan}", 
                                          text_color=color)
            
            self.update_counts(keputusan)
            self.update_stats()
            
            if keputusan == "REJECT" and self.config.alert_enabled:
                self.play_alert_sound()
                messagebox.showwarning("REJECT Detected!", 
                    f"Telur ditolak!\nAlasan: {', '.join(labels)}")
            
            self.chart_data["timestamps"].append(datetime.datetime.now())
            with self.stats_lock:
                self.chart_data["accept"].append(self.accept_count)
                self.chart_data["reject"].append(self.reject_count)
            self.update_live_chart()
            
            saved_path = self.save_rejected_image(result_img, keputusan, "upload")
            
            if self.config.aktifkan_log:
                log_entry = {
                    "waktu": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "sumber": "Upload Gambar",
                    "kelas_terdeteksi": ", ".join(labels) if labels else "Tidak ada",
                    "keputusan": keputusan,
                    "confidence": f"{max([float(c) for c in filtered_boxes.conf], default=0):.2f}",
                    "inference_time_ms": f"{self.inference_time:.1f}",
                    "saved_path": saved_path if saved_path else "N/A"
                }
                self.log_manager.add_entry(log_entry)
                self.log_manager.save_append([log_entry])
                self.refresh_log()
            
            if keputusan == "ACCEPT":
                messagebox.showinfo("Sukses", f"✅ Deteksi selesai!\nKeputusan: {keputusan}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan:\n{str(e)}")
    
    # ==================== CAMERA FEED ====================
    def start_camera(self):
        try:
            if self.camera_running:
                messagebox.showwarning("Peringatan", "Kamera sudah berjalan!")
                return
            
            if self.camera_source == "ip":
                self.config.ip_camera = self.ip_entry.get()
                if not self.config.ip_camera or not CameraManager.validate_ip(self.config.ip_camera):
                    messagebox.showerror("Error", "IP kamera tidak valid!")
                    return
            else:
                self.local_camera_index = self.get_selected_camera_index()
            
            self.tracker = CentroidTracker(maxDisappeared=30, maxDistance=80)
            self.processed_objects = {}
            self.last_decision_time = 0
            
            self.stop_camera = False
            self.camera_running = True
            self.start_cam_btn.configure(state="disabled")
            self.stop_cam_btn.configure(state="normal")
            
            threading.Thread(target=self.camera_loop, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memulai kamera:\n{str(e)}")
    
    def camera_loop(self):
        cap = None
        try:
            cap = self.open_camera()
            if not cap:
                return
            
            frame_count = 0
            frame_delay = 1.0 / TARGET_FPS
            
            while not self.stop_camera:
                frame_start = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.5)
                    continue
                
                frame_count += 1
                self.calculate_fps()
                frame = cv2.resize(frame, (640, 480))
                
                self.process_frame_with_tracking(frame)
                
                if frame_count % 30 == 0:
                    print(f"Frame {frame_count}: FPS {self.fps:.1f}, Tracked: {len(self.tracker.objects)}")
                
                elapsed = time.time() - frame_start
                time.sleep(max(0, frame_delay - elapsed))
            
        except Exception as e:
            print(f"Error in camera_loop: {e}")
            self.after(0, lambda: messagebox.showerror("Error", f"Error kamera:\n{str(e)}"))
        
        finally:
            if cap:
                cap.release()
            if self.config.device == "cuda":
                try:
                    import torch
                    torch.cuda.empty_cache()
                except:
                    pass
            gc.collect()
            self.camera_running = False
            self.after(0, lambda: self.start_cam_btn.configure(state="normal"))
            self.after(0, lambda: self.stop_cam_btn.configure(state="disabled"))
    
    def open_camera(self):
        """Open camera with retry logic"""
        for retry in range(3):
            if self.camera_source == "ip":
                url = f"http://{self.config.ip_camera}:4747/video"
                cap = cv2.VideoCapture(url)
                time.sleep(2)
            else:
                cap = cv2.VideoCapture(self.local_camera_index)
                time.sleep(1)
            
            if cap.isOpened() and cap.read()[0]:
                print(f"✅ Camera opened successfully")
                return cap
            
            cap.release()
            if retry < 2:
                time.sleep(1)
        
        self.after(0, lambda: messagebox.showerror("Error", "Gagal membuka kamera!"))
        return None
    
    def process_frame_with_tracking(self, frame):
        """Process frame dengan object tracking"""
        try:
            start_time = time.time()
            results = self.model_manager.predict(frame, self.config.min_conf, self.config.device)
            self.inference_time = (time.time() - start_time) * 1000
            
            if self.config.device == "cuda":
                import torch
                torch.cuda.synchronize()
            
            boxes = results[0].boxes
            filtered_boxes = boxes[boxes.conf > self.config.min_conf]
            
            rects = []
            labels_dict = {}
            
            for idx, box in enumerate(filtered_boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                rects.append((x1, y1, x2, y2))
                
                cls_id = int(filtered_boxes.cls[idx])
                label = self.model_manager.model.names[cls_id]
                conf = float(filtered_boxes.conf[idx])
                labels_dict[idx] = {"label": label, "conf": conf}
            
            objects = self.tracker.update(rects)
            
            frame_result = results[0].plot()
            current_time = time.time()
            
            decision_made = False
            decision_info = None
            
            for object_id, centroid in objects.items():
                cx, cy = centroid
                
                cv2.circle(frame_result, (cx, cy), 4, (0, 255, 0), -1)
                cv2.putText(frame_result, f"ID:{object_id}", (cx - 10, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                if self.is_in_detection_zone(cx):
                    cv2.circle(frame_result, (cx, cy), 8, (0, 255, 255), 2)
                    
                    should_decide, _ = self.should_make_decision(object_id, current_time)
                    
                    if should_decide and len(labels_dict) > 0:
                        object_labels = [labels_dict[i]["label"] for i in range(len(rects))]
                        keputusan = self.ambil_keputusan(object_labels)
                        
                        self.processed_objects[object_id] = {
                            "decision": keputusan,
                            "time": current_time,
                            "labels": object_labels
                        }
                        self.last_decision_time = current_time
                        
                        self.update_counts(keputusan)
                        
                        if self.serial_manager.connected:
                            self.serial_manager.send_decision(keputusan)
                        
                        decision_made = True
                        decision_info = {
                            "keputusan": keputusan,
                            "labels": object_labels,
                            "object_id": object_id,
                            "centroid": (cx, cy)
                        }
                        
                        if keputusan == "REJECT" and self.config.alert_enabled:
                            self.play_alert_sound()
                        
                        if self.config.aktifkan_log:
                            self.log_camera_detection_with_tracking(
                                datetime.datetime.now(),
                                object_labels,
                                keputusan,
                                object_id,
                                max([labels_dict[i]["conf"] for i in range(len(labels_dict))])
                            )
                        
                        print(f"🎯 Decision made for ID {object_id}: {keputusan} at zone X={cx}")
            
            zone_x = self.config.detection_zone_x
            cv2.line(frame_result, (zone_x, 0), (zone_x, frame_result.shape[0]), (0, 255, 255), 2)
            cv2.putText(frame_result, "DETECTION ZONE", (zone_x - 70, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            zone_start = zone_x - self.config.detection_zone_tolerance
            zone_end = zone_x + self.config.detection_zone_tolerance
            cv2.line(frame_result, (zone_start, 0), (zone_start, frame_result.shape[0]), (255, 255, 0), 1)
            cv2.line(frame_result, (zone_end, 0), (zone_end, frame_result.shape[0]), (255, 255, 0), 1)
            
            frame_rgb = cv2.cvtColor(frame_result, cv2.COLOR_BGR2RGB)
            
            self.display_image(frame_rgb)
            
            if decision_made and decision_info:
                self.after(0, self._update_keputusan_ui, decision_info["keputusan"])
                
                if decision_info["keputusan"] == "REJECT":
                    self.save_rejected_image(frame_rgb, decision_info["keputusan"], 
                                           decision_info["object_id"])
            
            self.after(0, self.update_stats)
            
            self.cleanup_old_objects(current_time)
        
        except Exception as e:
            print(f"Error processing frame: {e}")
    
    def cleanup_old_objects(self, current_time):
        """Remove old processed objects from memory"""
        to_remove = []
        for obj_id, data in self.processed_objects.items():
            if current_time - data["time"] > 10:
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self.processed_objects[obj_id]
    
    def log_camera_detection_with_tracking(self, now, labels, keputusan, object_id, confidence):
        """Log camera detection with tracking info"""
        log_entry = {
            "waktu": now.strftime("%Y-%m-%d %H:%M:%S"),
            "sumber": "Kamera (Tracking)",
            "object_id": f"ID-{object_id}",
            "kelas_terdeteksi": ", ".join(labels),
            "keputusan": keputusan,
            "confidence": f"{confidence:.2f}",
            "inference_time_ms": f"{self.inference_time:.1f}",
            "fps": f"{self.fps:.1f}",
            "serial_sent": "Yes" if self.serial_manager.connected else "No"
        }
        self.log_manager.add_entry(log_entry)
        self.log_manager.save_append([log_entry])
        
        self.chart_data["timestamps"].append(now)
        with self.stats_lock:
            self.chart_data["accept"].append(self.accept_count)
            self.chart_data["reject"].append(self.reject_count)
        self.after(0, self.update_live_chart)
    
    def stop_camera_feed(self):
        if not self.camera_running:
            return
        self.stop_camera = True
        self.stop_cam_btn.configure(state="disabled")
        messagebox.showinfo("Info", "Kamera sedang dihentikan...")
    
    def _update_keputusan_ui(self, keputusan):
        color = "#2ecc71" if keputusan == "ACCEPT" else "#e74c3c"
        self.keputusan_label.configure(text=f"{EMOJI['brain']} Keputusan: {keputusan}", 
                                      text_color=color)
    
    # ==================== UI UPDATES ====================
    def display_image(self, img_array):
        try:
            h, w = img_array.shape[:2]
            scale = min(900/w, 600/h)
            new_w, new_h = int(w*scale), int(h*scale)
            img_resized = cv2.resize(img_array, (new_w, new_h))
            img_pil = Image.fromarray(img_resized)
            img_tk = ImageTk.PhotoImage(img_pil)
            self.after(0, self._update_image_label, img_tk)
        except Exception as e:
            print(f"Error displaying image: {e}")
    
    def _update_image_label(self, img_tk):
        if hasattr(self.image_label, 'image'):
            try:
                del self.image_label.image
            except:
                pass
        self.image_label.configure(image=img_tk, text="")
        self.image_label.image = img_tk
    
    def update_live_chart(self):
        """Update live chart"""
        try:
            if not self.chart_data["timestamps"]:
                return
            
            if len(self.chart_data["timestamps"]) > self.max_chart_points:
                for key in ["timestamps", "accept", "reject"]:
                    self.chart_data[key] = self.chart_data[key][-self.max_chart_points:]
            
            x_data = list(range(len(self.chart_data["timestamps"])))
            self.line_accept.set_data(x_data, self.chart_data["accept"])
            self.line_reject.set_data(x_data, self.chart_data["reject"])
            
            if x_data:
                self.ax.set_xlim(0, max(x_data) + 1)
                max_y = max(max(self.chart_data["accept"] + self.chart_data["reject"], default=1), 5)
                self.ax.set_ylim(0, max_y + 2)
            
            self.canvas.draw()
        except Exception as e:
            print(f"Error updating chart: {e}")
    
    # ==================== LOG MANAGEMENT ====================
    def refresh_log(self):
        self.log_textbox.delete("1.0", "end")
        if not self.log_manager.log_deteksi:
            self.log_textbox.insert("1.0", "Belum ada log deteksi.\n")
            return
        df = pd.DataFrame(self.log_manager.log_deteksi)
        self.log_textbox.insert("1.0", df.to_string(index=False))
    
    def export_log(self):
        if not self.log_manager.log_deteksi:
            messagebox.showinfo("Info", "Tidak ada log untuk diekspor.")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                 filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                df = pd.DataFrame(self.log_manager.log_deteksi)
                df.to_csv(file_path, index=False)
                
                total = len(df)
                accepts = sum(1 for log in self.log_manager.log_deteksi if log.get("keputusan") == "ACCEPT")
                rejects = total - accepts
                
                messagebox.showinfo("Sukses",
                    f"✅ Log berhasil diekspor!\n\nTotal: {total}\n"
                    f"Accept: {accepts} ({accepts/total*100:.1f}%)\n"
                    f"Reject: {rejects} ({rejects/total*100:.1f}%)")
            except Exception as e:
                messagebox.showerror("Error", f"Gagal export log:\n{str(e)}")
    
    def clear_logs(self):
        if not messagebox.askyesno("Konfirmasi",
            "Yakin ingin menghapus semua log?\n\nData tidak dapat dikembalikan!"):
            return
        
        self.log_manager.reset()
        self.refresh_log()
        messagebox.showinfo("Sukses", "✅ Log berhasil dihapus!")
    
    def reset_data(self):
        if not messagebox.askyesno("Konfirmasi",
            "Yakin ingin mereset semua data?\n\nIni akan menghapus semua log dan statistik!"):
            return
        
        with self.stats_lock:
            self.reject_count = 0
            self.accept_count = 0
        
        self.log_manager.reset()
        self.last_log_time = datetime.datetime.min
        self.fps = 0
        self.inference_time = 0
        self.frame_times = []
        self.chart_data = {"timestamps": [], "accept": [], "reject": []}
        self.processed_objects = {}
        self.last_decision_time = 0
        
        self.update_live_chart()
        self.update_stats()
        self.refresh_log()
        self.keputusan_label.configure(text="")
        
        if hasattr(self.image_label, 'image'):
            try:
                del self.image_label.image
            except:
                pass
        self.image_label.configure(image=None, text="Tidak ada gambar")
        
        gc.collect()
        messagebox.showinfo("Sukses", "✅ Data berhasil direset!")


if __name__ == "__main__":
    app = EggSorterApp()
    app.mainloop()