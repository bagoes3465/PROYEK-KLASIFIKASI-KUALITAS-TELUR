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

#
# Set TCL_LIBRARY path
tcl_dir = os.path.join(sys.base_prefix, 'tcl', 'tcl8.6')
tk_dir = os.path.join(sys.base_prefix, 'tcl', 'tk8.6')

os.environ['TCL_LIBRARY'] = tcl_dir
os.environ['TK_LIBRARY'] = tk_dir
#

# Konfigurasi CustomTkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

# Constants
LOG_FILE = "log_deteksi.csv"
CONFIG_FILE = "app_config.json"
REJECTED_IMAGES_FOLDER = "rejected_images"
MODEL_PATH = Path("models/best.pt")  # Relative path
TARGET_FPS = 30  # FPS limiter

# Emoji constants (untuk avoid encoding issues)
EMOJI_EGG = "\U0001F95A"
EMOJI_SETTINGS = "\u2699\uFE0F"
EMOJI_CHART = "\U0001F4CA"
EMOJI_CHECK = "\u2705"
EMOJI_CROSS = "\u274C"
EMOJI_CAMERA = "\U0001F4F9"
EMOJI_BRAIN = "\U0001F9E0"
EMOJI_LIGHTNING = "\u26A1"
EMOJI_CLOCK = "\u23F1\uFE0F"
EMOJI_BELL = "\U0001F514"
EMOJI_SAVE = "\U0001F4BE"
EMOJI_FOLDER = "\U0001F4C2"
EMOJI_REFRESH = "\U0001F504"
EMOJI_UPLOAD = "\U0001F4E4"
EMOJI_PLAY = "\u25B6\uFE0F"
EMOJI_STOP = "\U0001F6D1"
EMOJI_SEARCH = "\U0001F50D"
EMOJI_DOWN = "\u2B07\uFE0F"

# Create folders if not exist
os.makedirs(REJECTED_IMAGES_FOLDER, exist_ok=True)
os.makedirs("models", exist_ok=True)


class EggSorterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Konfigurasi Window
        self.title(f"{EMOJI_EGG} Sortir Kualitas Telur Otomatis")
        self.geometry("1400x900")
        
        # Thread safety locks
        self.stats_lock = threading.Lock()
        self.log_lock = threading.Lock()
        
        # Load Model YOLO dengan error handling
        if not self.load_model():
            return
        
        # State Variables
        self.log_deteksi = []
        self.reject_count = 0
        self.accept_count = 0
        self.stop_camera = False
        self.camera_running = False
        self.last_log_time = datetime.datetime.min
        self.camera_source = "local"
        self.available_cameras = []
        self.local_camera_index = 0
        
        # Performance tracking
        self.fps = 0
        self.inference_time = 0
        self.frame_times = []
        self.last_frame_time = time.time()
        
        # Alert settings
        self.alert_enabled = True
        self.alert_sound_enabled = True
        
        # Chart data
        self.chart_data = {"timestamps": [], "accept": [], "reject": []}
        self.max_chart_points = 50
        
        # Settings
        self.min_conf = 0.3
        self.aktifkan_log = True
        self.ip_camera = "192.168.1.6"
        self.device = "cpu"
        
        # Load configuration and log
        self.load_configuration()
        self.log_deteksi = self.load_log()
        
        # Setup UI
        self.setup_ui()
        
        # Bind keyboard shortcuts
        self.bind("<Escape>", lambda e: self.stop_camera_feed() if self.camera_running else None)
        self.bind("<space>", lambda e: self.upload_image() if not self.camera_running else None)
    
    def load_model(self):
        """Load YOLO model with proper error handling"""
        try:
            # Check if model exists
            if not MODEL_PATH.exists():
                messagebox.showwarning(
                    "Model Not Found",
                    f"Model tidak ditemukan di:\n{MODEL_PATH.absolute()}\n\n"
                    "Silakan pilih file model YOLO (.pt)"
                )
                
                model_path = filedialog.askopenfilename(
                    title="Pilih File Model YOLO",
                    filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
                )
                
                if not model_path:
                    messagebox.showerror("Error", "Model tidak dipilih. Aplikasi akan ditutup.")
                    self.destroy()
                    return False
                
                model_file = Path(model_path)
            else:
                model_file = MODEL_PATH
            
            print(f"Loading model from: {model_file.absolute()}")
            self.model = YOLO(str(model_file))
            print(f"{EMOJI_CHECK} Model loaded successfully")
            
            # Display detected classes
            print(f"{EMOJI_CHECK} Detected classes: {self.model.names}")
            
            # Check CUDA availability
            import torch
            self.cuda_available = torch.cuda.is_available()
            if self.cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"{EMOJI_CHECK} CUDA available: {gpu_name} ({gpu_memory:.1f}GB)")
                print(f"{EMOJI_CHECK} CUDA version: {torch.version.cuda}")
            else:
                print(f"ℹ️ CUDA not available, using CPU only")
            
            return True
            
        except Exception as e:
            messagebox.showerror(
                "Error Loading Model",
                f"Gagal memuat model YOLO:\n{str(e)}\n\n"
                "Pastikan:\n"
                "1. File model (.pt) valid\n"
                "2. Ultralytics terinstall\n"
                "3. Versi PyTorch kompatibel"
            )
            self.destroy()
            return False
    
    def load_log(self):
        """Load log and rebuild statistics"""
        try:
            df = pd.read_csv(LOG_FILE)
            logs = df.to_dict(orient="records")
            
            # Rebuild counts dari log
            accept = sum(1 for log in logs if log.get("keputusan") == "ACCEPT")
            reject = sum(1 for log in logs if log.get("keputusan") == "REJECT")
            
            with self.stats_lock:
                self.accept_count = accept
                self.reject_count = reject
            
            # Rebuild chart data (last N points)
            temp_accept = 0
            temp_reject = 0
            
            for log in logs[-self.max_chart_points:]:
                keputusan = log.get("keputusan")
                if keputusan == "ACCEPT":
                    temp_accept += 1
                elif keputusan == "REJECT":
                    temp_reject += 1
                
                self.chart_data["timestamps"].append(log.get("waktu", ""))
                self.chart_data["accept"].append(temp_accept)
                self.chart_data["reject"].append(temp_reject)
            
            print(f"{EMOJI_CHECK} Loaded {len(logs)} log entries")
            print(f"  {EMOJI_CHECK} ACCEPT: {accept}, {EMOJI_CROSS} REJECT: {reject}")
            
            return logs
            
        except FileNotFoundError:
            print("No existing log file found")
            return []
        except Exception as e:
            print(f"Error loading log: {e}")
            return []
    
    def save_log_append(self, log_data):
        """Thread-safe log saving"""
        try:
            df_new = pd.DataFrame(log_data)
            df_new.to_csv(LOG_FILE, mode='a', index=False, 
                         header=not os.path.exists(LOG_FILE))
        except Exception as e:
            print(f"Error saving log: {e}")
    
    def load_configuration(self):
        """Load saved configuration from JSON file"""
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    self.ip_camera = config.get("ip_camera", "192.168.1.6")
                    self.min_conf = config.get("min_conf", 0.3)
                    self.aktifkan_log = config.get("aktifkan_log", True)
                    self.device = config.get("device", "cpu")
                    self.alert_enabled = config.get("alert_enabled", True)
                    self.alert_sound_enabled = config.get("alert_sound_enabled", True)
                    print(f"{EMOJI_CHECK} Configuration loaded")
        except Exception as e:
            print(f"Could not load configuration: {e}")
    
    def save_configuration(self):
        """Save current configuration to JSON file"""
        try:
            config = {
                "ip_camera": self.ip_camera,
                "min_conf": self.min_conf,
                "aktifkan_log": self.aktifkan_log,
                "device": self.device,
                "alert_enabled": self.alert_enabled,
                "alert_sound_enabled": self.alert_sound_enabled,
                "last_saved": datetime.datetime.now().isoformat()
            }
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"{EMOJI_CHECK} Configuration saved")
            return True
        except Exception as e:
            print(f"Could not save configuration: {e}")
            return False
    
    def detect_available_cameras(self):
        """Detect all available cameras on the system"""
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
                    
                    if i == 0:
                        camera_name = f"{EMOJI_CAMERA} Default Camera ({width}x{height})"
                    else:
                        camera_name = f"{EMOJI_CAMERA} Camera {i} ({width}x{height})"
                    
                    available.append({
                        "index": i,
                        "name": camera_name,
                        "backend": backend,
                        "resolution": f"{width}x{height}"
                    })
                    print(f"{EMOJI_CHECK} Found: {camera_name} [{backend}]")
                
                cap.release()
            time.sleep(0.1)
        
        if not available:
            print("⚠️ No cameras detected")
            available.append({
                "index": 0,
                "name": f"{EMOJI_CAMERA} Default Camera (Not detected)",
                "backend": "unknown",
                "resolution": "unknown"
            })
        
        return available
    
    def validate_ip(self, ip):
        """Validate IP address format"""
        pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        if not re.match(pattern, ip):
            return False
        
        try:
            parts = ip.split('.')
            return all(0 <= int(part) <= 255 for part in parts)
        except:
            return False
    
    def setup_ui(self):
        """Setup main UI"""
        # Grid layout
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # SIDEBAR
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar.grid_rowconfigure(10, weight=1)
        
        self.sidebar_label = ctk.CTkLabel(
            self.sidebar, 
            text=f"{EMOJI_SETTINGS} Pengaturan",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.sidebar_label.grid(row=0, column=0, padx=20, pady=20)
        
        # Class Info Display
        self.class_info_frame = ctk.CTkFrame(self.sidebar, fg_color="#2b2b2b")
        self.class_info_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        
        self.class_info_label = ctk.CTkLabel(
            self.class_info_frame,
            text="📋 Kelas Deteksi:",
            font=ctk.CTkFont(weight="bold", size=12)
        )
        self.class_info_label.pack(pady=(10, 5))
        
        # Accept classes
        self.accept_classes_label = ctk.CTkLabel(
            self.class_info_frame,
            text=f"{EMOJI_CHECK} ACCEPT:\nclean, yellow egg",
            font=ctk.CTkFont(size=11),
            text_color="#2ecc71"
        )
        self.accept_classes_label.pack(pady=2)
        
        # Reject classes
        self.reject_classes_label = ctk.CTkLabel(
            self.class_info_frame,
            text=f"{EMOJI_CROSS} REJECT:\ncrack, dirty",
            font=ctk.CTkFont(size=11),
            text_color="#e74c3c"
        )
        self.reject_classes_label.pack(pady=(2, 10))
        
        # Camera Source Selection
        self.camera_source_label = ctk.CTkLabel(
            self.sidebar, 
            text="Sumber Kamera:",
            font=ctk.CTkFont(weight="bold")
        )
        self.camera_source_label.grid(row=2, column=0, padx=20, pady=(10, 0))
        
        self.camera_source_var = ctk.StringVar(value="local")
        self.camera_source_menu = ctk.CTkSegmentedButton(
            self.sidebar,
            values=["Webcam", "DroidCam IP"],
            variable=self.camera_source_var,
            command=self.change_camera_source
        )
        self.camera_source_menu.grid(row=3, column=0, padx=20, pady=10)
        self.camera_source_menu.set("Webcam")
        
        # Local Camera Frame
        self.local_cam_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.local_cam_frame.grid(row=4, column=0, padx=20, pady=5, sticky="ew")
        
        self.local_cam_label = ctk.CTkLabel(self.local_cam_frame, text="Pilih Kamera:")
        self.local_cam_label.pack(anchor="w", pady=(0, 5))
        
        self.cam_select_frame = ctk.CTkFrame(self.local_cam_frame, fg_color="transparent")
        self.cam_select_frame.pack(fill="x")
        
        self.detect_cam_btn = ctk.CTkButton(
            self.cam_select_frame,
            text=f"{EMOJI_SEARCH}",
            width=40,
            command=self.refresh_cameras
        )
        self.detect_cam_btn.pack(side="left", padx=(0, 5))
        
        self.local_cam_var = ctk.StringVar(value="Detecting...")
        self.local_cam_menu = ctk.CTkOptionMenu(
            self.cam_select_frame,
            values=["Detecting..."],
            variable=self.local_cam_var,
            width=200
        )
        self.local_cam_menu.pack(side="left", fill="x", expand=True)
        
        threading.Thread(target=self.initial_camera_detection, daemon=True).start()
        
        # IP Camera Frame
        self.ip_cam_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.ip_cam_frame.grid(row=5, column=0, padx=20, pady=5, sticky="ew")
        
        self.ip_label = ctk.CTkLabel(self.ip_cam_frame, text="IP DroidCam:")
        self.ip_label.pack(anchor="w", pady=(0, 5))
        
        self.ip_entry = ctk.CTkEntry(self.ip_cam_frame, placeholder_text="192.168.1.6")
        self.ip_entry.insert(0, self.ip_camera)
        self.ip_entry.pack(fill="x")
        
        self.ip_cam_frame.grid_remove()
        
        # Device Selection
        self.device_label = ctk.CTkLabel(
            self.sidebar,
            text="Device (CPU/GPU):",
            font=ctk.CTkFont(weight="bold")
        )
        self.device_label.grid(row=6, column=0, padx=20, pady=(20, 0))
        
        device_options = ["cpu"]
        if hasattr(self, 'cuda_available') and self.cuda_available:
            device_options.append("cuda")
        
        self.device_var = ctk.StringVar(value=self.device)
        self.device_menu = ctk.CTkSegmentedButton(
            self.sidebar,
            values=device_options,
            variable=self.device_var,
            command=self.change_device
        )
        self.device_menu.grid(row=7, column=0, padx=20, pady=10)
        
        device_info = "CPU Mode"
        if hasattr(self, 'cuda_available') and self.cuda_available:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
            device_info = f"GPU: {gpu_name}"
        
        self.device_info_label = ctk.CTkLabel(
            self.sidebar,
            text=f"ℹ️ {device_info}",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.device_info_label.grid(row=8, column=0, padx=20, pady=(0, 10))
        
        # Confidence Slider
        self.conf_label = ctk.CTkLabel(self.sidebar, text="Confidence Minimum:")
        self.conf_label.grid(row=9, column=0, padx=20, pady=(10, 0))
        
        self.conf_slider = ctk.CTkSlider(
            self.sidebar,
            from_=0.0,
            to=1.0,
            command=self.update_conf
        )
        self.conf_slider.set(self.min_conf)
        self.conf_slider.grid(row=10, column=0, padx=20, pady=10)
        
        self.conf_value = ctk.CTkLabel(self.sidebar, text=f"{self.min_conf:.2f}")
        self.conf_value.grid(row=11, column=0, padx=20, pady=0)
        
        # Log Checkbox
        self.log_checkbox = ctk.CTkCheckBox(
            self.sidebar,
            text="Aktifkan Log",
            command=self.toggle_log
        )
        if self.aktifkan_log:
            self.log_checkbox.select()
        self.log_checkbox.grid(row=12, column=0, padx=20, pady=10)
        
        # Statistics Frame
        self.stats_frame = ctk.CTkFrame(self.sidebar)
        self.stats_frame.grid(row=13, column=0, padx=20, pady=20, sticky="ew")
        
        self.stats_label = ctk.CTkLabel(
            self.stats_frame,
            text=f"{EMOJI_CHART} Statistik",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.stats_label.pack(pady=10)
        
        self.accept_label = ctk.CTkLabel(
            self.stats_frame,
            text=f"{EMOJI_CHECK} ACCEPT: {self.accept_count}",
            text_color="#2ecc71"
        )
        self.accept_label.pack(pady=5)
        
        self.reject_label = ctk.CTkLabel(
            self.stats_frame,
            text=f"{EMOJI_CROSS} REJECT: {self.reject_count}",
            text_color="#e74c3c"
        )
        self.reject_label.pack(pady=5)
        
        self.perf_label = ctk.CTkLabel(
            self.stats_frame,
            text=f"{EMOJI_LIGHTNING} FPS: 0.0",
            font=ctk.CTkFont(size=11)
        )
        self.perf_label.pack(pady=5)
        
        self.inference_label = ctk.CTkLabel(
            self.stats_frame,
            text=f"{EMOJI_CLOCK} Inference: 0ms",
            font=ctk.CTkFont(size=11)
        )
        self.inference_label.pack(pady=5)
        
        # Alert Settings
        self.alert_frame = ctk.CTkFrame(self.sidebar)
        self.alert_frame.grid(row=14, column=0, padx=20, pady=10, sticky="ew")
        
        self.alert_label = ctk.CTkLabel(
            self.alert_frame,
            text=f"{EMOJI_BELL} Alert Settings",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.alert_label.pack(pady=5)
        
        self.alert_checkbox = ctk.CTkCheckBox(
            self.alert_frame,
            text="Enable Alerts",
            command=self.toggle_alert
        )
        if self.alert_enabled:
            self.alert_checkbox.select()
        self.alert_checkbox.pack(pady=2)
        
        self.sound_checkbox = ctk.CTkCheckBox(
            self.alert_frame,
            text="Enable Sound",
            command=self.toggle_sound
        )
        if self.alert_sound_enabled:
            self.sound_checkbox.select()
        self.sound_checkbox.pack(pady=2)
        
        # Config Buttons
        self.config_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.config_frame.grid(row=15, column=0, padx=20, pady=10, sticky="ew")
        
        self.save_config_btn = ctk.CTkButton(
            self.config_frame,
            text=f"{EMOJI_SAVE} Save",
            command=self.save_config_ui,
            fg_color="#3498db",
            hover_color="#2980b9",
            height=32
        )
        self.save_config_btn.pack(side="left", padx=(0, 5), expand=True, fill="x")
        
        self.load_config_btn = ctk.CTkButton(
            self.config_frame,
            text=f"{EMOJI_FOLDER} Load",
            command=self.load_config_ui,
            fg_color="#9b59b6",
            hover_color="#8e44ad",
            height=32
        )
        self.load_config_btn.pack(side="left", padx=(5, 0), expand=True, fill="x")
        
        # Reset Button
        self.reset_btn = ctk.CTkButton(
            self.sidebar,
            text=f"{EMOJI_REFRESH} Reset Data",
            command=self.reset_data,
            fg_color="#e74c3c",
            hover_color="#c0392b"
        )
        self.reset_btn.grid(row=16, column=0, padx=20, pady=20)
        
        # MAIN CONTENT
        self.main_frame = ctk.CTkFrame(self, corner_radius=0)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text=f"{EMOJI_EGG} Sortir Kualitas Telur Otomatis",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        self.title_label.grid(row=0, column=0, padx=20, pady=20)
        
        # Tabview
        self.tabview = ctk.CTkTabview(self.main_frame)
        self.tabview.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        
        # Tab Deteksi
        self.tab_deteksi = self.tabview.add("Deteksi")
        self.tab_deteksi.grid_columnconfigure(0, weight=1)
        self.tab_deteksi.grid_rowconfigure(1, weight=1)
        
        self.control_frame = ctk.CTkFrame(self.tab_deteksi)
        self.control_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        
        self.upload_btn = ctk.CTkButton(
            self.control_frame,
            text=f"{EMOJI_UPLOAD} Upload Gambar",
            command=self.upload_image
        )
        self.upload_btn.pack(side="left", padx=10, pady=10)
        
        self.start_cam_btn = ctk.CTkButton(
            self.control_frame,
            text=f"{EMOJI_PLAY} Mulai Kamera",
            command=self.start_camera,
            fg_color="#3498db"
        )
        self.start_cam_btn.pack(side="left", padx=10, pady=10)
        
        self.stop_cam_btn = ctk.CTkButton(
            self.control_frame,
            text=f"{EMOJI_STOP} Stop Kamera",
            command=self.stop_camera_feed,
            fg_color="#e74c3c",
            state="disabled"
        )
        self.stop_cam_btn.pack(side="left", padx=10, pady=10)
        
        self.image_frame = ctk.CTkFrame(self.tab_deteksi)
        self.image_frame.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        self.image_frame.grid_rowconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)
        
        self.image_label = ctk.CTkLabel(self.image_frame, text="Tidak ada gambar")
        self.image_label.grid(row=0, column=0, sticky="nsew")
        
        self.keputusan_label = ctk.CTkLabel(
            self.tab_deteksi,
            text="",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.keputusan_label.grid(row=2, column=0, padx=20, pady=10)
        
        self.shortcut_label = ctk.CTkLabel(
            self.tab_deteksi,
            text="Shortcuts: Space=Upload | Esc=Stop Camera",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.shortcut_label.grid(row=3, column=0, padx=20, pady=(0, 10))
        
        # Tab Log
        self.tab_log = self.tabview.add("Riwayat Log")
        self.tab_log.grid_columnconfigure(0, weight=1)
        self.tab_log.grid_rowconfigure(1, weight=1)
        
        # Tab Chart
        self.tab_chart = self.tabview.add("Live Chart")
        self.tab_chart.grid_columnconfigure(0, weight=1)
        self.tab_chart.grid_rowconfigure(0, weight=1)
        
        self.setup_live_chart()
        
        self.log_control_frame = ctk.CTkFrame(self.tab_log)
        self.log_control_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        
        self.refresh_log_btn = ctk.CTkButton(
            self.log_control_frame,
            text=f"{EMOJI_REFRESH} Refresh Log",
            command=self.refresh_log
        )
        self.refresh_log_btn.pack(side="left", padx=10, pady=10)
        
        self.export_log_btn = ctk.CTkButton(
            self.log_control_frame,
            text=f"{EMOJI_DOWN} Export CSV",
            command=self.export_log
        )
        self.export_log_btn.pack(side="left", padx=10, pady=10)
        
        self.log_textbox = ctk.CTkTextbox(self.tab_log, wrap="none")
        self.log_textbox.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        
        self.refresh_log()
    
    def setup_live_chart(self):
        """Setup live chart"""
        self.fig = Figure(figsize=(8, 6), dpi=100, facecolor='#1e1e1e')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#2b2b2b')
        self.ax.set_title('Real-time Detection Statistics', color='white', 
                         fontsize=14, weight='bold')
        self.ax.set_xlabel('Time', color='white')
        self.ax.set_ylabel('Count', color='white')
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.grid(True, alpha=0.2, color='gray')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_chart)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=20)
        
        self.line_accept, = self.ax.plot([], [], 'g-', label='ACCEPT', linewidth=2)
        self.line_reject, = self.ax.plot([], [], 'r-', label='REJECT', linewidth=2)
        self.ax.legend(loc='upper left', facecolor='#2b2b2b', 
                      edgecolor='white', labelcolor='white', fontsize=10)
        
        if self.chart_data["timestamps"]:
            self.update_live_chart()
    
    def update_live_chart(self):
        """Update live chart"""
        try:
            if not self.chart_data["timestamps"]:
                return
            
            if len(self.chart_data["timestamps"]) > self.max_chart_points:
                self.chart_data["timestamps"] = self.chart_data["timestamps"][-self.max_chart_points:]
                self.chart_data["accept"] = self.chart_data["accept"][-self.max_chart_points:]
                self.chart_data["reject"] = self.chart_data["reject"][-self.max_chart_points:]
            
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
    
    def update_conf(self, value):
        self.min_conf = value
        self.conf_value.configure(text=f"{value:.2f}")
    
    def toggle_log(self):
        self.aktifkan_log = self.log_checkbox.get()
    
    def toggle_alert(self):
        self.alert_enabled = self.alert_checkbox.get()
    
    def toggle_sound(self):
        self.alert_sound_enabled = self.sound_checkbox.get()
    
    def change_camera_source(self, value):
        if value == "Webcam":
            self.camera_source = "local"
            self.local_cam_frame.grid()
            self.ip_cam_frame.grid_remove()
        else:
            self.camera_source = "ip"
            self.local_cam_frame.grid_remove()
            self.ip_cam_frame.grid()
    
    def initial_camera_detection(self):
        self.available_cameras = self.detect_available_cameras()
        self.after(0, self.update_camera_dropdown)
    
    def refresh_cameras(self):
        self.detect_cam_btn.configure(state="disabled", text="⏳")
        self.local_cam_var.set("Detecting...")
        
        def detect_thread():
            self.available_cameras = self.detect_available_cameras()
            self.after(0, self.update_camera_dropdown)
            self.after(0, lambda: self.detect_cam_btn.configure(state="normal", text=f"{EMOJI_SEARCH}"))
            self.after(0, lambda: messagebox.showinfo("Info", f"Ditemukan {len(self.available_cameras)} kamera"))
        
        threading.Thread(target=detect_thread, daemon=True).start()
    
    def update_camera_dropdown(self):
        if not self.available_cameras:
            camera_names = ["No Camera Detected"]
        else:
            camera_names = [cam["name"] for cam in self.available_cameras]
        
        self.local_cam_menu.configure(values=camera_names)
        if camera_names:
            self.local_cam_var.set(camera_names[0])
    
    def get_selected_camera_index(self):
        selected_name = self.local_cam_var.get()
        for cam in self.available_cameras:
            if cam["name"] == selected_name:
                return cam["index"]
        return 0
    
    def change_device(self, value):
        try:
            old_device = self.device
            self.device = value
            
            print(f"Changing device from {old_device} to {value}...")
            
            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
            test_result = self.model.predict(test_img, verbose=False, device=value, conf=0.5)
            print(f"{EMOJI_CHECK} Test prediction successful on {value}")
            
            if value == "cuda":
                import torch
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.device_info_label.configure(
                    text=f"🚀 {gpu_name} ({gpu_memory:.1f}GB)",
                    text_color="#2ecc71"
                )
                messagebox.showinfo("Device Changed", f"Model sekarang menggunakan GPU:\n{gpu_name}\n\nDeteksi akan lebih cepat!")
            else:
                self.device_info_label.configure(text="💻 Using: CPU", text_color="gray")
                messagebox.showinfo("Device Changed", "Model sekarang menggunakan CPU\n\nDeteksi akan lebih lambat tapi lebih stabil.")
            
            print(f"{EMOJI_CHECK} Device changed to {value}")
            
        except Exception as e:
            print(f"Error changing device: {e}")
            messagebox.showerror("Error", f"Gagal mengubah device:\n{str(e)}\n\nKemungkinan CUDA tidak terinstall dengan benar.")
            self.device_var.set(old_device)
            self.device = old_device
    
    def update_stats(self):
        with self.stats_lock:
            accept = self.accept_count
            reject = self.reject_count
            fps = self.fps
            inference = self.inference_time
        
        self.accept_label.configure(text=f"{EMOJI_CHECK} ACCEPT: {accept}")
        self.reject_label.configure(text=f"{EMOJI_CROSS} REJECT: {reject}")
        self.perf_label.configure(text=f"{EMOJI_LIGHTNING} FPS: {fps:.1f}")
        self.inference_label.configure(text=f"{EMOJI_CLOCK} Inference: {inference:.0f}ms")
    
    def save_config_ui(self):
        self.ip_camera = self.ip_entry.get()
        if self.save_configuration():
            messagebox.showinfo("Success", "Configuration saved successfully!")
        else:
            messagebox.showerror("Error", "Failed to save configuration!")
    
    def load_config_ui(self):
        self.load_configuration()
        self.ip_entry.delete(0, "end")
        self.ip_entry.insert(0, self.ip_camera)
        self.conf_slider.set(self.min_conf)
        self.conf_value.configure(text=f"{self.min_conf:.2f}")
        
        if self.aktifkan_log:
            self.log_checkbox.select()
        else:
            self.log_checkbox.deselect()
        
        if self.alert_enabled:
            self.alert_checkbox.select()
        else:
            self.alert_checkbox.deselect()
        
        if self.alert_sound_enabled:
            self.sound_checkbox.select()
        else:
            self.sound_checkbox.deselect()
        
        self.device_var.set(self.device)
        messagebox.showinfo("Success", "Configuration loaded successfully!")
    
    def play_alert_sound(self):
        if not self.alert_sound_enabled:
            return
        
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
    
    def save_rejected_image(self, image_array, labels, keputusan):
        if keputusan == "REJECT":
            try:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"REJECT_{timestamp}.jpg"
                filepath = os.path.join(REJECTED_IMAGES_FOLDER, filename)
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, image_bgr)
                print(f"{EMOJI_CHECK} Saved rejected image: {filename}")
                return filepath
            except Exception as e:
                print(f"Error saving rejected image: {e}")
                return None
        return None
    
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
    
    def ambil_keputusan(self, labels):
        """
        Determine decision based on detected classes
        UPDATED: New class names
        - ACCEPT: 'clean', 'yellow egg'
        - REJECT: 'crack', 'dirty'
        """
        # Reject classes - normalize to lowercase for case-insensitive comparison
        reject_classes = ["crack", "dirty"]
        
        # Normalize labels to lowercase
        labels_lower = [label.lower() for label in labels]
        
        # Check if any reject class is detected
        is_reject = any(lbl in reject_classes for lbl in labels_lower)
        
        return "REJECT" if is_reject else "ACCEPT"
    
    def update_counts(self, keputusan):
        with self.stats_lock:
            if keputusan == "REJECT":
                self.reject_count += 1
            else:
                self.accept_count += 1
    
    def add_log_entry(self, log_entry):
        with self.log_lock:
            self.log_deteksi.append(log_entry)
    
    def upload_image(self):
        try:
            file_path = filedialog.askopenfilename(
                title="Pilih Gambar",
                filetypes=[("Image files", "*.jpg *.jpeg *.png")]
            )
            
            if not file_path:
                return
            
            print(f"Loading image: {file_path}")
            image = Image.open(file_path).convert("RGB")
            print(f"Image loaded: {image.size}")
            
            start_time = time.time()
            results = self.model.predict(np.array(image), verbose=False, device=self.device, conf=self.min_conf, half=False)
            self.inference_time = (time.time() - start_time) * 1000
            print(f"Prediction done on {self.device} in {self.inference_time:.1f}ms")
            
            boxes = results[0].boxes
            filtered_boxes = boxes[boxes.conf > self.min_conf]
            labels = [self.model.names[int(cls)] for cls in filtered_boxes.cls]
            
            result_img = results[0].plot()
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            
            keputusan = self.ambil_keputusan(labels) if labels else "ACCEPT"
            
            self.display_image(result_img)
            color = "#2ecc71" if keputusan == "ACCEPT" else "#e74c3c"
            self.keputusan_label.configure(text=f"{EMOJI_BRAIN} Keputusan: {keputusan}", text_color=color)
            
            self.update_counts(keputusan)
            
            if keputusan == "REJECT" and self.alert_enabled:
                self.play_alert_sound()
                messagebox.showwarning("REJECT Detected!", f"Telur ditolak!\nAlasan: {', '.join(labels)}")
            
            self.update_stats()
            
            self.chart_data["timestamps"].append(datetime.datetime.now())
            with self.stats_lock:
                self.chart_data["accept"].append(self.accept_count)
                self.chart_data["reject"].append(self.reject_count)
            self.update_live_chart()
            
            saved_path = self.save_rejected_image(result_img, labels, keputusan)
            
            if self.aktifkan_log:
                log_baru = {
                    "waktu": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "sumber": "Upload Gambar",
                    "kelas_terdeteksi": ", ".join(labels) if labels else "Tidak ada",
                    "keputusan": keputusan,
                    "confidence": f"{max([float(c) for c in filtered_boxes.conf], default=0):.2f}",
                    "inference_time_ms": f"{self.inference_time:.1f}",
                    "saved_path": saved_path if saved_path else "N/A"
                }
                self.add_log_entry(log_baru)
                self.save_log_append([log_baru])
                self.refresh_log()
            
            if keputusan == "ACCEPT":
                messagebox.showinfo("Sukses", f"Deteksi selesai!\nKeputusan: {keputusan}")
            
        except Exception as e:
            print(f"Error in upload_image: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Terjadi kesalahan:\n{str(e)}")
    
    def start_camera(self):
        try:
            if self.camera_running:
                messagebox.showwarning("Peringatan", "Kamera sudah berjalan!")
                return
            
            if self.camera_source == "ip":
                self.ip_camera = self.ip_entry.get()
                if not self.ip_camera:
                    messagebox.showerror("Error", "Masukkan IP kamera terlebih dahulu!")
                    return
                if not self.validate_ip(self.ip_camera):
                    messagebox.showerror("Error", "Format IP tidak valid!\n\nContoh IP yang benar: 192.168.1.6")
                    return
                print(f"Starting IP camera: {self.ip_camera}")
            else:
                self.local_camera_index = self.get_selected_camera_index()
                selected_cam = self.local_cam_var.get()
                print(f"Starting local camera: {selected_cam} (index {self.local_camera_index})")
            
            self.stop_camera = False
            self.camera_running = True
            self.start_cam_btn.configure(state="disabled")
            self.stop_cam_btn.configure(state="normal")
            
            thread = threading.Thread(target=self.camera_loop, daemon=True)
            thread.start()
            
        except Exception as e:
            print(f"Error in start_camera: {e}")
            messagebox.showerror("Error", f"Gagal memulai kamera:\n{str(e)}")
    
    def camera_loop(self):
        cap = None
        retry_count = 0
        max_retries = 3
        
        try:
            while retry_count < max_retries:
                if self.camera_source == "ip":
                    url = f"http://{self.ip_camera}:4747/video"
                    print(f"Connecting to IP camera: {url} (attempt {retry_count + 1})")
                    cap = cv2.VideoCapture(url)
                    time.sleep(2)
                    camera_name = f"DroidCam ({self.ip_camera})"
                else:
                    print(f"Opening local camera index: {self.local_camera_index}")
                    cap = cv2.VideoCapture(self.local_camera_index)
                    time.sleep(1)
                    camera_info = next((cam for cam in self.available_cameras if cam["index"] == self.local_camera_index), None)
                    camera_name = camera_info["name"] if camera_info else f"Webcam {self.local_camera_index}"
                
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        print(f"{EMOJI_CHECK} Camera opened successfully: {camera_name}")
                        break
                    else:
                        cap.release()
                
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Retrying... ({retry_count}/{max_retries})")
                    time.sleep(1)
            
            if not cap or not cap.isOpened():
                error_msg = f"Gagal membuka {camera_name}!\n\n"
                if self.camera_source == "ip":
                    error_msg += "Pastikan:\n1. DroidCam sudah running\n2. IP address benar\n3. Port 4747 terbuka"
                else:
                    error_msg += "Pastikan:\n1. Kamera tidak digunakan aplikasi lain\n2. Index kamera benar\n3. Izin kamera sudah diberikan"
                self.after(0, lambda: messagebox.showerror("Error", error_msg))
                return
            
            if self.device == "cuda":
                import torch
                print("GPU mode detected, enabling CUDA synchronization")
            
            frame_count = 0
            frame_delay = 1.0 / TARGET_FPS
            last_detection_time = time.time()
            detection_interval = 0.1
            
            while not self.stop_camera:
                frame_start = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame, attempting reconnect...")
                    time.sleep(0.5)
                    continue
                
                frame_count += 1
                
                try:
                    self.calculate_fps()
                    frame = cv2.resize(frame, (640, 480))
                    
                    current_time = time.time()
                    if current_time - last_detection_time >= detection_interval:
                        start_time = time.time()
                        results = self.model.predict(frame, verbose=False, conf=self.min_conf, device=self.device, half=False)
                        self.inference_time = (time.time() - start_time) * 1000
                        last_detection_time = current_time
                        
                        if self.device == "cuda":
                            import torch
                            torch.cuda.synchronize()
                        
                        boxes = results[0].boxes
                        filtered_boxes = boxes[boxes.conf > self.min_conf]
                        labels = [self.model.names[int(cls)] for cls in filtered_boxes.cls]
                        
                        frame_result = results[0].plot()
                        frame_rgb = cv2.cvtColor(frame_result, cv2.COLOR_BGR2RGB)
                        
                        keputusan = self.ambil_keputusan(labels) if labels else "ACCEPT"
                        
                        self.display_image(frame_rgb)
                        self.after(0, self._update_keputusan_ui, keputusan)
                        
                        if labels:
                            self.update_counts(keputusan)
                            if keputusan == "REJECT" and self.alert_enabled:
                                self.play_alert_sound()
                        
                        self.after(0, self.update_stats)
                        
                        now = datetime.datetime.now()
                        if self.aktifkan_log and labels and (now - self.last_log_time).total_seconds() >= 1:
                            log_baru = {
                                "waktu": now.strftime("%Y-%m-%d %H:%M:%S"),
                                "sumber": "Kamera",
                                "kelas_terdeteksi": ", ".join(labels),
                                "keputusan": keputusan,
                                "confidence": f"{max([float(c) for c in filtered_boxes.conf]):.2f}",
                                "inference_time_ms": f"{self.inference_time:.1f}",
                                "fps": f"{self.fps:.1f}"
                            }
                            self.add_log_entry(log_baru)
                            self.save_log_append([log_baru])
                            self.last_log_time = now
                            
                            self.chart_data["timestamps"].append(now)
                            with self.stats_lock:
                                self.chart_data["accept"].append(self.accept_count)
                                self.chart_data["reject"].append(self.reject_count)
                            self.after(0, self.update_live_chart)
                        
                        if frame_count % 30 == 0:
                            print(f"Frame {frame_count}: Detected {len(boxes)} objects (device: {self.device}, FPS: {self.fps:.1f})")
                    
                    elapsed = time.time() - frame_start
                    sleep_time = max(0, frame_delay - elapsed)
                    time.sleep(sleep_time)
                    
                except Exception as frame_error:
                    print(f"Error processing frame {frame_count}: {frame_error}")
                    continue
            
            print(f"Camera stopped. Total frames: {frame_count}")
            
        except Exception as e:
            print(f"Error in camera_loop: {e}")
            import traceback
            traceback.print_exc()
            self.after(0, lambda: messagebox.showerror("Error", f"Error kamera:\n{str(e)}"))
        
        finally:
            if cap is not None:
                cap.release()
            
            if self.device == "cuda":
                try:
                    import torch
                    torch.cuda.empty_cache()
                    print("GPU cache cleared")
                except:
                    pass
            
            gc.collect()
            self.camera_running = False
            self.after(0, lambda: self.start_cam_btn.configure(state="normal"))
            self.after(0, lambda: self.stop_cam_btn.configure(state="disabled"))
            print("Camera released")
    
    def _update_keputusan_ui(self, keputusan):
        color = "#2ecc71" if keputusan == "ACCEPT" else "#e74c3c"
        self.keputusan_label.configure(text=f"{EMOJI_BRAIN} Keputusan: {keputusan}", text_color=color)
    
    def stop_camera_feed(self):
        if not self.camera_running:
            return
        print("Stopping camera...")
        self.stop_camera = True
        self.stop_cam_btn.configure(state="disabled")
        messagebox.showinfo("Info", "Kamera sedang dihentikan...")
    
    def display_image(self, img_array):
        try:
            h, w = img_array.shape[:2]
            max_w, max_h = 800, 600
            scale = min(max_w/w, max_h/h)
            new_w, new_h = int(w*scale), int(h*scale)
            img_resized = cv2.resize(img_array, (new_w, new_h))
            img_pil = Image.fromarray(img_resized)
            img_tk = ImageTk.PhotoImage(img_pil)
            self.after(0, self._update_image_label, img_tk)
        except Exception as e:
            print(f"Error displaying image: {e}")
    
    def _update_image_label(self, img_tk):
        if hasattr(self.image_label, 'image') and self.image_label.image:
            try:
                del self.image_label.image
            except:
                pass
        self.image_label.configure(image=img_tk, text="")
        self.image_label.image = img_tk
    
    def refresh_log(self):
        self.log_textbox.delete("1.0", "end")
        if not self.log_deteksi:
            self.log_textbox.insert("1.0", "Belum ada log deteksi.\n")
            return
        df = pd.DataFrame(self.log_deteksi)
        log_text = df.to_string(index=False)
        self.log_textbox.insert("1.0", log_text)
    
    def export_log(self):
        if not self.log_deteksi:
            messagebox.showinfo("Info", "Tidak ada log untuk diekspor.")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        
        if file_path:
            try:
                df = pd.DataFrame(self.log_deteksi)
                df.to_csv(file_path, index=False)
                
                total = len(df)
                accepts = sum(1 for log in self.log_deteksi if log.get("keputusan") == "ACCEPT")
                rejects = sum(1 for log in self.log_deteksi if log.get("keputusan") == "REJECT")
                accept_pct = (accepts / total * 100) if total > 0 else 0
                reject_pct = (rejects / total * 100) if total > 0 else 0
                
                messagebox.showinfo("Sukses", f"Log berhasil diekspor ke:\n{file_path}\n\nSummary:\nTotal: {total}\nAccept: {accepts} ({accept_pct:.1f}%)\nReject: {rejects} ({reject_pct:.1f}%)")
            except Exception as e:
                messagebox.showerror("Error", f"Gagal export log:\n{str(e)}")
    
    def reset_data(self):
        confirm = messagebox.askyesno("Konfirmasi", "Yakin ingin mereset semua data?\n\nIni akan menghapus:\n- Semua log deteksi\n- Statistik counter\n- Data chart\n\nFile CSV akan dihapus!")
        
        if confirm:
            with self.stats_lock:
                self.reject_count = 0
                self.accept_count = 0
            
            with self.log_lock:
                self.log_deteksi = []
            
            self.last_log_time = datetime.datetime.min
            self.fps = 0
            self.inference_time = 0
            self.frame_times = []
            
            self.chart_data = {"timestamps": [], "accept": [], "reject": []}
            self.update_live_chart()
            
            if os.path.exists(LOG_FILE):
                os.remove(LOG_FILE)
            
            self.update_stats()
            self.refresh_log()
            self.keputusan_label.configure(text="")
            
            if hasattr(self.image_label, 'image') and self.image_label.image:
                try:
                    del self.image_label.image
                except:
                    pass
            
            self.image_label.configure(image=None, text="Tidak ada gambar")
            gc.collect()
            
            messagebox.showinfo("Sukses", "Data berhasil direset!")


if __name__ == "__main__":
    app = EggSorterApp()
    app.mainloop()