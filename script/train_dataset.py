from ultralytics import YOLO
import torch
from pathlib import Path
import gc

if __name__ == "__main__":
    # Clear CUDA cache
    torch.cuda.empty_cache()
    gc.collect()
    
    print("="*70)
    print("YOLO11 TRAINING - OPTIMIZED FOR RTX 3050 4GB")
    print("="*70)
    
    # GPU Check
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n🎮 GPU: {gpu_name}")
        print(f"💾 VRAM: {vram_total:.2f} GB")
        device = 0
    else:
        print("⚠️  Using CPU")
        device = "cpu"
    
    # Paths
    data_path = Path(r"E:\PROYEK_KLASIFIKASI_TELUR\dataset\datav11\data.yaml")
    project_path = Path(r"E:\PROYEK_KLASIFIKASI_TELUR\runs\detect")
    
    # Load model
    print(f"\n📦 Loading YOLOv11n model...")
    model = YOLO("yolo11n.pt")
    
    # Training dengan konfigurasi OPTIMAL untuk 4GB VRAM
    print(f"\n🚀 Starting training with optimized settings...")
    print("="*70)
    
    results = model.train(
        # Dataset
        data=str(data_path),
        
        # === CRITICAL: VRAM OPTIMIZATION ===
        epochs=100,
        batch=8,               # ⬇️ Turunkan dari 16 ke 8
        imgsz=416,             # ⬇️ Turunkan dari 640 ke 416 (HEMAT VRAM)
        device=device,
        workers=2,             # ⬇️ Turunkan dari 8 ke 4
        
        # === PERFORMANCE ===
        cache=False,           # ⚠️ Disable cache (hemat VRAM, tapi lambat)
        amp=True,              # ✅ Keep AMP untuk hemat VRAM
        close_mosaic=0,        # Disable mosaic di epoch akhir
        
        # === OPTIMIZER ===
        optimizer='SGD',       # SGD lebih hemat VRAM dari AdamW
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        
        # === AUGMENTATION (Simplified) ===
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=0.5,            # ⬇️ Turunkan mosaic intensity
        mixup=0.0,             # Disable mixup (hemat VRAM)
        copy_paste=0.0,        # Disable copy-paste
        
        # === VALIDATION ===
        val=True,
        patience=20,           # ⬆️ Naikkan patience karena konvergensi lambat
        
        # === SAVING ===
        save=True,
        save_period=10,        # Save setiap 10 epoch
        
        # === OUTPUT ===
        project=str(project_path),
        name="yolo11n_telur_optimized",
        exist_ok=False,
        verbose=True,
        plots=True,
    )
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED!")
    print("="*70)