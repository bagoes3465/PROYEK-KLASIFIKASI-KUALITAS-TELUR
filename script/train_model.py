"""
YOLO Training Module untuk Egg Sorter
Optimized untuk RTX 3050 4GB VRAM
"""
from ultralytics import YOLO
import torch
from pathlib import Path
import gc
import time
from datetime import datetime

class YOLOTrainer:
    def __init__(self, callback=None):
        """
        Initialize YOLO Trainer
        
        Args:
            callback: Function untuk update progress (epoch, metrics)
        """
        self.callback = callback
        self.model = None
        self.training_active = False
        self.training_results = None
        
    def check_gpu(self):
        """Check GPU availability dan info"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            vram_free = (torch.cuda.get_device_properties(0).total_memory - 
                        torch.cuda.memory_allocated(0)) / 1e9
            return {
                "available": True,
                "name": gpu_name,
                "vram_total": vram_total,
                "vram_free": vram_free,
                "device": 0
            }
        return {
            "available": False,
            "name": "CPU",
            "device": "cpu"
        }
    
    def prepare_training(self, model_size="yolo11n.pt"):
        """Load YOLO model"""
        try:
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            print(f"📦 Loading {model_size}...")
            self.model = YOLO(model_size)
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def train(self, 
              data_yaml_path,
              project_path="runs/detect",
              name="yolo11_telur",
              epochs=100,
              batch=8,
              imgsz=416,
              workers=2,
              patience=20,
              save_period=10,
              device=None):
        """
        Start training with optimized settings for 4GB VRAM
        
        Args:
            data_yaml_path: Path ke data.yaml
            project_path: Folder output
            name: Nama experiment
            epochs: Jumlah epoch
            batch: Batch size (8 untuk 4GB VRAM)
            imgsz: Image size (416 untuk hemat VRAM)
            workers: Jumlah workers
            patience: Early stopping patience
            save_period: Save model setiap N epoch
            device: Device (None=auto, 0=GPU, 'cpu'=CPU)
        """
        if not self.model:
            print("❌ Model belum dimuat!")
            return False
        
        try:
            self.training_active = True
            
            # Auto-detect device jika tidak dispesifikasikan
            if device is None:
                gpu_info = self.check_gpu()
                device = gpu_info["device"]
            
            print("\n" + "="*70)
            print("🚀 STARTING YOLO TRAINING")
            print("="*70)
            print(f"📊 Dataset: {data_yaml_path}")
            print(f"🎯 Model: YOLOv11")
            print(f"💾 Batch: {batch} | Image Size: {imgsz}x{imgsz}")
            print(f"🔄 Epochs: {epochs} | Workers: {workers}")
            print(f"💻 Device: {device}")
            print("="*70 + "\n")
            
            # Training dengan callback untuk progress
            self.training_results = self.model.train(
                # Dataset
                data=str(data_yaml_path),
                
                # === CRITICAL: VRAM OPTIMIZATION ===
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                device=device,
                workers=workers,
                
                # === PERFORMANCE ===
                cache=False,           # Disable cache (hemat VRAM)
                amp=True,              # Mixed precision training
                close_mosaic=0,
                
                # === OPTIMIZER ===
                optimizer='SGD',       # SGD lebih hemat VRAM
                lr0=0.01,
                lrf=0.01,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=3,
                warmup_momentum=0.8,
                
                # === AUGMENTATION ===
                augment=True,
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=0.0,
                translate=0.1,
                scale=0.5,
                fliplr=0.5,
                mosaic=0.5,
                mixup=0.0,
                copy_paste=0.0,
                
                # === VALIDATION ===
                val=True,
                patience=patience,
                
                # === SAVING ===
                save=True,
                save_period=save_period,
                
                # === OUTPUT ===
                project=str(project_path),
                name=name,
                exist_ok=False,
                verbose=True,
                plots=True,
            )
            
            self.training_active = False
            
            print("\n" + "="*70)
            print("✅ TRAINING COMPLETED!")
            print("="*70)
            
            # Get best model path
            best_model = Path(project_path) / name / "weights" / "best.pt"
            last_model = Path(project_path) / name / "weights" / "last.pt"
            
            print(f"\n📁 Results saved to: {Path(project_path) / name}")
            print(f"🏆 Best model: {best_model}")
            print(f"💾 Last model: {last_model}")
            
            return {
                "success": True,
                "best_model": str(best_model),
                "last_model": str(last_model),
                "results": self.training_results
            }
            
        except Exception as e:
            self.training_active = False
            print(f"❌ Training error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def stop_training(self):
        """Stop training (belum full support dari ultralytics)"""
        self.training_active = False
        print("⚠️ Training akan berhenti setelah epoch selesai...")
    
    def validate_dataset(self, data_yaml_path):
        """Validate dataset sebelum training"""
        try:
            import yaml
            
            with open(data_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Check required fields
            required_fields = ['train', 'val', 'nc', 'names']
            for field in required_fields:
                if field not in data:
                    return False, f"Missing field: {field}"
            
            # Check paths exist
            base_path = Path(data_yaml_path).parent
            train_path = base_path / data['train']
            val_path = base_path / data['val']
            
            if not train_path.exists():
                return False, f"Training path not found: {train_path}"
            if not val_path.exists():
                return False, f"Validation path not found: {val_path}"
            
            return True, {
                "nc": data['nc'],
                "names": data['names'],
                "train": str(train_path),
                "val": str(val_path)
            }
            
        except Exception as e:
            return False, f"Error validating dataset: {e}"


# Standalone training script
if __name__ == "__main__":
    trainer = YOLOTrainer()
    
    # Check GPU
    gpu_info = trainer.check_gpu()
    if gpu_info["available"]:
        print(f"🎮 GPU: {gpu_info['name']}")
        print(f"💾 VRAM: {gpu_info['vram_total']:.2f} GB")
    else:
        print("⚠️  Using CPU")
    
    # Paths (SESUAIKAN DENGAN PATH ANDA!)
    data_path = Path("dataset/data.yaml")  # <-- UBAH INI
    project_path = Path("runs/detect")
    
    # Validate dataset
    valid, result = trainer.validate_dataset(data_path)
    if not valid:
        print(f"❌ Dataset validation failed: {result}")
        exit(1)
    
    print(f"✅ Dataset valid:")
    print(f"   Classes: {result['nc']}")
    print(f"   Names: {result['names']}")
    
    # Load model
    if not trainer.prepare_training("yolo11n.pt"):
        print("❌ Failed to load model")
        exit(1)
    
    # Start training
    results = trainer.train(
        data_yaml_path=data_path,
        project_path=project_path,
        name="yolo11_telur_optimized",
        epochs=100,
        batch=8,
        imgsz=416,
        workers=2,
        patience=20,
        save_period=10
    )
    
    if results["success"]:
        print(f"\n🎉 Training berhasil!")
        print(f"Best model: {results['best_model']}")
    else:
        print(f"\n❌ Training gagal: {results['error']}")