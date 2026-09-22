from ultralytics import YOLO
import torch
import os

if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    print(f"Jumlah GPU terdeteksi: {num_gpus}")
    selected_gpu = 1 if num_gpus > 1 else 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
    device = torch.device(f"cuda:{selected_gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device} - {torch.cuda.get_device_name(selected_gpu) if torch.cuda.is_available() else 'CPU'}")
    model = YOLO("yolov8n.pt") 

    model.train(
        data=r"E:\PROYEK_KLASIFIKASI_TELUR\dataset\dataV8\data.yaml",
        epochs=50,
        batch=32,
        imgsz=416,
        device=0,
        workers=2,
        cache=True,
        amp=True,
        single_cls=False,
        augment=True,
        val=True,
        patience=10,
        project="E:/PROYEK_KLASIFIKASI_TELUR/runs/detect",
        name="train_tes",
        box=0.5,
        iou=0.5,
    )