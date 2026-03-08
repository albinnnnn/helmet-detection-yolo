from ultralytics import YOLO

def main():
    model = YOLO("yolo11s.pt")

    model.train(
        data=r"dataset\data.yaml",
        epochs=80,
        imgsz=640,
        batch=16,
        device=0,
        workers=8,

        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,

        weight_decay=0.0005,
        plots=True,
        save=True
    )

if __name__ == "__main__":
    main()