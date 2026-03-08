from ultralytics import YOLO

# Load trained model
model = YOLO(r"C:\Users\Sourav\OneDrive\문서\Desktop\ML\runs\detect\train4\weights\best.pt")

# Run webcam detection
model.predict(
    source=0,   # webcam
    show=True,
    conf=0.6
)
