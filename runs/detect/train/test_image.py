from ultralytics import YOLO
import cv2
import os

# Load trained model
model = YOLO(r"C:\Users\Sourav\OneDrive\문서\Desktop\ML\runs\detect\train4\weights\best.pt")

# Dataset image folder
image_folder = r"C:\Users\Sourav\OneDrive\문서\Desktop\ML\test\images"

images = os.listdir(image_folder)

for img_name in images:
    img_path = os.path.join(image_folder, img_name)

    results = model(
        img_path,
        conf=0.25,
        iou=0.45,
        imgsz=1024,
        max_det=100
    )

    r = results[0]
    img = r.orig_img.copy()

    for box in r.boxes:

        cls = int(box.cls[0])

        # Only allow helmet and without_helmet
        if cls not in [0, 2]:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        label = model.names[cls]

        # 🔴🟢 Color logic
        if label == "without_helmet":
            color = (0,0,255)   # RED
        else:
            color = (0,255,0)   # GREEN

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        cv2.putText(img, f"{label} {conf:.2f}",
                    (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2)

    cv2.imshow("Detection", img)

    key = cv2.waitKey(0)

    if key == 27:
        break

cv2.destroyAllWindows()