from ultralytics import YOLO
import os
import cv2

model = YOLO("yolov8n.pt")  # lightweight

image_folder = os.path.expanduser("~/Documents/gsplat_project/AliceVision_processed/desk_acfr_robot")

for filename in os.listdir(image_folder):
    # if not filename.lower().endswith((".jpg", ".png")):
    #     continue
    path = os.path.join(image_folder, filename)
    img = cv2.imread(path)

    results = model(img)

    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0:  # class 0 = person
                print("Removing:", filename)
                os.remove(path)
                break
