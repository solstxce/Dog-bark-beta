from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os
import threading

app = Flask(__name__)

model = YOLO("yolo12n.pt")

# Sound files
DOG_BARK = "/home/admin/free-dog-bark-419014_2hlHxSyr.wav"
CAT_MEOW = "/home/admin/cat-meowing-401730.wav"

def play_sound(path):
    os.system(f"aplay '{path}'")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({"error": "No image data"}), 400

    # Decode image from browser
    img_data = data['image'].split(',')[1]
    np_img = np.frombuffer(base64.b64decode(img_data), np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Run YOLO
    results = model(frame, imgsz=320, verbose=False)
    boxes = results[0].boxes
    names = results[0].names

    detected_label = None
    detected_boxes = []

    for box in boxes:
        cls_name = names[int(box.cls)].lower()
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls_name in ["cat", "rat"]:
            detected_boxes.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "label": cls_name
            })
            if not detected_label:
                detected_label = cls_name
                # Play sound asynchronously
                sound = DOG_BARK if cls_name == "cat" else CAT_MEOW
                threading.Thread(target=play_sound, args=(sound,), daemon=True).start()

    return jsonify({
        "detected": bool(detected_label),
        "label": detected_label,
        "boxes": detected_boxes
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
