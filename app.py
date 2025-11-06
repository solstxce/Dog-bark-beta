from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os
import threading
import requests
import io

app = Flask(__name__)

model = YOLO("yolo12n.pt")

# Sound files
DOG_BARK = "/home/admin/free-dog-bark-419014_2hlHxSyr.wav"
CAT_MEOW = "/home/admin/cat-meowing-401730.wav"

def play_sound(path):
    # Try to play sound in a cross-platform-friendly way; this still
    # calls an OS command — adjust paths and player as needed for Windows.
    os.system(f"aplay '{path}'")


def _draw_and_encode(frame, boxes, names):
    # Draw boxes on a copy of the frame and return base64 JPEG
    out = frame.copy()
    detected_label = None
    detected_boxes = []

    for box in boxes:
        cls_id = int(box.cls)
        cls_name = names.get(cls_id, str(cls_id)).lower()
        xy = box.xyxy[0]
        x1, y1, x2, y2 = map(int, [xy[0].item(), xy[1].item(), xy[2].item(), xy[3].item()])

        if cls_name in ["cat", "rat"]:
            color = (34, 197, 94) if cls_name == "cat" else (250, 204, 21)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, cls_name.upper(), (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            detected_boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": cls_name})
            if not detected_label:
                detected_label = cls_name

    # Encode to JPEG
    success, encoded = cv2.imencode('.jpg', out)
    if not success:
        raise RuntimeError('Failed to encode image')
    b64 = base64.b64encode(encoded.tobytes()).decode('utf-8')

    return detected_label, detected_boxes, f"data:image/jpeg;base64,{b64}", out

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

    detected_label, detected_boxes, image_b64, _ = _draw_and_encode(frame, boxes, names)

    # Play sound asynchronously for the first detected label
    if detected_label:
        sound = DOG_BARK if detected_label == "cat" else CAT_MEOW
        threading.Thread(target=play_sound, args=(sound,), daemon=True).start()

    return jsonify({
        "detected": bool(detected_label),
        "label": detected_label,
        "boxes": detected_boxes,
        "image": image_b64
    })


@app.route('/fetch_stream', methods=['POST'])
def fetch_stream():
    """Fetch a JPEG from a remote URL (simple IP camera that serves a single JPEG per GET),
    run detection and return annotated image + metadata.
    Expects JSON: {"url": "http://<ip>/stream"}
    """
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({"error": "No url provided"}), 400

    url = data['url']
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
    except Exception as e:
        return jsonify({"error": f"Failed to fetch URL: {e}"}), 502

    # Try to decode JPEG bytes
    try:
        arr = np.frombuffer(resp.content, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError('Could not decode image')
    except Exception as e:
        return jsonify({"error": f"Failed to decode image: {e}"}), 502

    results = model(frame, imgsz=320, verbose=False)
    boxes = results[0].boxes
    names = results[0].names

    detected_label, detected_boxes, image_b64, _ = _draw_and_encode(frame, boxes, names)

    if detected_label:
        sound = DOG_BARK if detected_label == "cat" else CAT_MEOW
        threading.Thread(target=play_sound, args=(sound,), daemon=True).start()

    return jsonify({
        "detected": bool(detected_label),
        "label": detected_label,
        "boxes": detected_boxes,
        "image": image_b64
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
