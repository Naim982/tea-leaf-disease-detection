from flask import Flask, render_template, request, jsonify, send_file, url_for, Response, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image
import os, cv2, io, time
import numpy as np
import base64
import re
import time

app = Flask(__name__)

# Dynamic routing for simple pages
@app.route('/')
@app.route('/about')
@app.route('/single')
@app.route('/multiple')
def render_page():
    path = request.path
    if path == '/':
        return render_template('index.html')
    return render_template(f"{path.strip('/')}.html")

@app.route("/detect-single", methods=["POST"])
def detect_single():
    if 'file' in request.files:
        f = request.files['file']
        file_extension = f.filename.rsplit('.', 1)[1].lower()
        # img_name = f.filename

        if file_extension in ['jpg', 'jpeg', 'png']:
            basepath = os.path.dirname(__file__)
            upload_folder = os.path.join(basepath, 'uploads', 'single')
            os.makedirs(upload_folder, exist_ok=True)

            # imgg_path = os.path.join(upload_folder, img_name)
            # best_path = os.path.join(basepath, best.pt)
            # print("Upload folder created:", os.path.exists(best_path))
            # print("\n\n")
            
            timestamp = int(time.time() * 1000)
            new_filename = f"img_{timestamp}.{file_extension}"
            upload_path = os.path.join(upload_folder, new_filename)
            f.save(upload_path)

            img = cv2.imread(upload_path)
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            image = Image.open(io.BytesIO(frame))

            yolo = YOLO('best.pt')
            results = yolo.predict(source=image_path, save=True, imgsz=640, device='cpu')

            # results = yolo.predict(image, save=True)

            detections = []
            r = results[0]
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = yolo.names[cls_id]
                detections.append({
                    "class": class_name,
                    "confidence": round(conf, 2)
                })

            basepath = os.path.dirname(__file__)
            folder_path = os.path.join(basepath, 'runs', 'detect')
            subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

            if subfolders:
                latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
                latest_subfolder_path = os.path.join(folder_path, latest_subfolder)
                imgg_path = os.path.join(latest_subfolder_path, image0.jpg)
                files = [f for f in os.listdir(latest_subfolder_path) if os.path.isfile(os.path.join(latest_subfolder_path, f))]
                if files:
                    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(latest_subfolder_path, x)))
                    image_path = os.path.join('runs', 'detect', latest_subfolder, latest_file)
                    return jsonify({
                        "image_path": image_path,
                        "total_spots": len(detections),
                        "detections": detections
                    })
            # return imgg_path

    return "Detection failed", 500



@app.route("/detect-folder", methods=["POST"])
def detect_folder():
    files = request.files.getlist("files")
    if not files:
        return "No files found", 400

    basepath = os.path.dirname(__file__)
    timestamp = int(time.time() * 1000)
    upload_folder = os.path.join(basepath, 'uploads', f'folder_{timestamp}')
    os.makedirs(upload_folder, exist_ok=True)

    filenames = []
    for file in files:
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        filenames.append(filename)

    yolo = YOLO('best.pt')
    results = yolo.predict(upload_folder, save=True)

    result_descriptions = {}
    for i, r in enumerate(results):
        detections = []
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = yolo.names[cls_id]
            detections.append({
                "class": class_name,
                "confidence": round(conf, 2)
            })
        result_descriptions[filenames[i]] = {
            "total_spots": len(detections),
            "detections": detections
        }

    folder_path = os.path.join(basepath, 'runs', 'detect')
    subfolders = sorted(os.listdir(folder_path), key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    latest_folder = os.path.join(folder_path, subfolders[-1])

    result_paths = []
    for f in os.listdir(latest_folder):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            result_paths.append({
                "image": url_for('send_detected_image', folder=subfolders[-1], filename=f),
                "description": result_descriptions.get(f, {"total_spots": 0, "detections": []})
            })
    return jsonify(result_paths)

@app.route('/runs/detect/<folder>/<filename>')
def send_detected_image(folder, filename):
    return send_from_directory(os.path.join('runs', 'detect', folder), filename)




@app.route("/history")
def history():
    basepath = os.path.dirname(__file__)
    detect_path = os.path.join(basepath, 'runs', 'detect')

    if not os.path.exists(detect_path):
        return render_template("history.html", history=[])

    folders = sorted(os.listdir(detect_path), key=lambda x: os.path.getctime(os.path.join(detect_path, x)), reverse=True)

    history = []
    for folder in folders:
        folder_path = os.path.join(detect_path, folder)
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if images:
            history.append((folder, images))

    return render_template("history.html", history=history)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Flask app exposing YOLOv8 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(debug=True, port=args.port)
