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


yolo = YOLO('best.pt')

@app.route("/detect-single", methods=["POST"])
def detect_single():
    if 'file' in request.files:
        f = request.files['file']
        file_extension = f.filename.rsplit('.', 1)[1].lower()

        if file_extension in ['jpg', 'jpeg', 'png']:
            basepath = os.path.dirname(__file__)
            upload_folder = os.path.join(basepath, 'uploads', 'single')
            os.makedirs(upload_folder, exist_ok=True)
            detect_f = os.path.join(basepath, 'runs', 'detect','predict')
            os.makedirs(detect_f, exist_ok=True)

            timestamp = int(time.time() * 1000)
            new_filename = f"img_{timestamp}.{file_extension}"
            upload_path = os.path.join(upload_folder, new_filename)
            f.save(upload_path)

            img = cv2.imread(upload_path)
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            image = Image.open(io.BytesIO(frame))

            # yolo = YOLO('best.pt')
            results = yolo.predict(image, save=False)

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
                files = [f for f in os.listdir(latest_subfolder_path) if os.path.isfile(os.path.join(latest_subfolder_path, f))]
                if files:
                    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(latest_subfolder_path, x)))
                    image_path = os.path.join('runs', 'detect', latest_subfolder, latest_file)
                    return jsonify({
                        "image_path": image_path,
                        "total_spots": len(detections),
                        "detections": detections
                    })

    return "Detection failed", 500



@app.route("/detect-folder", methods=["POST"])
def detect_folder():
    files = request.files.getlist("files")
    if not files:
        return "No files found", 400

    basepath = os.path.dirname(__file__)
    timestamp = int(time.time() * 1000)
    upload_folder = os.path.join(basepath, 'uploads', f'folder_{timestamp}')
    os.makedirs(upload_folder, exist_ok=False)

    filenames = []
    for file in files:
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        filenames.append(filename)

    # yolo = YOLO('best.pt')
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

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Flask app exposing YOLOv8 models")
#     parser.add_argument("--port", default=5000, type=int, help="port number")
#     args = parser.parse_args()
#     app.run(debug=True, port=args.port)





import pkg_resources
import logging

# List of required packages
required_packages = [
    "blinker==1.9.0",
    "certifi==2025.4.26",
    "charset-normalizer==3.4.2",
    "click==8.2.1",
    "colorama==0.4.6",
    "contourpy==1.3.2",
    "cycler==0.12.1",
    "filelock==3.18.0",
    "Flask==3.1.1",
    "fonttools==4.58.1",
    "fsspec==2025.5.1",
    "gunicorn==23.0.0",
    "idna==3.10",
    "itsdangerous==2.2.0",
    "Jinja2==3.1.6",
    "kiwisolver==1.4.8",
    "MarkupSafe==3.0.2",
    "matplotlib==3.10.3",
    "mpmath==1.3.0",
    "networkx==3.4.2",
    "numpy==2.2.6",
    "opencv-python==4.11.0.86",
    "packaging==25.0",
    "pandas==2.3.0",
    "pillow==11.2.1",
    "psutil==7.0.0",
    "py-cpuinfo==9.0.0",
    "pyparsing==3.2.3",
    "python-dateutil==2.9.0.post0",
    "pytz==2025.2",
    "PyYAML==6.0.2",
    "requests==2.32.3",
    "scipy==1.15.3",
    "six==1.17.0",
    "sympy==1.14.0",
    "torch==2.7.1",
    "torchvision==0.22.1",
    "tqdm==4.67.1",
    "typing_extensions==4.14.0",
    "tzdata==2025.2",
    "ultralytics==8.3.151",
    "ultralytics-thop==2.0.14",
    "urllib3==2.4.0",
    "Werkzeug==3.1.3",
]

# Check installation
def check_packages():
    results = []
    for package in required_packages:
        try:
            pkg_resources.require(package)
            results.append(f"{package} is installed")
        except pkg_resources.DistributionNotFound:
            results.append(f"{package} is NOT installed")
        except pkg_resources.VersionConflict as e:
            results.append(f"{package} version conflict: {e}")
    logging.info("\n" + "\n".join(results))

check_packages()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)

