from flask import Flask, render_template, jsonify, request
import cv2
from ultralytics import YOLO
import threading
import time
import base64
import numpy as np
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load YOLO models (from your original code)
vehicle_model = YOLO("yolov8n.pt")  # or yolov8s.pt for better accuracy
pedestrian_model = YOLO("yolov8n.pt")

# Classes (from your original code)
vehicle_classes = ['car', 'bus', 'truck', 'motorcycle']
pedestrian_class = 'person'

# Thresholds (from your original code)
vehicle_threshold = 6
pedestrian_threshold = 8
stop_threads = False

# Global variables to share data between threads
vehicle_count = 0
pedestrian_count = 0
vehicle_frame = None
pedestrian_frame = None
signal_status = "System Initializing..."
system_active = False
pause_vehicle = threading.Event()
pause_vehicle.set()  # Vehicle video runs by default

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint to get current system status
@app.route('/status')
def get_status():
    global vehicle_count, pedestrian_count, signal_status, system_active
    return jsonify({
        'vehicle_count': vehicle_count,
        'pedestrian_count': pedestrian_count,
        'signal_status': signal_status,
        'system_active': system_active,
        'vehicle_frame': frame_to_base64(vehicle_frame),
        'pedestrian_frame': frame_to_base64(pedestrian_frame)
    })

# API endpoint to start the system
@app.route('/start', methods=['POST'])
def start_system():
    global system_active, stop_threads
    
    if not system_active:
        system_active = True
        stop_threads = False
        
        # Start processing threads
        vehicle_thread = threading.Thread(target=process_vehicle_feed, daemon=True)
        pedestrian_thread = threading.Thread(target=process_pedestrian_feed, daemon=True)
        control_thread = threading.Thread(target=traffic_control_loop, daemon=True)
        
        vehicle_thread.start()
        pedestrian_thread.start()
        control_thread.start()
        
        return jsonify({'status': 'System started'})
    else:
        return jsonify({'status': 'System already running'})

# API endpoint to stop the system
@app.route('/stop', methods=['POST'])
def stop_system():
    global system_active, stop_threads
    system_active = False
    stop_threads = True
    return jsonify({'status': 'System stopped'})

# API endpoint to upload videos
@app.route('/upload', methods=['POST'])
def upload_video():
    global vehicle_video_path, pedestrian_video_path
    
    video_type = request.form.get('type')
    file = request.files['file']
    
    if video_type == 'vehicle':
        vehicle_video_path = f"uploads/vehicle_{int(time.time())}.mp4"
        file.save(vehicle_video_path)
    elif video_type == 'pedestrian':
        pedestrian_video_path = f"uploads/pedestrian_{int(time.time())}.mp4"
        file.save(pedestrian_video_path)
    
    return jsonify({'status': 'success'})

# Helper function to convert frame to base64
def frame_to_base64(frame):
    if frame is not None:
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
    return None

# Your original functions with minimal adaptations for Flask
def process_vehicle_feed():
    global vehicle_count, vehicle_frame, stop_threads, vehicle_video_path
    
    cap = cv2.VideoCapture(vehicle_video_path if vehicle_video_path else 0)
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened() and not stop_threads:
        pause_vehicle.wait()  # Pause vehicle video if event is cleared

        ret, frame = cap.read()
        if not ret:
            break

        results = vehicle_model(frame, verbose=False)[0]
        count = 0
        for box in results.boxes:
            cls = vehicle_model.names[int(box.cls[0])]
            if cls in vehicle_classes:
                count += 1
        vehicle_count = count
        vehicle_frame = results.plot()

        time.sleep(1 / fps)

    cap.release()

def process_pedestrian_feed():
    global pedestrian_count, pedestrian_frame, stop_threads, pedestrian_video_path
    
    cap = cv2.VideoCapture(pedestrian_video_path if pedestrian_video_path else 0)
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened() and not stop_threads:
        ret, frame = cap.read()
        if not ret:
            break

        height = frame.shape[0]
        results = pedestrian_model(frame, verbose=False)[0]
        count = 0
        for box in results.boxes:
            cls = pedestrian_model.names[int(box.cls[0])]
            ymin = int(box.xyxy[0][1])
            if cls == pedestrian_class and ymin > height * 0.65:
                count += 1
        pedestrian_count = count
        pedestrian_frame = results.plot()

        time.sleep(1 / fps)

    cap.release()

def traffic_control_loop():
    global vehicle_count, pedestrian_count, signal_status, stop_threads
    
    while not stop_threads:
        signal_status = "\n--- TRAFFIC SIGNAL STATUS ---\n"
        signal_status += f"Detected Vehicles: {vehicle_count} | Pedestrians in Zone: {pedestrian_count}\n"

        if pedestrian_count >= pedestrian_threshold:
            signal_status += "Signal: GREEN for Pedestrians | RED for Vehicles\n"
            pause_vehicle.clear()  # Pause vehicle feed
            countdown = min(pedestrian_count * 2, 30)
            for t in range(countdown, 0, -1):
                signal_status = f"Pedestrian crossing in progress... {t} sec remaining"
                time.sleep(1)
            pause_vehicle.set()

        elif vehicle_count < 5 and pedestrian_count > 0:
            signal_status += "Low vehicle density. Giving GREEN to Pedestrians for 20 seconds.\n"
            pause_vehicle.clear()
            for t in range(20, 0, -1):
                signal_status = f"Pedestrian crossing (low traffic)... {t} sec remaining"
                time.sleep(1)
            pause_vehicle.set()

        elif vehicle_count >= vehicle_threshold:
            signal_status += "Traffic Congestion Detected. Forcing GREEN for Vehicles!\n"
            pause_vehicle.set()
            for t in range(15, 0, -1):
                signal_status = f"Vehicles passing... {t} sec remaining"
                time.sleep(1)

        else:
            signal_status += "Normal Flow: GREEN for Vehicles | RED for Pedestrians\n"
            pause_vehicle.set()
            for t in range(6, 0, -1):
                signal_status = f"Normal flow... {t} sec remaining"
                time.sleep(1)

        signal_status += "\nCycle complete.\n"
        time.sleep(1)

if __name__ == '__main__':
    import os
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, threaded=True)