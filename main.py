from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import os
import cv2
import threading
import logging
from virtual_tryon import detect_eyes_and_forehead, overlay_clothing
import cvzone
from cvzone.PoseModule import PoseDetector

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Folder paths for accessories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
hat_folder = os.path.join(DATA_DIR, "hat")
glasses_folder = os.path.join(DATA_DIR, "glasses")
shirt_folder = os.path.join(DATA_DIR, "Shirts")

# Helper function to load images
def load_images_from_folder(folder):
    return sorted(
        [f for f in os.listdir(folder) if f.endswith(('png', 'jpeg', 'jpg'))]
    )

# Load accessories dynamically
if not all(os.path.exists(folder) for folder in [hat_folder, glasses_folder, shirt_folder]):
    raise FileNotFoundError("Ensure hat, glasses, and shirt folders exist in the 'data' directory.")

hat_files = load_images_from_folder(hat_folder)
glasses_files = load_images_from_folder(glasses_folder)
shirt_files = load_images_from_folder(shirt_folder)

if not (hat_files and glasses_files and shirt_files):
    raise FileNotFoundError("Ensure hat, glasses, and shirt images exist in their respective folders.")

hats = [cv2.imread(os.path.join(hat_folder, f), cv2.IMREAD_UNCHANGED) for f in hat_files]
glasses_list = [cv2.imread(os.path.join(glasses_folder, f), cv2.IMREAD_UNCHANGED) for f in glasses_files]
shirts = [cv2.imread(os.path.join(shirt_folder, f), cv2.IMREAD_UNCHANGED) for f in shirt_files]

# Load classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)
detector = PoseDetector()

# Initialize variables for selecting accessories
hat_index = 0
glasses_index = 0
shirt_index = 0

def overlay_shirt(frame, lmList, bboxInfo, shirt):
    """Overlay shirt on detected person using pose keypoints."""
    if bboxInfo:
        bbox = bboxInfo["bbox"]
        center = bboxInfo["center"]

        shirt_width = int(bbox[2] * 0.8)  # 80% of person's width
        shirt_height = int(shirt_width * 581 / 440)  # Maintain aspect ratio

        x1 = max(0, center[0] - shirt_width // 2)
        y1 = max(0, bbox[1] + bbox[3] // 6)  # Start from 1/6th of the body height

        shirt_resized = cv2.resize(shirt, (shirt_width, shirt_height))
        frame = cvzone.overlayPNG(frame, shirt_resized, (x1, y1))
    return frame

def generate_frames():
    """Generate video frames and apply virtual try-on."""
    global cap, hat_index, glasses_index, shirt_index
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Detect pose
        frame = detector.findPose(frame, draw=False)
        lmList, bboxInfo = detector.findPosition(frame, bboxWithHands=False, draw=False)

        # Detect eyes and forehead
        eyes, forehead = detect_eyes_and_forehead(frame, face_cascade, eye_cascade)

        # Overlay accessories (glasses, hat, shirt)
        frame = overlay_clothing(frame, eyes, forehead, hats[hat_index], glasses_list[glasses_index])
        frame = overlay_shirt(frame, lmList, bboxInfo, shirts[shirt_index])

        # Encode frame in JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/')
def index():
    """Render home page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Stream video feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_items')
def get_items():
    """Dynamically fetch accessory file paths."""
    return jsonify({
        'hats': [f'/data/hat/{file}' for file in hat_files],
        'glasses': [f'/data/glasses/{file}' for file in glasses_files],
        'shirts': [f'/data/Shirts/{file}' for file in shirt_files]
    })

@app.route('/change_hat', methods=['POST'])
def change_hat():
    """Change hat accessory."""
    global hat_index
    try:
        index = int(request.form.get('hatIndex', 0))
        if 0 <= index < len(hats):
            hat_index = index
            logging.info(f"Hat index changed to {hat_index}")
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'error', 'message': 'Invalid hat index'}), 400
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Invalid input'}), 400

@app.route('/change_glasses', methods=['POST'])
def change_glasses():
    """Change glasses accessory."""
    global glasses_index
    try:
        index = int(request.form.get('glassesIndex', 0))
        if 0 <= index < len(glasses_list):
            glasses_index = index
            logging.info(f"Glasses index changed to {glasses_index}")
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'error', 'message': 'Invalid glasses index'}), 400
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Invalid input'}), 400

@app.route('/change_shirt', methods=['POST'])
def change_shirt():
    """Change shirt accessory."""
    global shirt_index
    try:
        index = int(request.form.get('shirtIndex', 0))
        if 0 <= index < len(shirts):
            shirt_index = index
            logging.info(f"Shirt index changed to {shirt_index}")
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'error', 'message': 'Invalid shirt index'}), 400
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Invalid input'}), 400

@app.route('/data/<path:filename>')
def static_data(filename):
    """Serve static data files."""
    return send_from_directory('data', filename)

if __name__ == "__main__":
    # Set port to 5000 explicitly for local development
    app.run(host="127.0.0.1", port=5000, debug=True)

