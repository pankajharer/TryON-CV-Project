import cv2
import os
import numpy as np


def detect_eyes_and_forehead(frame, face_cascade, eye_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    eyes = []
    forehead = None

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        detected_eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

        for (ex, ey, ew, eh) in detected_eyes:
            eye_center = (x + ex + ew // 2, y + ey + eh // 2)
            eyes.append(eye_center)

        forehead = (x + w // 2, y - h // 6)
        break

    return eyes, forehead


def overlay_clothing(frame, eyes, forehead, hat, glasses):
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[0])
        left_eye, right_eye = eyes[:2]

        glasses_width = int(abs(right_eye[0] - left_eye[0]) * 2.0)
        glasses_height = int(glasses_width * glasses.shape[0] / glasses.shape[1])
        glasses_x = left_eye[0] - glasses_width // 4
        glasses_y = left_eye[1] - glasses_height // 2

        frame = overlay_image(frame, glasses, glasses_x, glasses_y, glasses_width, glasses_height)

    if forehead:
        hat_width = int(abs(right_eye[0] - left_eye[0]) * 2.5) if len(eyes) >= 2 else 150
        hat_height = int(hat_width * hat.shape[0] / hat.shape[1])
        hat_x = forehead[0] - hat_width // 2
        hat_y = forehead[1] - hat_height

        frame = overlay_image(frame, hat, hat_x, hat_y, hat_width, hat_height)

    return frame


def overlay_image(background, overlay, x, y, width, height):
    overlay = cv2.resize(overlay, (width, height))

    if x < 0:
        overlay = overlay[:, -x:]
        x = 0
    if y < 0:
        overlay = overlay[-y:, :]
        y = 0
    if x + overlay.shape[1] > background.shape[1]:
        overlay = overlay[:, :background.shape[1] - x]
    if y + overlay.shape[0] > background.shape[0]:
        overlay = overlay[:background.shape[0] - y, :]

    if overlay.shape[2] == 4:
        alpha_channel = overlay[:, :, 3] / 255.0
        overlay_rgb = overlay[:, :, :3]

        for c in range(3):
            background[y:y + overlay.shape[0], x:x + overlay.shape[1], c] = \
                alpha_channel * overlay_rgb[:, :, c] + \
                (1 - alpha_channel) * background[y:y + overlay.shape[0], x:x + overlay.shape[1], c]

    return background


def process_image(input_path, output_path):
    """
    Processes an uploaded image, applies virtual try-on (hats, glasses, shirts), and saves the output.
    """
    frame = cv2.imread(input_path)
    if frame is None:
        raise FileNotFoundError("Input image not found.")

    # Accessory folders
    hat_folder = "data/hat"
    glasses_folder = "data/glasses"

    # Load accessories
    hat_files = sorted([f for f in os.listdir(hat_folder) if f.endswith(('png', 'jpeg', 'jpg'))])
    glasses_files = sorted([f for f in os.listdir(glasses_folder) if f.endswith(('png', 'jpeg', 'jpg'))])

    if not hat_files or not glasses_files:
        raise FileNotFoundError("Ensure hat and glasses images exist at the specified paths.")

    # Load the first hat and glasses as an example
    hat = cv2.imread(os.path.join(hat_folder, hat_files[0]), cv2.IMREAD_UNCHANGED)
    glasses = cv2.imread(os.path.join(glasses_folder, glasses_files[0]), cv2.IMREAD_UNCHANGED)

    # Load classifiers for face and eyes
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Detect eyes and forehead
    eyes, forehead = detect_eyes_and_forehead(frame, face_cascade, eye_cascade)

    # Overlay hat and glasses
    frame = overlay_clothing(frame, eyes, forehead, hat, glasses)

    # Save the result
    cv2.imwrite(output_path, frame)


if __name__ == "__main__":
    # Example usage for standalone testing
    process_image("example_input.jpg", "example_output.jpg")
    print("Image processing complete. Check the output file.")
