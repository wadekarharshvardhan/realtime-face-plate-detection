import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import os
import re
from detector import NumberPlateDetector
from ocr import OCRProcessor
from camera import Camera
from logger import Logger  # <-- Add this import

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize MTCNN and FaceNet
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=40, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def format_plate_number(plate):
    # Try to match and format Indian number plates like MH12ER3445 -> MH 12 ER 3445
    match = re.match(r'^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})$', plate.replace(' ', ''))
    if match:
        return ' '.join(match.groups())
    return plate

def load_known_faces(known_dir, max_images=10):
    known_embeddings = []
    known_names = []
    for filename in sorted(os.listdir(known_dir)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(filename)[0]
            img_path = os.path.join(known_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face = mtcnn(img_rgb)
            if face is not None:
                embedding = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()
                known_embeddings.append(embedding)
                known_names.append(name)
    return known_names, known_embeddings

def recognize_face_and_box(frame, known_names, known_embeddings, threshold=1.8):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(img_rgb)
    name = "Unknown"
    box = None
    if boxes is not None:
        box = boxes[0].astype(int)
        face = mtcnn(img_rgb)
        if face is not None:
            embedding = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()
            distances = [np.linalg.norm(embedding - k_emb) for k_emb in known_embeddings]
            if distances:
                min_dist = min(distances)
                best_match_idx = np.argmin(distances)
                best_match_name = known_names[best_match_idx]
                if min_dist < threshold:
                    name = best_match_name
    return name, box

def main():
    # --- Load known faces ---
    known_faces_dir = os.path.join(os.path.dirname(__file__), "known_faces")
    known_names, known_embeddings = load_known_faces(known_faces_dir, max_images=10)
    if not known_names:
        print("No faces loaded! Add images to known_faces folder.")
        return

    detector = NumberPlateDetector()
    ocr_processor = OCRProcessor()
    cam = Camera()
    logger = Logger()  # <-- Initialize logger

    last_plate_coords = None
    last_plate_number = "No Number Plate Detected"
    stable_count = 0
    not_found_count = 0
    STABLE_THRESHOLD = 3
    NOT_FOUND_RESET = 10  # Number of frames to wait before resetting if plate is lost

    frame_count = 0
    last_name, last_box = "Unknown", None

    while True:
        frame = cam.get_frame()
        if frame is None:
            break

        # --- FaceNet Recognition (every 3rd frame for speed) ---
        frame_count += 1
        if frame_count % 3 == 0:
            name, box = recognize_face_and_box(frame, known_names, known_embeddings, threshold=1.8)
            if box is not None:
                last_name, last_box = name, box
            else:
                last_name, last_box = "Unknown", None

        if last_box is not None:
            x1, y1, x2, y2 = last_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, last_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # --- Number Plate Detection ---
        plate_image, plate_coords = detector.detect_plate(frame)

        # Stabilize detection
        if plate_coords is not None:
            if last_plate_coords is not None and np.linalg.norm(np.array(plate_coords) - np.array(last_plate_coords)) < 20:
                stable_count += 1
            else:
                stable_count = 1
            last_plate_coords = plate_coords
            not_found_count = 0
        else:
            stable_count = 0
            not_found_count += 1

        # If plate is stable, update last_plate_number
        if plate_image is not None and hasattr(plate_image, "size") and plate_image.size > 0 and stable_count >= STABLE_THRESHOLD:
            plate_image = cv2.resize(plate_image, (200, 64))
            text = ocr_processor.extract_text(plate_image)
            match = re.search(r'[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{1,4}', text.replace('\n', '').replace('\r', ''))
            if match:
                last_plate_number = match.group(0)
            else:
                last_plate_number = text if text else last_plate_number
            last_plate_number = format_plate_number(last_plate_number)
            not_found_count = 0

        # If plate not found for several frames, reset
        if not_found_count > NOT_FOUND_RESET:
            last_plate_coords = None
            last_plate_number = "No Number Plate Detected"

        # Draw box and number if we have a last known plate
        if last_plate_coords is not None:
            x, y, w, h = last_plate_coords
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, last_plate_number, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Always display the plate number at the top-left
        cv2.putText(frame, last_plate_number, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # --- Log to database ---
        logger.log(last_name, last_plate_number)

        cv2.imshow('Face & Number Plate Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    logger.close()  # <-- Close logger
    del cam
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()