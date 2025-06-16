import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize MTCNN and FaceNet
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=40, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def load_known_faces(known_dir, max_images=10):
    known_embeddings = []
    known_names = []
    # Map filenames to actual names
    name_mapping = {
        '1': 'Aditya',
        '2': 'Harshvardhan'
    }
    
    count = 0
    for filename in sorted(os.listdir(known_dir)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            if count >= max_images:
                break
            file_name = os.path.splitext(filename)[0]
            # Use mapping if available, otherwise use filename
            name = name_mapping.get(file_name, file_name)
            img_path = os.path.join(known_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face = mtcnn(img_rgb)
            if face is not None:
                print(f"Loaded face: {name}")
                embedding = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()
                known_embeddings.append(embedding)
                known_names.append(name)
                count += 1
            else:
                print(f"No face detected in: {name}")
    print(f"Total faces loaded: {len(known_names)}")
    return known_names, known_embeddings

def recognize_face_and_box(frame, known_names, known_embeddings, threshold=1.8):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(img_rgb)
    name = "Unknown"
    box = None
    
    if boxes is not None:
        box = boxes[0].astype(int)
        # Use mtcnn directly instead of extract
        face = mtcnn(img_rgb)
        if face is not None:
            print("Face extracted successfully")  # Debug print
            # Face is already in correct format from mtcnn
            embedding = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()
            distances = [np.linalg.norm(embedding - k_emb) for k_emb in known_embeddings]
            
            if distances:
                min_dist = min(distances)
                best_match_idx = np.argmin(distances)
                best_match_name = known_names[best_match_idx]
                
                # Print distance information
                print(f"Distances: {[f'{d:.3f}' for d in distances]}")
                print(f"Best match: {best_match_name} (distance: {min_dist:.3f})")
                print(f"Current threshold: {threshold}")
                print(f"Suggested threshold for recognition: {min_dist + 0.1:.3f}")
                print("-" * 40)
                
                if min_dist < threshold:
                    name = best_match_name
                    print(f"✓ RECOGNIZED: {name}")
                else:
                    print(f"✗ NOT RECOGNIZED - Unknown person")
            else:
                print("No distances computed")  # Debug print
        else:
            print("Face extraction failed")  # Debug print
    else:
        print("No faces detected")  # Debug print
    
    return name, box

def main():
    known_faces_dir = os.path.join(os.path.dirname(__file__), "known_faces")
    known_names, known_embeddings = load_known_faces(known_faces_dir, max_images=10)
    
    if not known_names:
        print("No faces loaded! Add images to known_faces folder.")
        return
    
    cam = cv2.VideoCapture(0)
    frame_count = 0
    last_name, last_box = "Unknown", None

    while True:
        ret, frame = cam.read()
        if not ret:
            break

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

        cv2.imshow('FaceNet Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()




