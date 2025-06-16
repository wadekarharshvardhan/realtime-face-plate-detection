# Face & Number Plate Recognition System

A real-time Python application that detects and recognizes faces and vehicle number plates from a webcam feed.  
- **Face Recognition:** Uses FaceNet (facenet-pytorch) for accurate face detection and recognition.
- **Number Plate Recognition:** Uses Haar Cascade for plate detection and OCR for plate text extraction.
- **Live Display:** Draws bounding boxes and labels for both faces and number plates on the video stream.
- **Database Logging:** All recognized faces and number plates are logged with timestamps in an SQLite database.

---

## Features

- 🚗 **Automatic Number Plate Detection & Recognition**
- 😃 **Face Detection & Recognition with Deep Learning**
- 🖼️ **Real-time Video Processing**
- 🗃️ **SQLite Logging of All Events**
- 🖥️ **Modular, Clean, and Extensible Codebase**
- 🛡️ **Privacy-First:** No personal images included; users add their own.

---

## Demo

https://ibb.co/wZHnsWHJ (Face detection)
https://ibb.co/1fF83h6J (Number PLate detection)
https://ibb.co/dsf7hmVr (Logs)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/face-number-plate-recognition.git
cd face-number-plate-recognition/src
```

### 2. Install Dependencies

It is recommended to use a virtual environment.

```bash
pip install -r ../requirements.txt
```

**Key dependencies:**
- `opencv-python`
- `facenet-pytorch`
- `torch`
- `numpy`
- `easyocr` or `pytesseract` (depending on your OCR backend)

### 3. Prepare Known Faces

- Place clear, front-facing images of people you want to recognize in the `known_faces` folder.
- Name each image as the person's name (e.g., `alice.jpg`, `bob.jpg`).
- **No personal images are included in this repo for privacy.**

### 4. Prepare Haar Cascade

- The repo includes `haarcascade_russian_plate_number.xml` for number plate detection.
- You may use your own cascade for better results in your region.

---

## Usage

```bash
python main.py
```

- Press `q` to quit the application.

---

## Viewing and Exporting Logs

To view logs in the terminal:
```bash
python view_logs.py
```

To export logs to CSV, create a file `export_logs.py` as shown in the documentation.

---

## Folder Structure

```
project-root/
│
├─ src/
│   ├─ main.py
│   ├─ detector.py
│   ├─ ocr.py
│   ├─ camera.py
│   ├─ logger.py
│   ├─ view_logs.py
│   ├─ haarcascade_russian_plate_number.xml
│   └─ ...
│
├─ known_faces/
│   ├─ test.jpg
│   └─ .gitkeep
│
├─ requirements.txt
└─ README.md
```

---

## How It Works

- **Face Recognition:**  
  Loads known faces, extracts embeddings using FaceNet, and compares live webcam faces for recognition.

- **Number Plate Recognition:**  
  Detects plates using Haar Cascade, crops and preprocesses the plate, then extracts text using OCR.

- **Display:**  
  Draws a green box and name for faces, a red box and formatted number for plates.  
  If no plate is detected, displays "No Number Plate Detected".

- **Logging:**  
  Every recognized face and number plate is logged with a timestamp in `logs.db`.  
  Use `view_logs.py` to view logs.

---

## Customization

- **Add More Known Faces:**  
  Add more images to `known_faces` and restart the app.

- **Change OCR Backend:**  
  Modify `ocr.py` to use EasyOCR or Tesseract as needed.

- **Improve Plate Detection:**  
  Replace the Haar Cascade with a deep learning detector for better accuracy.

---

## Troubleshooting

- **Laggy Video:**  
  Lower the webcam resolution or process every Nth frame for better performance.
- **No Face Detected:**  
  Ensure your `known_faces` images are clear and front-facing.
- **No Number Plate Detected:**  
  Make sure the cascade XML is present and the plate is visible to the camera.
- **No logs shown:**  
  Run `main.py` at least once to create the database and table.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
- [OpenCV](https://opencv.org/)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) / [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

---

## Contact

For questions or suggestions, open an issue or contact [harshvardhan.hudhp22@sinhgad.edu](mailto:harshvardhan.hudhp22@sinhgad.edu).