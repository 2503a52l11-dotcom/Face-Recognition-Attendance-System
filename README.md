# 🎯 Face Recognition Attendance System

A real-time **face recognition–based automated attendance system** built using **Python and OpenCV**.  
The system captures faces through a webcam, trains a recognition model, identifies individuals in live video, and automatically records attendance with timestamps.

---

## 🚀 Features

- 📸 Real-time face detection using Haar Cascade Classifier
- 🧠 Face recognition using **LBPH (Local Binary Pattern Histogram)**
- ⚡ Automatic face image capture (20 images per person)
- 🔄 Continuous auto-learning for new faces
- 🕒 Attendance marked with timestamp
- 📄 Attendance stored in CSV format
- 🎥 Live webcam-based recognition
- 🧑‍🤝‍🧑 Supports multiple users

---

## 🛠️ Technologies Used

- **Programming Language:** Python  
- **Libraries:** OpenCV, NumPy, Pandas  
- **Computer Vision:** Haar Cascade  
- **Face Recognition Algorithm:** LBPH  
- **Data Storage:** CSV  

---

## 📁 Project Structure

Face_Attendance/
├── dataset/ # Face images (ignored in GitHub)
├── embeddings/ # Trained model & labels
├── attendance/ # Attendance CSV files
├── attendance_system_classroom.py
├── collect_faces.py
├── train_embeddings_opencv.py
├── recognize_attendance_opencv.py
├── README.md
└── .gitignore


---

## ▶️ How to Run the Project

### 1️⃣ Install Required Libraries
```bash
pip install opencv-python numpy pandas

#to  run the program
python attendance_system_classroom.py


