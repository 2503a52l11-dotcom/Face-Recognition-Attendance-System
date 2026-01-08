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

⚒️Workflow

1.Enter names of students
2.System captures face images automatically
3.Model is trained
4.Live face recognition starts
5.Attendance is marked automatically

📈 Future Enhancements

1.Cloud-based attendance storage
2.Face recognition accuracy improvement using deep learning
3.Mobile app integration
4.Mask & spoof detection
---

📸 Screenshots
1.Face Capture<img width="821" height="647" alt="face_capture png" src="https://github.com/user-attachments/assets/81b15973-61f0-4be2-9302-f01082d51a0a" />

2.Live Recognition<img width="796" height="647" alt="live_recognition png" src="https://github.com/user-attachments/assets/b551608f-8c49-447c-9e8d-c3d55595df5e" />

3.Attendance CSV Output<img width="1911" height="1020" alt="attendance_csv png" src="https://github.com/user-attachments/assets/3f35831a-e3b8-492a-a1da-30480784d959" />


## ▶️ How to Run the Project

### 1️⃣ Install Required Libraries
```bash
pip install opencv-python numpy pandas

#to  run the program
python attendance_system_classroom.py


