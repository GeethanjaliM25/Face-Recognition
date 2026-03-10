👤 Face Recognition System with DeepFace
https://img.shields.io/badge/Python-3.8%252B-blue
https://img.shields.io/badge/Streamlit-1.28%252B-red
https://img.shields.io/badge/DeepFace-0.0.79-green
https://img.shields.io/badge/OpenCV-4.8%252B-orange
https://img.shields.io/badge/License-MIT-yellow
https://img.shields.io/badge/Maintained%253F-yes-green.svg
https://img.shields.io/badge/PRs-welcome-brightgreen.svg
https://img.shields.io/badge/TensorFlow-2.0%252B-orange
https://img.shields.io/badge/Keras-3.0%252B-red

A professional face recognition system built with DeepFace and Streamlit, featuring real-time face registration and recognition capabilities. This application allows you to register faces and identify individuals with high accuracy using state-of-the-art deep learning models.

📋 Features
Face Registration: Register new individuals with their photos

Real-time Recognition: Upload images and instantly recognize faces

Multiple Deep Learning Models: Choose from VGG-Face, Facenet, OpenFace, or DeepFace

Adjustable Threshold: Fine-tune recognition sensitivity

Visual Feedback: Clear success/error messages for recognition results

Session Management: Maintain registered faces during the session

User-friendly Interface: Clean tabbed interface for easy navigation

🚀 Quick Start
Prerequisites
Python 3.8 or higher

pip package manager

Git (optional)

Installation
Clone the repository

bash
git clone https://github.com/yourusername/face-recognition-deepface.git
cd face-recognition-deepface
Install required packages

bash
pip install -r requirements.txt
Run the application

bash
streamlit run app.py
Open your browser and navigate to http://localhost:8501

📦 Requirements
Create a requirements.txt file with:

txt
streamlit>=1.28.0
deepface>=0.0.79
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
tensorflow>=2.0.0
tf-keras>=2.0.0
🎯 How It Works
Face Registration
Enter the person's name

Upload a clear photo of their face

DeepFace extracts a 128-dimensional face embedding

The embedding is stored in the session state

Face Recognition
Upload a photo to recognize

DeepFace extracts the face embedding

System compares with all registered faces

Calculates Euclidean distances and similarity scores

Returns the best match if similarity exceeds threshold

🖥️ Usage Guide
Registering a Face
Navigate to the "📝 Register" tab

Enter the person's name in the text field

Upload a clear photo (JPG, JPEG, PNG)

Click the "Register" button

Wait for the success message

Recognizing a Face
Navigate to the "🔍 Recognize" tab

Ensure you have registered at least one face

Upload a photo to recognize

View the recognition result:

✅ Success: Shows the person's name and confidence

❌ Unknown: Indicates the person is not registered

Viewing Registered Faces
Navigate to the "📋 List" tab

See all registered individuals

⚙️ Configuration
Model Selection
Choose from four deep learning models:

Model	Accuracy	Speed	Description
Facenet (Default)	⭐⭐⭐⭐⭐	⭐⭐⭐	Google's FaceNet - Best balance
VGG-Face	⭐⭐⭐⭐	⭐⭐	Oxford's VGG - High accuracy
OpenFace	⭐⭐⭐	⭐⭐⭐⭐	Open source - Faster but less accurate
DeepFace	⭐⭐⭐⭐	⭐⭐	Facebook's DeepFace - Good accuracy
Threshold Adjustment
Range: 0.2 to 0.8

Default: 0.4



🤝 Contributing
Contributions are welcome! Please follow these steps:

Fork the repository

Create a feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

Development Guidelines
Follow PEP 8 style guide

Add comments for complex logic

Update documentation for new features

Write unit tests when applicable

Ensure backward compatibility

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

👏 Acknowledgments
DeepFace library by Sefik Ilkin Serengil

Streamlit for the amazing web framework

FaceNet for the pre-trained models

OpenCV for computer vision capabilities

📧 Contact
geethanjalishetty34@gmail.com
Lower values: Stricter matching (more false rejections)

Higher values: Looser matching (more false acceptances)
