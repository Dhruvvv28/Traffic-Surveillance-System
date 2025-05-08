
# 🚦 Traffic Surveillance System using Deep Learning

This project presents a deep learning-based **Traffic Surveillance System** that classifies real-world traffic scenarios into predefined categories using Convolutional Neural Networks (CNNs). The system is designed to assist traffic management authorities by automatically identifying critical road conditions from images or video frames.

---

## 📌 Features

- Classifies traffic scenes into 8 categories:
  1. Accident  
  2. Dense Traffic  
  3. Fire  
  4. Improper Parking  
  5. Road Construction  
  6. Traffic Signal Failure  
  7. Unplanned Events / Gatherings  
  8. Vehicle Breakdown  
- Real-time classification through a user-friendly GUI
- Visualization of prediction confidence
- Trained using TensorFlow and Keras with custom CNN architecture
- Incorporates result screenshots for validation

---

## 🧠 Model Architecture

- **Layers Used**:  
  - Convolutional Layers with ReLU activation  
  - Max Pooling Layers  
  - Dropout Layers for regularization  
  - Fully Connected Dense Layers  
  - Softmax Output Layer for multi-class classification  

- **Training Details**:  
  - Optimizer: Adam  
  - Loss Function: Categorical Crossentropy  
  - Epochs: 20–30  
  - Evaluation Metrics: Accuracy, Precision, Recall, F1-Score  

---

## 🗃️ Dataset Information

- Total Images: **16,208**
- 8 Classes (Balanced)
- Sources: OpenCV Public Repositories, CrowdAI Datasets, Augmented Synthetic Images
- Preprocessing: Resized to 150x150, normalized pixel values

---

## 🏗️ System Architecture

- Input Layer (Image from user)
- Preprocessing (Resizing, normalization)
- Feature Extraction via CNN
- Classification (Dense + Softmax Layer)
- Output Layer (Class label + confidence)
- GUI for interaction and result display

---

## ⚙️ Tools & Technologies Used

- **Programming Language**: Python  
- **Libraries**: TensorFlow, Keras, NumPy, OpenCV, scikit-learn, Tkinter  
- **IDE**: Jupyter Notebook / VS Code  
- **GUI Framework**: Tkinter  

---

## 📁 Project Structure

```
TrafficSurveillanceSystem/
├── dataset/                  # Image dataset for training/testing
├── model/                    # Trained model (.h5)
├── gui/                      # GUI application scripts
├── results/                  # Screenshots of predictions
├── train_model.py            # Model training script
├── main.py                   # Main GUI interface
├── utils.py                  # Utility functions
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

---

## 🚀 How to Run

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Train the Model (Optional)
```bash
python train_model.py
```

### 3. Run the Application
```bash
python main.py
```

---

## 🖼️ Sample Results

Actual classification results with GUI screenshots are included in the `/results/` folder. These showcase model predictions with confidence scores for each of the 8 traffic scenarios.

---

## 🔮 Future Enhancements

- Integration with live CCTV video streams
- Real-time object detection with bounding boxes
- Deployment on cloud or embedded devices
- Alert generation for high-risk events

---

## 💡 Applications

- Smart city traffic monitoring
- Emergency response alert systems
- Automated surveillance for metro and highways
- Data analytics for urban planning and congestion management

