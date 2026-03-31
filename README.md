# 🌿 Plant Disease Detection System

An AI-powered web application that detects plant leaf diseases using a Convolutional Neural Network (CNN). The system allows users to upload leaf images and get instant predictions with confidence scores.

---

## 🚀 Features

* 📷 Upload leaf images through web interface
* 🤖 Detect diseases:

  * Healthy
  * Early Blight
  * Late Blight
* 📊 Displays prediction confidence
* 🌐 Interactive web app built with Flask

---

## 🧠 Tech Stack

* Python
* TensorFlow / Keras
* Flask
* NumPy, OpenCV
* Matplotlib

---

## 📊 Model Details

* Convolutional Neural Network (CNN)
* Trained on the **PlantVillage dataset**
* Achieved ~85–90% accuracy

---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Then open:

http://127.0.0.1:5000

---

## 📁 Project Structure

```
├── app.py
├── predict.py
├── templates/
├── static/
├── plant_disease_detection.ipynb
├── requirements.txt
└── setup_dataset.py
```

---

## ⚠️ Limitations

* Works best on dataset-like images
* Accuracy may drop for real-world images due to:

  * lighting variations
  * complex backgrounds
  * different camera angles

---

## 📚 Dataset

PlantVillage Dataset (available on Kaggle)

---

## 👨‍💻 Author

**Utsav Kumar**

---

## ⭐ Acknowledgements

* PlantVillage Dataset
* TensorFlow & Keras community
* Open-source contributors
