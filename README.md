<p align="center">
  <img src="https://raw.githubusercontent.com/afzalt3ch/banner.png/main/Gemini_Generated_Image_hb31fqhb31fqhb31.png" alt="WeeBee Banner" width="100%" />
</p>

# 🤟 SignToText – Real-Time Sign Language Detection

SignToText is a real-time sign language recognition web app that detects hand gestures using a webcam and translates them into text. It combines a frontend powered by TensorFlow.js Handpose model and a backend powered by Flask + PyTorch for deep learning inference.

---

## 🖼️ Screenshot

![App Screenshot](https://github.com/afzalt3ch/Sign-To-Text/blob/main/screenshots/sign-to-text-demo.jpeg)

---

## ✨ Features

- 📷 Real-time webcam feed using TensorFlow.js Handpose
- ✋ Automatic image capture when a hand is detected
- 🔄 Undo, Reset, Start, and Stop buttons for user control
- 🔤 Translates detected hand signs into alphabets and builds words
- 🤖 PyTorch backend model (EfficientNet) for sign prediction
- 🌐 Clean UI with responsive layout and animations

---

## 🛠️ Tech Stack

- **Frontend:** HTML, CSS, JavaScript, jQuery, TensorFlow.js, Handpose
- **Backend:** Python, Flask, PyTorch, EfficientNet
- **Model:** Trained EfficientNet model for static gesture classification

---

## 📁 Folder Structure

```
sign-to-text/
├── model/
│   └── efficientnet_model.pth          # Trained PyTorch model
├── templates/
│   └── index.html                      # Frontend UI
├── static/                             # (Optional for CSS/JS if added)
├── app.py                              # Flask backend API
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/afzalt3ch/Sign-To-Text.git
cd Sign-To-Text
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Flask backend

```bash
python app.py
```

### 4. Open the app

Visit [http://localhost:5000](http://localhost:5000) in your browser.  
Grant webcam access to begin signing!

---

## 🧠 Model Info

The backend uses a custom-trained **EfficientNet-B0** model to classify images of hand signs captured from the frontend webcam stream.

---

## 📦 Requirements

See `requirements.txt`:

```txt
torch
torchvision
efficientnet_pytorch
flask
Pillow
numpy
```

---

## 📜 License

MIT License

---

<p align="center">Made with ❤️ by <strong>Afzal T3ch</strong></p>
