# 🔥 Fire Detection System

## 📌 Project Overview
This project is an **AI-powered real-time monitoring system** that detects **fires** using YOLOv8 and Llava-Phi3 Vision. When an incident is detected, the system sends an alert via **Telegram**, along with an image or a short video of the event.

## 🚀 Features
- **Fire detection** using a custom-trained YOLOv8 model.
- **Verification with Llava-Phi3 Vision** to reduce false positives.
- **Alerts via Telegram**, including:
  - A warning message.
  - An image of the detected event.
  - A short video clip.
- **Real-time processing** with OpenCV.
- **Multithreaded execution** to handle detection efficiently.

## 📂 Project Structure
```
📁 FireFallDetection
│── fire_and_fall_test.py   # Main script for fire and fall detection
│── fire_YOLOv8.pt          # Trained YOLOv8 model for fire detection
│── config.py               # Configuration file (Telegram API keys, settings, etc.)
│── requirements.txt        # Dependencies
│── .gitignore              # Files to be ignored by Git
```

## 🛠️ Installation & Setup
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)
```sh
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 4️⃣ Set Up Configuration
- Rename `config_example.py` to `config.py`.
- Open `config.py` and add your **Telegram API Key** and **Chat ID**.

### 5️⃣ Run the System
```sh
python fire_and_fall_test.py
```

## 📡 How It Works
1. **YOLOv8 detects fire or falls in real-time**.
2. **If a fire is detected, Llava-Phi3 Vision verifies it** to reduce false alarms.
3. **If confirmed, an alert is sent to Telegram** with an image and video of the event.
4. The system **resets after 30 seconds** to allow new detections.

## 📝 Customization
- **Thresholds:** Adjust the confidence level for detections in the script.
- **Alert Delay:** Modify the alert cooldown period.
- **Video Clip Length:** Adjust the duration of the recorded video.

## 🔧 Troubleshooting
- If you get a `Bad Request: chat not found` error in Telegram, make sure you have started a conversation with your bot first.
- If `YOLOv8` is not working, ensure the model files (`.pt`) are correctly loaded.
- If **performance is slow**, consider reducing the frame resolution or increasing hardware acceleration.

## 📜 License
This project is **open-source** under the MIT License.

## 🤝 Contributing
Feel free to submit **issues** or **pull requests** if you want to improve the system!

## 📬 Contact
If you have any questions, reach out via GitHub issues or Telegram.
