import cv2
import os
import threading
import time
import requests
import numpy as np
from ultralytics import YOLO
import ollama
from dotenv import load_dotenv

load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    response = requests.post(url, data=data)
    return response.status_code == 200

def send_telegram_image(image_path):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(image_path, "rb") as image:
        files = {"photo": image}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": "📷 Situation image."}
        response = requests.post(url, files=files, data=data)
    return response.status_code == 200

# Load YOLO model
yolo_model = YOLO('fire_YOLOv8.pt')
cap = cv2.VideoCapture("videos/fire2.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
fire_detected = False

# Fall detection
bg_subtractor = cv2.createBackgroundSubtractorMOG2()
fall_frame_count = 0  # Frame counter
fall_threshold = 12  # Frames to confirm a fall
fall_detected = False

latest_frame = None
frame_lock = threading.Lock()
sent_message = False
frame_processed = False  # Avoid repeated analysis in Llava

def save_image_with_detection(frame):
    filename = "detection.jpg"
    cv2.imwrite(filename, frame)
    return filename

def yolo_detection():
    global latest_frame, fire_detected, frame_processed, fall_detected, fall_frame_count
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = yolo_model.predict(frame)
        annotated_frame = results[0].plot()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = bg_subtractor.apply(gray)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            max_area = max(areas, default=0)
            max_area_index = areas.index(max_area)
            cnt = contours[max_area_index]
            x, y, w, h = cv2.boundingRect(cnt)
            # If height is less than width, possible fall
            if h < w:  
                fall_frame_count += 1
            else:
                fall_frame_count = 0
            
            if fall_frame_count >= fall_threshold:
                fall_detected = True
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(annotated_frame, "FALL DETECTED!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                fall_detected = False
        
        with frame_lock:
            latest_frame = annotated_frame.copy()
        
        fire_detected = False
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = box.conf[0].item()
                if cls == 0 and conf > 0.60:
                    print(f"FIRE DETECTED ({conf:.2f}) 🔥")
                    fire_detected = True
                    frame_processed = False
        
        time.sleep(1 / fps)

def llama_vision_analysis():
    global fire_detected, sent_message, frame_processed, fall_detected
    while True:
        if (fire_detected or fall_detected) and not sent_message and not frame_processed:
            frame_processed = True
            with frame_lock:
                if latest_frame is None:
                    continue
                frame = latest_frame.copy()
            _, buffer = cv2.imencode('.jpg', frame)
            image_data = buffer.tobytes()
            try:
                response = ollama.chat(model="llava-phi3:latest", messages=[
                    {"role": "user", "content": "Do you see any fire or a person who has fallen in the image?", "images": [image_data]}
                ])
                llava_response = response.get("message", {}).get("content", "").lower()
                
                print(f"Llava-Phi3 Response: {llava_response}")
                
                if "fire" in llava_response or "fallen" in llava_response:
                    os.system("afplay /System/Library/Sounds/Sosumi.aiff")
                    message = "🚨 Emergency detected! "
                    if fire_detected:
                        message += "🔥 Fire detected."
                    if fall_detected:
                        message += " Person fallen detected."
                    if send_telegram_message(message):
                        print("✅ Message sent to Telegram.")
                    image_path = save_image_with_detection(frame)
                    if send_telegram_image(image_path):
                        print("✅ Image sent to Telegram.")
                    sent_message = True
                    threading.Timer(30, reset_sent_message).start()
            except Exception as e:
                print(f"Error in Ollama request: {e}")
        time.sleep(1)

def reset_sent_message():
    global sent_message
    sent_message = False

threading.Thread(target=yolo_detection, daemon=True).start()
threading.Thread(target=llama_vision_analysis, daemon=True).start()

while cap.isOpened():
    with frame_lock:
        if latest_frame is not None:
            cv2.imshow('SafeVision AI', latest_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
