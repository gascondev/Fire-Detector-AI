import cv2
import os
import threading
import time
import requests
from ultralytics import YOLO
import ollama
from dotenv import load_dotenv

# Carga las variables desde el archivo .env
load_dotenv()  


# Configuración de Telegram
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
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": "🚨 📷 Imagen de la situación. 🔥"}
        response = requests.post(url, files=files, data=data)
    return response.status_code == 200

# Cargar modelo YOLO
model_yolo = YOLO('fire_YOLOv8.pt')
cap = cv2.VideoCapture(1)
fps = cap.get(cv2.CAP_PROP_FPS) or 30

latest_frame = None
frame_lock = threading.Lock()
fire_detected = False
sent_message = False
frame_processed = False  # Para evitar análisis repetidos en Llava

def save_image_with_detection(frame):
    filename = "detección.jpg"
    cv2.imwrite(filename, frame)
    return filename


def yolo_detection():
    global latest_frame, fire_detected, frame_processed
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model_yolo.predict(frame)
        annotated_frame = results[0].plot()
        with frame_lock:
            latest_frame = annotated_frame.copy()
        
        fire_detected = False  # Reiniciamos la detección antes de procesar cada frame

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])  # Clase detectada
                conf = box.conf[0].item()  # Confianza de la detección

                if cls == 0 and conf > 0.60:  # Si detecta fuego con más de 60% de confianza
                    print(f"🔥 INCENDIO DETECTADO ({conf:.2f}) 🔥")
                    fire_detected = True
                    frame_processed = False  # Permitir nuevo análisis en Llava

        time.sleep(1 / fps)  # Control de velocidad del video


def llama_vision_analysis():
    global fire_detected, sent_message, frame_processed
    while True:
        if fire_detected and not sent_message and not frame_processed:
            frame_processed = True  # Evitar análisis repetidos
            with frame_lock:
                if latest_frame is None:
                    continue
                frame = latest_frame.copy()
            _, buffer = cv2.imencode('.jpg', frame)
            image_data = buffer.tobytes()
            try:
                response = ollama.chat(model="llava-phi3:latest", messages=[
                    {"role": "user", "content": "Do you see any fire in the image?", "images": [image_data]}
                ])
                respuesta_llama = response.get("message", {}).get("content", "").lower()
                if "fire" in respuesta_llama or "incendio" in respuesta_llama:
                    os.system("afplay /System/Library/Sounds/Sosumi.aiff")
                    if send_telegram_message("🚨 ¡Incendio detectado! 🚨"):
                        print("✅ Mensaje enviado a Telegram.")
                    image_path = save_image_with_detection(frame)
                    if send_telegram_image(image_path):
                        print("✅ Imagen enviada a Telegram.")
                    sent_message = True
                    threading.Timer(30, reset_sent_message).start()
            except Exception as e:
                print(f"❌ Error en la llamada a Ollama: {e}")
        time.sleep(1)

def reset_sent_message():
    global sent_message
    sent_message = False

threading.Thread(target=yolo_detection, daemon=True).start()
threading.Thread(target=llama_vision_analysis, daemon=True).start()

while cap.isOpened():
    with frame_lock:
        if latest_frame is not None:
            cv2.imshow('Detección de Incendios', latest_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
