from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import model_from_json
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

app = Flask(__name__)

class SafetyAnalytics:
    def __init__(self):
        # Load emotion detection model
        with open('models/facialemotionmodel.json', 'r') as json_file:
            emotion_model_json = json_file.read()
        self.emotion_model = model_from_json(emotion_model_json)
        self.emotion_model.load_weights('models/facialemotionmodel.h5')

        # Load gender and age detection model
        with open('models/Gender-and-Age-Predictions.json', 'r') as json_file:
            gender_age_model_json = json_file.read()
        self.gender_age_model = model_from_json(gender_age_model_json)
        self.gender_age_model.load_weights('models/Gender-and-Age-Predictions.weights.h5')

        # Load face cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Labels
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.gender_dict = {0: 'Female', 1: 'Male'}

        # Alert tracking
        self.alert_cooldown = 30  # seconds between alerts
        self.last_alert_time = 0
        self.alert_condition_start_time = None
        
    def send_sos_alert(self, emotion, gender, age):
        try:
            smtp_server = "smtp.gmail.com"
            smtp_port = 587
            sender_email = "xyz@gmail.com"  # Replace with your email
            receiver_email = "abc@example.com"  # Replace with receiver email
            password = "5432#1"  # Replace with your app password

            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = receiver_email
            msg['Subject'] = "URGENT: Safety Alert Detected"

            body = f"""
            Safety Alert Details:
            Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            Location: Camera 1
            Details:
            - Gender: {gender}
            - Age: {age}
            - Emotional State: {emotion}
            
            This alert was triggered due to detection of potential distress.
            Please investigate immediately.
            """

            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, password)
                server.send_message(msg)

            print("Alert sent successfully")
            return True
        except Exception as e:
            print(f"Failed to send alert: {str(e)}")
            return False

    def process_frame(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        
        current_time = time.time()
        
        for (x, y, w, h) in faces:
            # Process face for emotion detection
            face = gray_frame[y:y+h, x:x+w]
            face_emotion = cv2.resize(face, (48, 48))
            face_emotion = face_emotion / 255.0
            face_emotion = np.expand_dims(face_emotion, axis=0)
            face_emotion = np.expand_dims(face_emotion, axis=-1)
            
            # Process face for gender and age detection
            face_gender = cv2.resize(face, (128, 128))
            face_gender = face_gender / 255.0
            face_gender = np.expand_dims(face_gender, axis=0)
            face_gender = np.expand_dims(face_gender, axis=-1)
            
            # Get predictions
            emotion_prediction = self.emotion_model.predict(face_emotion)
            gender_age_prediction = self.gender_age_model.predict(face_gender)
            
            emotion_label = self.emotion_labels[np.argmax(emotion_prediction)]
            gender_label = self.gender_dict[round(gender_age_prediction[0][0][0])]
            predicted_age = round(gender_age_prediction[1][0][0])
            
            # Draw on frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f'Emotion: {emotion_label}', (x, y-40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f'Gender: {gender_label}, Age: {predicted_age}', 
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Check alert conditions
            if (gender_label == 'Female' and 
                emotion_label in ['Sad', 'Disgust', 'Fear', 'Surprise']):
                
                if self.alert_condition_start_time is None:
                    self.alert_condition_start_time = current_time
                elif (current_time - self.alert_condition_start_time >= 30 and 
                      current_time - self.last_alert_time >= self.alert_cooldown):
                    if self.send_sos_alert(emotion_label, gender_label, predicted_age):
                        self.last_alert_time = current_time
                    self.alert_condition_start_time = None
            else:
                self.alert_condition_start_time = None
                
        return frame

def gen_frames():
    analytics = SafetyAnalytics()
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Process frame
        processed_frame = analytics.process_frame(frame)
        
        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
