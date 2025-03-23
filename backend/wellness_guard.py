import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
import json
from cryptography.fernet import Fernet  # For data encryption
import screen_brightness_control as sbc  # For screen brightness control
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import threading

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React front-end

class WellnessGuard:
    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize detection modules
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Initialize variables for tracking
        self.blink_counter = 0
        self.last_blink_time = time.time()
        self.blink_rate = 0  # Blinks per minute
        self.posture_status = "Unknown"
        self.eye_status = "Unknown"
        self.fatigue_level = 0  # 0-100 scale
        
        # Add smoothing for alerts
        self.posture_history = []
        self.eye_status_history = []
        self.history_length = 10  # Number of frames to average
        
        # Store session data
        self.session_start = datetime.now()
        self.session_data = {
            "posture_violations": 0,
            "blink_rate_history": [],
            "break_recommendations": [],
            "fatigue_levels": [],
            "sitting_duration": 0,  # Track sitting duration in seconds
            "screen_distance_violations": 0  # Track screen distance violations
        }
        
        # Face landmark indices (MediaPipe face mesh)
        # Eye landmarks
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        # Alert display settings
        self.alert_start_time = 0
        self.current_alerts = []
        self.alert_duration = 5  # Show alerts for 5 seconds
        
        # Flag to track if there are any issues detected
        self.issues_detected = False

        # Screen distance monitoring
        self.avg_face_width = 0  # Average face width in pixels
        self.face_width_history = []
        self.face_width_samples = 10  # Number of samples to calculate average face width
        self.screen_distance_status = "Unknown"

        # Sitting duration tracking
        self.last_stand_time = time.time()  # Track the last time the user stood up
        self.sitting_duration = 0  # Track sitting duration in seconds
        self.sitting_alert_threshold = 30 * 60  # Alert after 30 minutes of sitting

        # Privacy and Data Security
        self.encryption_key = self.generate_encryption_key()  # Generate encryption key
        self.privacy_policy = """
        Privacy Policy:
        - Your data is stored locally and encrypted for security.
        - No data is shared with third parties.
        - You can delete your data at any time.
        """

        # Screen brightness adjustment
        self.current_brightness = sbc.get_brightness()[0]  # Get current screen brightness
        self.min_brightness = 10  # Minimum brightness level
        self.max_brightness = 100  # Maximum brightness level
        self.healthy_blink_rate = 15  # Healthy blink rate (blinks per minute)

        # Webcam and processing state
        self.is_running = False
        self.cap = None

    def generate_encryption_key(self):
        """Generate a secure encryption key using Fernet."""
        return Fernet.generate_key()

    def encrypt_data(self, data):
        """Encrypt session data using Fernet."""
        fernet = Fernet(self.encryption_key)
        encrypted_data = fernet.encrypt(json.dumps(data).encode())
        return encrypted_data

    def decrypt_data(self, encrypted_data):
        """Decrypt session data using Fernet."""
        fernet = Fernet(self.encryption_key)
        decrypted_data = fernet.decrypt(encrypted_data).decode()
        return json.loads(decrypted_data)

    def delete_session_data(self, filepath="wellness_session_data.json"):
        """Delete session data file."""
        import os
        if os.path.exists(filepath):
            os.remove(filepath)
            print("Session data deleted successfully.")
        else:
            print("No session data found to delete.")

    def display_privacy_policy(self):
        """Display the privacy policy to the user and get consent."""
        print(self.privacy_policy)
        while True:
            consent = input("Do you agree to the privacy policy? (yes/no): ").strip().lower()
            if consent in ["yes", "no"]:
                return consent == "yes"
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")

    def calculate_eye_aspect_ratio(self, landmarks, eye_indices):
        """Calculate the ratio of the eye height to width"""
        points = []
        for i in eye_indices:
            points.append([landmarks[i].x, landmarks[i].y])
        
        # Vertical eye landmarks (top to bottom)
        A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
        B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
        
        # Horizontal eye landmarks (left to right)
        C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
        
        # Eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def detect_blink(self, frame):
        """Detect blinks using eye aspect ratio"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            
            # Calculate eye aspect ratio for both eyes
            left_ear = self.calculate_eye_aspect_ratio(face_landmarks, self.LEFT_EYE)
            right_ear = self.calculate_eye_aspect_ratio(face_landmarks, self.RIGHT_EYE)
            
            # Average the eye aspect ratio
            ear = (left_ear + right_ear) / 2.0
            
            # Blink detection threshold (may need adjustment)
            EYE_AR_THRESHOLD = 0.2
            
            if ear < EYE_AR_THRESHOLD:
                current_time = time.time()
                if current_time - self.last_blink_time > 0.4:  # Prevent counting multiple frames as blinks
                    self.blink_counter += 1
                    self.last_blink_time = current_time
            
            # Calculate blink rate (blinks per minute)
            elapsed_time = (time.time() - self.session_start.timestamp()) / 60.0
            if elapsed_time > 0:
                self.blink_rate = self.blink_counter / elapsed_time
            
            # Evaluate eye strain
            if self.blink_rate < self.healthy_blink_rate * 0.7:
                current_status = "Eye strain detected"
            else:
                current_status = "Healthy"
            
            # Add to history for smoothing
            self.eye_status_history.append(current_status)
            if len(self.eye_status_history) > self.history_length:
                self.eye_status_history.pop(0)
            
            # Only change status if consistent over multiple frames
            strain_count = self.eye_status_history.count("Eye strain detected")
            if strain_count > self.history_length * 0.7:  # 70% of recent frames show strain
                self.eye_status = "Eye strain detected"
            elif strain_count < self.history_length * 0.3:  # Less than 30% show strain
                self.eye_status = "Healthy"
            # Otherwise, keep the previous status for stability
                
            # Record blink rate history every 5 minutes (changed from 30 minutes)
            if len(self.session_data["blink_rate_history"]) == 0 or time.time() - self.session_data["blink_rate_history"][-1]["timestamp"] > 300:  # 300 seconds = 5 minutes
                self.session_data["blink_rate_history"].append({
                    "timestamp": time.time(),
                    "rate": self.blink_rate
                })
            
            return self.eye_status, self.blink_rate
        
        return "Face not detected", 0
    
    def adjust_screen_brightness(self):
        """Adjust screen brightness based on blink rate."""
        if self.blink_rate <= 4.5:  # Adjusted blink rate range
            new_brightness = max(self.min_brightness, self.current_brightness - 10)  # Reduce brightness
        else:
            new_brightness = min(self.max_brightness, self.current_brightness + 10)  # Increase brightness
        
        # Set the new brightness
        sbc.set_brightness(new_brightness)
        self.current_brightness = new_brightness

    def detect_posture(self, frame):
        """Detect posture using shoulder and head position"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Get key body points
            left_shoulder = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y)
            right_shoulder = (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                             landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y)
            nose = (landmarks[mp.solutions.pose.PoseLandmark.NOSE.value].x,
                   landmarks[mp.solutions.pose.PoseLandmark.NOSE.value].y)
            
            # Calculate shoulder slope (should be close to horizontal for good posture)
            shoulder_slope = abs(left_shoulder[1] - right_shoulder[1])
            
            # Calculate the position of the nose relative to the shoulders' midpoint
            shoulder_midpoint_x = (left_shoulder[0] + right_shoulder[0]) / 2
            shoulder_midpoint_y = (left_shoulder[1] + right_shoulder[1]) / 2
            
            # Check if head is too far forward (forward head posture)
            head_forward_threshold = 0.1  # May need adjustment
            
            # A well-aligned head would be approximately above the shoulder midpoint
            head_shoulder_horizontal_distance = abs(nose[0] - shoulder_midpoint_x)
            
            # Check if shoulders are hunched (by checking if they're too high relative to the ears)
            shoulder_level_threshold = 0.05  # May need adjustment
            
            # Determine current frame's posture status
            if shoulder_slope > shoulder_level_threshold or head_shoulder_horizontal_distance > head_forward_threshold:
                current_posture = "Poor posture detected"
            else:
                current_posture = "Good posture"
            
            # Add to history for smoothing
            self.posture_history.append(current_posture)
            if len(self.posture_history) > self.history_length:
                self.posture_history.pop(0)
            
            # Only change status if consistent over multiple frames
            poor_count = self.posture_history.count("Poor posture detected")
            if poor_count > self.history_length * 0.7:  # 70% of recent frames show poor posture
                if self.posture_status != "Poor posture detected":
                    self.session_data["posture_violations"] += 1
                self.posture_status = "Poor posture detected"
            elif poor_count < self.history_length * 0.3:  # Less than 30% show poor posture
                self.posture_status = "Good posture"
            # Otherwise, keep the previous status for stability
            
            return self.posture_status
        
        return "Body not detected"
    
    def monitor_screen_distance(self, frame):
        """Monitor the distance between the user's eyes and the screen"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            
            # Calculate face width using eye landmarks
            left_eye = np.array([face_landmarks[self.LEFT_EYE[0]].x, face_landmarks[self.LEFT_EYE[0]].y])
            right_eye = np.array([face_landmarks[self.RIGHT_EYE[0]].x, face_landmarks[self.RIGHT_EYE[0]].y])
            face_width = np.linalg.norm(left_eye - right_eye)
            
            # Update face width history
            self.face_width_history.append(face_width)
            if len(self.face_width_history) > self.face_width_samples:
                self.face_width_history.pop(0)
            
            # Calculate average face width
            if len(self.face_width_history) == self.face_width_samples:
                self.avg_face_width = np.mean(self.face_width_history)
            
            # Determine screen distance status
            if self.avg_face_width > 0:
                if self.avg_face_width < 0.1:  # Too far (face is too small)
                    self.screen_distance_status = "Too far from screen"
                    self.session_data["screen_distance_violations"] += 1
                elif self.avg_face_width > 0.2:  # Too close (face is too large)
                    self.screen_distance_status = "Too close to screen"
                    self.session_data["screen_distance_violations"] += 1
                else:
                    self.screen_distance_status = "Optimal distance"
            
            return self.screen_distance_status
        
        return "Face not detected"
    
    def track_sitting_duration(self):
        """Track how long the user has been sitting"""
        current_time = time.time()
        self.sitting_duration = current_time - self.last_stand_time
        self.session_data["sitting_duration"] = self.sitting_duration
        
        # Check if sitting duration exceeds the threshold
        if self.sitting_duration > self.sitting_alert_threshold:
            return "Take a break and stand up"
        return None
    
    def analyze_fatigue(self):
        """Analyze fatigue levels based on posture and eye metrics"""
        # Simple fatigue model combining posture violations and blink rate
        posture_factor = min(self.session_data["posture_violations"] / 10, 50)  # Max 50 points for posture
        
        # Blink rate factor (both too low and too high can indicate fatigue)
        if len(self.session_data["blink_rate_history"]) > 0:
            recent_blink_rates = [entry["rate"] for entry in self.session_data["blink_rate_history"][-5:] if "rate" in entry]
            if recent_blink_rates:
                avg_blink_rate = sum(recent_blink_rates) / len(recent_blink_rates)
                blink_deviation = abs(avg_blink_rate - self.healthy_blink_rate)
                blink_factor = min(blink_deviation * 2, 50)  # Max 50 points for blink issues
            else:
                blink_factor = 0
        else:
            blink_factor = 0
        
        # Calculate fatigue level (0-100)
        self.fatigue_level = posture_factor + blink_factor
        
        # Record fatigue level
        self.session_data["fatigue_levels"].append({
            "timestamp": time.time(),
            "level": self.fatigue_level
        })
        
        return self.fatigue_level
    
    def recommend_break(self):
        """Generate adaptive break recommendations based on fatigue level"""
        # Analyze current fatigue level
        fatigue_level = self.analyze_fatigue()
        
        # Calculate time since last break
        last_break_time = 0
        if self.session_data["break_recommendations"]:
            last_break_time = self.session_data["break_recommendations"][-1]["timestamp"]
        
        time_since_break = time.time() - last_break_time if last_break_time > 0 else time.time() - self.session_start.timestamp()
        
        # Determine break recommendations based on fatigue and time
        if fatigue_level > 70 and time_since_break > 20 * 60:  # High fatigue, 20+ minutes since break
            recommendation = {
                "timestamp": time.time(),
                "duration": "10 minutes",
                "urgency": "high",
                "exercises": ["Walk around", "Look at distant objects", "Stretch shoulders and neck"]
            }
        elif fatigue_level > 50 and time_since_break > 30 * 60:  # Medium fatigue, 30+ minutes
            recommendation = {
                "timestamp": time.time(),
                "duration": "5 minutes",
                "urgency": "medium",
                "exercises": ["Look at distant objects", "Shoulder rolls", "Neck stretches"]
            }
        elif fatigue_level > 30 and time_since_break > 45 * 60:  # Low fatigue, 45+ minutes
            recommendation = {
                "timestamp": time.time(),
                "duration": "2 minutes",
                "urgency": "low",
                "exercises": ["Eye palming", "Blink exercises", "Deep breathing"]
            }
        elif time_since_break > 60 * 60:  # 1+ hour regardless of fatigue
            recommendation = {
                "timestamp": time.time(),
                "duration": "5 minutes",
                "urgency": "medium",
                "exercises": ["Stand and stretch", "Eye relaxation", "Hydrate"]
            }
        else:
            return None
        
        # Add recommendation to history
        self.session_data["break_recommendations"].append(recommendation)
        return recommendation
    
    def get_corrective_guidance(self):
        """Provide personalized corrective guidance based on detected issues"""
        guidance = []
        self.issues_detected = False
        
        # Posture guidance
        if self.posture_status == "Poor posture detected":
            self.issues_detected = True
            guidance.append({
                "type": "posture",
                "message": "Straighten your back and align your head with your shoulders",
                "exercise": "Pull your shoulders back and down, imagine a string pulling the top of your head upward"
            })
        
        # Eye strain guidance
        if self.eye_status == "Eye strain detected":
            self.issues_detected = True
            guidance.append({
                "type": "eye",
                "message": "You're not blinking enough. Practice the 20-20-20 rule.",
                "exercise": "Every 20 minutes, look at something 20 feet away for 20 seconds"
            })
        
        # Screen distance guidance
        if self.screen_distance_status == "Too close to screen":
            self.issues_detected = True
            guidance.append({
                "type": "screen_distance",
                "message": "You're too close to the screen. Move back to maintain a healthy distance.",
                "exercise": "Sit at least an arm's length away from the screen"
            })
        elif self.screen_distance_status == "Too far from screen":
            self.issues_detected = True
            guidance.append({
                "type": "screen_distance",
                "message": "You're too far from the screen. Move closer for better visibility.",
                "exercise": "Adjust your seating position to maintain an optimal distance"
            })
        
        # Sitting duration guidance
        sitting_alert = self.track_sitting_duration()
        if sitting_alert:
            self.issues_detected = True
            guidance.append({
                "type": "sitting_duration",
                "message": sitting_alert,
                "exercise": "Stand up, stretch, and walk around for a few minutes"
            })
        
        # Break recommendation based on fatigue
        break_rec = self.recommend_break()
        if break_rec:
            self.issues_detected = True
            guidance.append({
                "type": "break",
                "message": f"Take a {break_rec['duration']} break (Urgency: {break_rec['urgency']})",
                "exercise": ", ".join(break_rec['exercises'])
            })
        
        # Update current alerts if we have new guidance
        if guidance and (time.time() - self.alert_start_time > self.alert_duration):
            self.current_alerts = guidance
            self.alert_start_time = time.time()
        
        return guidance
    
    def save_session_data(self, filepath="wellness_session_data.json"):
        """Save session data to a file"""
        data = {
            "session_start": self.session_start.isoformat(),
            "session_end": datetime.now().isoformat(),
            "session_duration_minutes": (datetime.now() - self.session_start).total_seconds() / 60,
            "data": self.session_data
        }
        
        # Encrypt session data before saving
        encrypted_data = self.encrypt_data(data)
        with open(filepath, 'wb') as f:
            f.write(encrypted_data)
        
        return filepath
    
    def process_frame(self, frame):
        """Process a video frame and return analysis results without overlay."""
        # Mirror the frame horizontally for more natural interaction
        frame = cv2.flip(frame, 1)
        
        # Detect posture and eye metrics
        posture_status = self.detect_posture(frame)
        eye_status, blink_rate = self.detect_blink(frame)
        
        # Monitor screen distance
        screen_distance_status = self.monitor_screen_distance(frame)
        
        # Analyze fatigue levels
        fatigue_level = self.analyze_fatigue()
        
        # Adjust screen brightness based on blink rate
        self.adjust_screen_brightness()
        
        # Get guidance if needed
        guidance = self.get_corrective_guidance()
        
        # Return the frame without any overlay
        return frame

    def generate_frames(self):
        """Generate frames from the webcam for streaming without overlay."""
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = self.process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Initialize WellnessGuard instance
wellness_guard = WellnessGuard()

# Flask API Endpoints
@app.route('/start', methods=['POST'])
def start_wellness_guard():
    try:
        # Start the webcam and processing
        wellness_guard.start_webcam()
        return jsonify({"status": "WellnessGuard started"})
    except Exception as e:
        print(f"Error starting WellnessGuard: {e}")
        return jsonify({"status": "Error starting WellnessGuard", "error": str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    status = {
        "posture_status": wellness_guard.posture_status,
        "eye_status": wellness_guard.eye_status,
        "blink_rate": wellness_guard.blink_rate,
        "fatigue_level": wellness_guard.fatigue_level,
        "screen_distance_status": wellness_guard.screen_distance_status,
        "sitting_duration": wellness_guard.sitting_duration,
        "alerts": wellness_guard.current_alerts
    }
    return jsonify(status)

@app.route('/stop', methods=['POST'])
def stop_wellness_guard():
    wellness_guard.save_session_data()
    return jsonify({"status": "WellnessGuard stopped"})

@app.route('/video_feed')
def video_feed():
    """Route to stream the webcam feed."""
    return Response(wellness_guard.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run Flask server
if __name__ == "__main__":
    threading.Thread(target=app.run, kwargs={"host": "0.0.0.0", "port": 5000}).start()