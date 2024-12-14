from flask import Flask, request, Response
import cv2
import mediapipe as mp
import numpy as np
import json
from datetime import datetime

app = Flask(__name__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def convert_np_int32_to_int(o):
    if isinstance(o, np.integer):
        return int(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

@app.route('/process_frame', methods=['POST'])
def process_frame():
    file = request.files['frame']
    frame = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    rep_feedback = {}
    if results.pose_landmarks:
        def get_joint_position(landmark):
            return [landmark.x, landmark.y]

        joints = {
            'right_shoulder': get_joint_position(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]),
            'right_elbow': get_joint_position(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value]),
            'right_wrist': get_joint_position(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value]),
            'left_shoulder': get_joint_position(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]),
            'left_elbow': get_joint_position(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value]),
            'left_wrist': get_joint_position(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value])
        }

        right_angle = calculate_angle(joints['right_shoulder'], joints['right_elbow'], joints['right_wrist'])
        left_angle = calculate_angle(joints['left_shoulder'], joints['left_elbow'], joints['left_wrist'])
        rep_feedback = {
            "right_angle": right_angle,
            "left_angle": left_angle,
            "timestamp": datetime.now().isoformat()
        }
        return jsonify(rep_feedback)

    return jsonify({"error": "No pose landmarks detected"})

@app.route('/')
def index():
    return "Flask server is running."

if __name__ == '__main__':
    app.run(debug=True)