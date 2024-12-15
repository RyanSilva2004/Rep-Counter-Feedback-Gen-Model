import cv2
import mediapipe as mp
import numpy as np
import json
from datetime import datetime

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

counter = 0
clean_counter = 0
stage = None

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

def process_frame(frame):
    global counter, clean_counter, stage

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

        feedback = ""
        if right_angle > 160 and left_angle > 160:
            stage = "down"
            feedback = "Lower your arms"
        elif right_angle < 45 and left_angle < 45 and stage == "down":
            stage = "up"
            counter += 1
            if right_angle < 35 and left_angle < 35:
                clean_counter += 1
                feedback = "Good, clean rep!"
            else:
                feedback = "Rep counted, but improve form."

            rep_feedback = {
                "rep_count": counter,
                "timestamp": datetime.now().isoformat(),
                "form_quality": {
                    "right_shoulder_to_elbow_to_wrist": {
                        "status": "green" if right_angle < 35 else "yellow",
                        "pixel_loc_right_shoulder": tuple(np.multiply(joints['right_shoulder'], [frame.shape[1], frame.shape[0]]).astype(int))
                    },
                    "left_shoulder_to_elbow_to_wrist": {
                        "status": "green" if left_angle < 35 else "yellow",
                        "pixel_loc_left_shoulder": tuple(np.multiply(joints['left_shoulder'], [frame.shape[1], frame.shape[0]]).astype(int))
                    }
                },
                "logic": f"Right angle was {right_angle:.2f}, classified as {'green' if right_angle < 35 else 'yellow'}. "
                         f"Left angle was {left_angle:.2f}, classified as {'green' if left_angle < 35 else 'yellow'}"
            }
            print(json.dumps(rep_feedback, default=convert_np_int32_to_int, indent=4))
        else:
            feedback = "Maintain good form!"

        cv2.putText(image, f'Reps: {counter}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'Clean Reps: {clean_counter}', (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    return image

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = process_frame(frame)
    cv2.imshow('Rep Feedback Tracker', processed_frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()