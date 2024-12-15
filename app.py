import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

app = FastAPI()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
counter = 0
stage = None

def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

@app.post('/process_frame')
async def process_frame(file: UploadFile = File(...)):
    global counter, stage

    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        def get_joint_position(landmark):
            return [landmark.x, landmark.y]

        frame_height, frame_width = frame.shape[:2]

        joints = {
            'right_shoulder': get_joint_position(
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            ),
            'right_elbow': get_joint_position(
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            ),
            'right_wrist': get_joint_position(
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            ),
            'left_shoulder': get_joint_position(
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            ),
            'left_elbow': get_joint_position(
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            ),
            'left_wrist': get_joint_position(
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            ),
        }

        right_angle = calculate_angle(
            joints['right_shoulder'], joints['right_elbow'], joints['right_wrist']
        )
        left_angle = calculate_angle(
            joints['left_shoulder'], joints['left_elbow'], joints['left_wrist']
        )

        if right_angle > 160 and left_angle > 160:
            stage = "down"
        elif right_angle < 45 and left_angle < 45 and stage == "down":
            stage = "up"
            counter += 1

            pixel_right_shoulder = tuple(
                map(int, np.multiply(joints['right_shoulder'], [frame_width, frame_height]))
            )
            pixel_left_shoulder = tuple(
                map(int, np.multiply(joints['left_shoulder'], [frame_width, frame_height]))
            )

            rep_feedback = {
                "counted": True,
                "timestamp": datetime.now().isoformat(),
                "form_quality": {
                    "right_shoulder_to_elbow_to_wrist": {
                        "status": "green" if right_angle < 35 else "yellow",
                        "pixel_loc_right_shoulder": pixel_right_shoulder
                    },
                    "left_shoulder_to_elbow_to_wrist": {
                        "status": "green" if left_angle < 35 else "yellow",
                        "pixel_loc_left_shoulder": pixel_left_shoulder
                    }
                },
                "logic": (
                    f"Right angle was {right_angle:.2f}, classified as "
                    f"{'green' if right_angle < 35 else 'yellow'}. "
                    f"Left angle was {left_angle:.2f}, classified as "
                    f"{'green' if left_angle < 35 else 'yellow'}"
                )
            }
            return JSONResponse(content=rep_feedback)

    return JSONResponse(content={"counted": False})

if __name__ == "__main__":
    uvicorn.run("your_script_name:app", host="0.0.0.0", port=8000, reload=True)