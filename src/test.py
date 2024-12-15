import cv2
import requests
import time
import threading

def main():
    # URL of your FastAPI server endpoint
    url = "http://localhost:8000/process_frame"

    # Initialize the webcam capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    # Set the desired frame rate (e.g., 15 frames per second)
    desired_frame_rate = 15.0
    frame_interval = 1.0 / desired_frame_rate

    # Initialize counter and last API response
    counter = 0
    last_api_response = {}

    # Create a lock and event for thread synchronization
    frame_lock = threading.Lock()
    frame_event = threading.Event()
    frame_data = None

    # Function to capture frames from the webcam
    def capture_frames():
        nonlocal frame_data
        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame from webcam.")
                break

            # Store the captured frame with thread safety
            with frame_lock:
                frame_data = frame.copy()
                frame_event.set()  # Signal that a new frame is available

            # Wait to maintain the desired frame rate
            elapsed_time = time.time() - start_time
            time_to_wait = frame_interval - elapsed_time
            if time_to_wait > 0:
                time.sleep(time_to_wait)

    # Start the frame capture thread
    capture_thread = threading.Thread(target=capture_frames)
    capture_thread.daemon = True
    capture_thread.start()

    try:
        while True:
            # Wait until a new frame is available
            frame_event.wait()
            with frame_lock:
                frame = frame_data.copy()
                frame_event.clear()  # Reset the event for the next frame

            # Create a copy of the frame for display
            display_frame = frame.copy()

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                print("Error: Failed to encode frame as JPEG.")
                continue  # Skip this frame

            # Prepare the payload
            files = {'file': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}

            # Send the frame to the FastAPI server
            try:
                response = requests.post(url, files=files, timeout=0.5)
                response.raise_for_status()  # Raise an exception for HTTP errors

                # Parse the JSON response
                data = response.json()

                # Print the full JSON response in the terminal
                print("Response:", data)

                # Update counter if a rep was counted
                if data.get('counted'):
                    counter += 1

                # Store the last API response
                last_api_response = data

            except requests.exceptions.RequestException as e:
                print("Error:", e)

            # Overlay counted reps on the frame
            cv2.putText(
                display_frame,
                f"Reps: {counter}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # Display the last API response on the frame
            y0, dy = 60, 30
            for i, (key, value) in enumerate(last_api_response.items()):
                text = f"{key}: {value}"
                y = y0 + i * dy
                cv2.putText(
                    display_frame,
                    text,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

            # Show the live footage with overlays
            cv2.imshow('Live Footage', display_frame)

            # Exit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release the webcam and destroy windows
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()