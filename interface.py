import streamlit as st

import cv2
import numpy as np
import queue
import threading
import time


#.............................................................................................
def home_page():
    st.title("Home Page")
    st.write("Welcome to the Home Page of the Vision Project!")


#...............................................................................................
def detecting_page():
    st.title("Detecting a 3D Object Page")

#....................................................................................................


import cv2
import numpy as np
import queue
import threading
import time
from collections import deque
import streamlit as st

# Thread for processing frames and performing camera calibration
import cv2
import numpy as np
import queue
import threading
import time
import streamlit as st
from collections import deque


# Processing thread for calibration
def processing_thread(queue_in, queue_out, checkerboard_width, checkerboard_height, square_size, motion_detected_event):
    objp = np.zeros((checkerboard_width * checkerboard_height, 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_width, 0:checkerboard_height].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []  # Store 3D points
    imgpoints = []  # Store corresponding 2D points
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    calibrated = False
    mtx, dist, rvecs, tvecs = None, None, None, None

    while True:
        if not queue_in.empty():
            frame_resized, gray = queue_in.get()

            # Always detect corners for visualization
            ret, corners = cv2.findChessboardCorners(gray, (checkerboard_width, checkerboard_height), None)
            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                frame_resized = cv2.drawChessboardCorners(frame_resized, (checkerboard_width, checkerboard_height), corners2, ret)

            # Perform calibration if motion is detected or first run
            if (motion_detected_event.is_set() or not calibrated) and ret:
                imgpoints.append(corners2)
                objpoints.append(objp)

                if len(objpoints) >= 10:  # Adjust threshold as needed
                    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
                    if ret:
                        calibrated = True
                        motion_detected_event.clear()  # Reset motion detection
                        # Send calibration results to the queue
                        params = {
                            "ret": ret,
                            "camera_matrix": mtx.tolist(),
                            "dist_coefficients": dist.tolist(),
                            "rotation_vectors": [vec.tolist() for vec in rvecs],
                            "translation_vectors": [vec.tolist() for vec in tvecs],
                        }
                        queue_out.put((frame_resized, params))

            # Always send the processed frame for visualization
            queue_out.put((frame_resized, None))


# Streamlit UI for calibration
def calibration_page():
    st.title("Camera Calibration Interface")

    # Input for URL
    url = st.text_input("Enter the video stream URL:", "http://192.168.100.4:8080/video")

    # Calibration parameters
    checkerboard_width = st.number_input("Checkerboard Width:", value=9, step=1)
    checkerboard_height = st.number_input("Checkerboard Height:", value=7, step=1)
    square_size = st.number_input("Square Size (in mm):", value=20, step=1)

    

    if st.button("Start Calibration", key="start_calibration"):
        cap = cv2.VideoCapture(url)

        if not cap.isOpened():
            st.error("Failed to connect to the video stream. Please check the URL.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        params = {
            "checkerboard_width": checkerboard_width,
            "checkerboard_height": checkerboard_height,
            "square_size": square_size
        }

        # Queues for threading
        queue_in = queue.Queue()
        queue_out = queue.Queue()

        # Motion detection
        motion_detected_event = threading.Event()
        motion_detector = cv2.createBackgroundSubtractorMOG2(history=700, varThreshold=40, detectShadows=True)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_levels = deque(maxlen=5)  # Rolling average for motion stabilization

        # Start the processing thread
        thread = threading.Thread(target=processing_thread,
                                  args=(queue_in, queue_out, checkerboard_width, checkerboard_height, square_size, motion_detected_event))
        thread.daemon = True
        thread.start()

        # Streamlit live video display
        stframe = st.empty()
        st.subheader("Calibration Results:")
        result_text = st.empty()

        prev_gray_frame = None

        stop_calibration = st.button("Stop Calibration", key="stop_calibration")  
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab a frame.")
                break

            frame_resized = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

            # Skip motion detection on the first frame
            if prev_gray_frame is None:
                prev_gray_frame = gray
                continue

            # Detect motion using absolute difference
            diff_frame = cv2.absdiff(prev_gray_frame, gray)
            _, diff_thresh = cv2.threshold(diff_frame, 25, 255, cv2.THRESH_BINARY)

            # Remove noise using morphological operations
            motion_mask = motion_detector.apply(gray)
            motion_mask_cleaned = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)

            # Calculate motion level and apply a rolling average
            motion_level = np.sum(motion_mask_cleaned) / motion_mask_cleaned.size
            motion_levels.append(motion_level)
            average_motion = sum(motion_levels) / len(motion_levels)

            # Trigger motion detected event if above threshold
            if average_motion > 0.1:  # Adjust this threshold as needed
                motion_detected_event.set()

            # Send frame for processing
            queue_in.put((frame_resized, gray))

            # Display processed frame with corners
            if not queue_out.empty():
                processed_frame, calibration_params = queue_out.get()
                stframe.image(processed_frame, channels="BGR")

                # Display calibration results if available
                if calibration_params:
                    formatted_params = "\n".join([ 
                        f"ret: {calibration_params['ret']}",
                        f"Camera Matrix:\n{np.array(calibration_params['camera_matrix'])}",
                        f"Distortion Coefficients:\n{np.array(calibration_params['dist_coefficients'])}",
                        f"Rotation Vectors:\n{calibration_params['rotation_vectors']}",
                        f"Translation Vectors:\n{calibration_params['translation_vectors']}",
                    ])
                    result_text.text(formatted_params)

            prev_gray_frame = gray.copy()

            if stop_calibration:
                cap.release()
                cv2.destroyAllWindows()
                break




#..............................................................................................................


def position_page():
    st.title("Position Page")
    

def improuvemenets_page():
    st.title("Improuvement Page")
    


#side menu bar 
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Home", "Detecting a 3D point", "Camera Calibration", "Position of a 3D object","Improuvements"])

if menu == "Home":
    home_page()
elif menu == "Detecting a 3D point":
    detecting_page()
elif menu == "Camera Calibration":
    calibration_page()
elif menu == "Position of a 3D object":
    position_page()
elif menu == "Improuvements":
    improuvemenets_page()
