import cv2
import numpy as np
import queue
import threading
import time
from collections import deque


# Thread for processing frames and performing calibration
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
                        print("Camera calibrated successfully.")
                        print("Camera matrix:\n", mtx)
                        print("Distortion coefficients:\n", dist)
                        print("Translation:\n", tvecs)
                        print("Rotation:\n", rvecs)
                        calibrated = True
                        motion_detected_event.clear()  # Reset motion detection

            # Send frame with corners drawn to the output queue
            queue_out.put(frame_resized)


# Main function
def main():
    url = "http://192.168.100.4:8080/video"
    cap = cv2.VideoCapture(url)

    while not cap.isOpened():
        print("Reconnecting to the camera...")
        cap = cv2.VideoCapture(url)
        time.sleep(2)

    fps = cap.get(cv2.CAP_PROP_FPS)
    checkerboard_width = 9
    checkerboard_height = 7
    square_size = 20

    # Queues for thread communication
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

    prev_gray_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab a frame.")
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
            processed_frame = queue_out.get()
            cv2.imshow('Camera Calibration', processed_frame)

        prev_gray_frame = gray.copy()

        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord('q'):
            print("Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
