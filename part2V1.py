'''
Ce fichier contient le code pour le calibrage de la caméra en  utilisant la notion parallelisme (threads) 
il est presenté dans l'interface dans la page calibration_page()
mais a été simplifié avant etre utilisé dans la partie 3 

'''


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

    objpoints = []  # Sauvgarder les points 3D 
    imgpoints = []  # Sauvgarder les points 2D correspondants
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    calibrated = False
    mtx, dist, rvecs, tvecs = None, None, None, None

    while True:
        if not queue_in.empty():
            frame_resized, gray = queue_in.get()

            # Détéction du checkerboard
            ret, corners = cv2.findChessboardCorners(gray, (checkerboard_width, checkerboard_height), None)
            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                frame_resized = cv2.drawChessboardCorners(frame_resized, (checkerboard_width, checkerboard_height), corners2, ret)

            # Faire le calibrage pour la première fois ou en cas de mouvement
            if (motion_detected_event.is_set() or not calibrated) and ret:
                imgpoints.append(corners2)
                objpoints.append(objp)

                if len(objpoints) >= 10:  
                    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
                    if ret:
                        print("Camera calibrated successfully.")
                        print("Camera matrix:\n", mtx)
                        print("Distortion coefficients:\n", dist)
                        print("Translation:\n", tvecs)
                        print("Rotation:\n", rvecs)
                        calibrated = True
                        motion_detected_event.clear() 

            # Envoyer la frame pour visualisation
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

    # File pour communication entre threads
    queue_in = queue.Queue()
    queue_out = queue.Queue()

    # Motion detection
    motion_detected_event = threading.Event()
    motion_detector = cv2.createBackgroundSubtractorMOG2(history=700, varThreshold=40, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    motion_levels = deque(maxlen=5) 

    # Lancement du thread qui s'occupe des calculs du calibrage 
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

        # Ignorer la detection de mouvement pour la prmière frame récupérée
        if prev_gray_frame is None:
            prev_gray_frame = gray
            continue

        # Detecetr le mouvement en utilisant la diffrence entre les frames 
        diff_frame = cv2.absdiff(prev_gray_frame, gray)
        _, diff_thresh = cv2.threshold(diff_frame, 25, 255, cv2.THRESH_BINARY)

        #  Enlever le bruit en utilisant les operations morphologiques 
        motion_mask = motion_detector.apply(gray)
        motion_mask_cleaned = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)

        # Calcul du niveau du mouvement et la moyenne de ce dernier 
        motion_level = np.sum(motion_mask_cleaned) / motion_mask_cleaned.size
        motion_levels.append(motion_level)
        average_motion = sum(motion_levels) / len(motion_levels)

        # Détecter les mouvements si il depasse notre seuil 
        if average_motion > 0.1:  
            motion_detected_event.set()

        # Envoyer la frame au thread pour faire les calculs 
        queue_in.put((frame_resized, gray))

        # Afficher l'image avec les corners du cherboard 
        if not queue_out.empty():
            processed_frame = queue_out.get()
            cv2.imshow('Camera Calibration', processed_frame)

        prev_gray_frame = gray.copy()

        key = cv2.waitKey(int(1000 / fps)) & 0xFF # utiliser l'information nombre de frame par seconds pour personaliser la valeurs du waitkey 
        if key == ord('q'):
            print("Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
