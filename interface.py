import streamlit as st
import cv2
import numpy as np
import queue
import threading
import time
from collections import deque


from part1 import bgr_to_hsv, in_range, find_contours, min_enclosing_circle, gaussian_blur
from part3 import calculate_3d_position, detect_camera_movement, calculate_mid_point, calculate_horizontal_displacement,validate_y_coordinates

#.............................................................................................
def home_page():
    st.title("Home Page")
    st.write("Welcome to the Home Page of the Vision Project!")


#...............................................................................................
# PARTIE 1 : Conversion, détection de couleur et flou


def detecting_page():
    st.title("Detecting a 3D Object Page")
    # Input for URL
    url = st.text_input("Enter the video stream URL:", "http://192.168.100.4:8080/video")

    if st.button("Start Detecting", key="start_detecting"):
        
        cap = cv2.VideoCapture(url)

        if not cap.isOpened():
            st.error("Failed to connect to the video stream. Please check the URL.")
            return
        # Plage de couleurs pour détecter le vert (peut être ajustée selon votre objet)
        lower = np.array([35, 100, 50])  # Valeurs ajustées
        upper = np.array([85, 255, 255])  # Valeurs ajustées

        stframe = st.empty()
        stop_detection = st.button("Stop Detection", key="stop_detection")  

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Erreur : Impossible de lire l'image")
                break

            # Convertir l'image en HSV
            hsv = bgr_to_hsv(frame)

            # Appliquer un flou gaussien pour réduire le bruit
            blurred = gaussian_blur(hsv, 3)

            # Créer le masque pour la couleur verte
            mask = in_range(blurred, lower, upper)
   
            # Trouver les contours dans le masque
            contours = find_contours(mask)
    
            # Si des contours sont trouvés
            if contours:
                # Trier les contours en fonction de la taille
                contours = sorted(contours, key=cv2.contourArea, reverse=True)

                for contour in contours:
                    # Trouver le cercle minimum qui entoure le contour
                    center, radius = min_enclosing_circle(contour)
                    
                    # Si le rayon est suffisant pour être un objet visible
                    if radius > 400:
                        # Dessiner le cercle sur l'image
                        print('detected')
                        cv2.circle(frame, center, radius, (0, 255, 0), 2)  # Cercle vert
                        cv2.putText(frame, "Objet detecte dans la camera  ", (center[0] - 50, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    cv2.circle(frame, tuple(map(int, center)), int(radius), (0, 255, 0), 2)  

            stframe.image(frame, channels="BGR")

            if stop_detection:
                cap.release()
                cv2.destroyAllWindows()
                break
    
#....................................................................................................

# Apporter des modifications au code déja definit dans part2V1 pour qu'il puisse etre utilisé dans l'interface

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

# Apporter des modifications au code de la fonction calibrate_camera_from_video 
# déja definit dans part2V2 pour qu'il puisse etre utilisé dans l'interface

def calibrate_camera_from_video(camera_source, rows, cols, square_size):
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        st.error(f"Error: Unable to open camera source: {camera_source}")
        return False, None, None, None, None

    obj_points = []  # Points 3D in the real world
    img_points = []  # Points 2D in the image
    obj_p = np.zeros((rows * cols, 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj_p *= square_size

    progress_bar = st.progress(0)
    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    while len(img_points) < 10:  # Minimum 10 images for calibration
        ret, frame = cap.read()
        if not ret:
            status_placeholder.error("Error reading video stream.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        if ret:
            img_points.append(corners)
            obj_points.append(obj_p)
            cv2.drawChessboardCorners(frame, (cols, rows), corners, ret)

        # Update the progress bar and display the frame
        progress_bar.progress(min(len(img_points) / 10, 1.0))
        frame_resized = cv2.resize(frame, (640, 480))
        frame_placeholder.image(frame_resized, channels="BGR")

        if len(img_points) >= 10:
            status_placeholder.success("Calibration completed!")
            break

        time.sleep(0.1)  # Small delay to simulate smooth progress

    cap.release()
    cv2.destroyAllWindows()

    if len(img_points) < 10:
        st.error("Not enough valid frames for calibration.")
        return False, None, None, None, None

    # Perform calibration
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    return ret, K, dist, rvecs, tvecs

def position_page():
    st.title("Position Page")

    url1 = st.text_input("Enter the first video stream URL:", "http://192.168.100.4:8080/video")
    url2 = st.text_input("Enter the second video stream URL:", "http://192.168.100.4:8080/video")

    # Calibration parameters
    checkerboard_width = st.number_input("Checkerboard Width:", value=9, step=1)
    checkerboard_height = st.number_input("Checkerboard Height:", value=7, step=1)
    square_size = st.number_input("Square Size (in mm):", value=20, step=1)

    

    frame_skip = 5  
    frame_count = 0

    previous_frame1 = None
    previous_frame2 = None

    movement2 = False
    movement1 = False
    # Add a flag to track recalibration
    recalibration_done = False

    

    if st.button("Start", key="start"):
         
        with st.spinner("Calibrating first camera..."):
            ret1, K1, dist1, rvecs1, tvecs1 = calibrate_camera_from_video(
                url1, checkerboard_width, checkerboard_height, square_size
            )
        
        if not ret1:
            st.error("Calibration failed for the first camera.")
            exit(0)
        else:
            st.success("First camera calibration successful!")
            st.write("Camera Matrix (K1):", K1)
            st.write("Distortion Coefficients (dist1):", dist1)
            st.write("Rotation R:  (rvecs1):", rvecs1)
            st.write("Translation T  (tvecs1):", tvecs1)
        
        with st.spinner("Calibrating second camera..."):
            ret2, K2, dist2, rvecs2, tvecs2 = calibrate_camera_from_video(
                url2, checkerboard_width, checkerboard_height, square_size
            )
        
        if not ret2:
            st.error("Calibration failed for the second camera.")
            exit(0)
        else:
            st.success("Second camera calibration successful!")
            st.write("Camera Matrix (K2):", K2)
            st.write("Distortion Coefficients (dist2):", dist2)
            st.write("Rotation R:  (rvecs2):", rvecs2)
            st.write("Translation T  (tvecs2):", tvecs2)

        if ret1 and ret2:
            st.success("Both cameras successfully calibrated!")

        cap1 = cv2.VideoCapture(url1)
        cap2 = cv2.VideoCapture(url2)
        
        if not cap1.isOpened() or not cap2.isOpened():
            st.error("Erreur lors de l'ouverture des caméras.")
            exit(0)

        stframe1 = st.empty()
        stframe2 = st.empty()

        stop = st.button("Stop", key="stop")
        while True:
            frame_count += 1

            # Skip frames
            if frame_count % frame_skip != 0:
                continue

            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                st.error("Erreur lors de la lecture des caméras.")
                break

            if previous_frame1 is not None and previous_frame2 is not None:
                movement1 = detect_camera_movement(previous_frame1, frame1, change_ratio_threshold=0.2)
                movement2 = detect_camera_movement(previous_frame2, frame2, change_ratio_threshold=0.2)

            if (movement1 or movement2) and not recalibration_done:
                st.write("Mouvement détecté. Recalibration des caméras...")

                with st.spinner("Calibrating first camera..."):
                    ret1, K1, dist1, rvecs1, tvecs1 = calibrate_camera_from_video(
                    url1, checkerboard_width, checkerboard_height, square_size
                )

                with st.spinner("Calibrating second camera..."):
                    ret2, K2, dist2, rvecs2, tvecs2 = calibrate_camera_from_video(
                     url2, checkerboard_width, checkerboard_height, square_size
                )
           
                recalibration_done = True  # Mark recalibration as done
                movement2 = False
                movement1 = False 
                if not ret1 or not ret2:
                    st.error("Erreur lors de la recalibration.")
                    break
                elif not movement1 and not movement2:
                    recalibration_done = False  # Reset the flag when no movement is detectedq

            # Process frames (color conversion, blurring, detection, etc.)
            hsv1 = bgr_to_hsv(frame1)
            hsv2 = bgr_to_hsv(frame2)

            frame1_blurred = gaussian_blur(hsv1, 3)
            frame2_blurred = gaussian_blur(hsv2, 3)

            lower = np.array([35, 100, 50])
            upper = np.array([85, 255, 255])

            mask1 = in_range(frame1_blurred, lower, upper)
            mask2 = in_range(frame2_blurred, lower, upper)

            contours1 = find_contours(mask1)
            contours2 = find_contours(mask2)

            if contours1 and contours2:

                # Sort contours by area
                contours1 = sorted(contours1, key=cv2.contourArea, reverse=True)
                contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)

                # Trouver les cercles minimums
                center1, radius1 = min_enclosing_circle(contours1[0])
                center2, radius2 = min_enclosing_circle(contours2[0])

                # Use new threshold to detect the object
                if radius1 > 400 and radius2 > 400:

                    # Dessiner le cercle sur l'image
                    cv2.circle(frame1, center1, radius1, (255, 0, 0), 2)  # Cercle vert
                    cv2.putText(frame1, "Objet detecte dans la camera 1 ", (center1[0] - 50, center1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Dessiner le cercle sur l'image
                    cv2.circle(frame2, center2, radius2, (255, 0, 0), 2)  # Cercle vert
                    cv2.putText(frame2, "Objet detecte dans la camera 2 ", (center2[0] - 50, center2[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Calcul de la position 3D
                points_3d = calculate_3d_position(
                    center1, center2, K1, K2, rvecs1[0], tvecs1[0], rvecs2[0], tvecs2[0]
                )
                st.write(f"Position 3D estimée : {points_3d}")

                # Validation de l'écart vertical entre les caméras
                if validate_y_coordinates(center1, center2):
                    st.write("Les coordonnées y sont similaires.")
                else:
                    st.write("Les coordonnées y ne sont pas similaires.")

                # Calcul et affichage de l'écart horizontal
                horizontal_displacement = calculate_horizontal_displacement(center1, center2)
                st.write(f"Écart horizontal entre les centres : {horizontal_displacement} pixels")

                # Afficher les résultats
                cv2.circle(frame1, tuple(map(int, center1)), int(radius1), (0, 255, 0), 2)
                cv2.circle(frame2, tuple(map(int, center2)), int(radius2), (0, 255, 0), 2)

            # Update previous frames
            previous_frame1 = frame1.copy()
            previous_frame2 = frame2.copy()
            
            

            stframe1.image(frame1, channels="BGR")
            stframe2.image(frame2, channels="BGR")

           
            if stop:
                cap1.release()
                cap2.release()
                cv2.destroyAllWindows()
                break

    

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
