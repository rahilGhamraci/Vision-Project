import streamlit as st
import cv2
import numpy as np
import queue
import threading
import time
from collections import deque


from part1V1 import bgr_to_hsv, in_range, find_contours, min_enclosing_circle, gaussian_blur
from part3V1 import calculate_3d_position, detect_camera_movement, calculate_mid_point, calculate_horizontal_displacement,validate_y_coordinates
from part3V2 import load_model,detect_objects_mobinet,transform_to_new_origin
#.............................................................................................
def home_page():
    st.title("Home Page")
    st.write("Welcome to the Home Page of the Vision Project!")


#...............................................................................................
# PARTIE 1 : Conversion, détection de couleur et flou


def detecting_page():
    st.title("Detecting a 3D Object Page")

    # Input for URL
    url = st.text_input("Enter the video stream URL:", "http://192.168.100.122:8080/video")

    # Choice of detection method
    detection_method = st.radio(
        "Choose the detection method:",
        ("Color-based Detection", "MobileNet SSD Detection")
    )

    # MobileNet SSD model paths and class names
    prototxt_path = "deploy.prototxt.txt"
    model_path = "mobilenet_iter_73000.caffemodel"
    class_names = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    
    # Color range for color-based detection (green)
    lower = np.array([35, 100, 50])
    upper = np.array([85, 255, 255])
    
    if detection_method == "MobileNet SSD Detection":
        # Load MobileNet SSD model
        with st.spinner("Loading Mobinet ssd model..."):
            model = load_model(prototxt_path, model_path)
        
       

    if st.button("Start Detecting", key="start_detecting"):
        cap = cv2.VideoCapture(url)

        if not cap.isOpened():
            print('err')
            st.error("Failed to connect to the video stream. Please check the URL.")
            return
        
        
      
        while True:
            ret, frame = cap.read()

            if not ret:
                st.error("Error: Unable to read the video stream.")
                break

            if detection_method == "Color-based Detection":
                # Convert the frame to HSV
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # Apply Gaussian blur
                blurred = cv2.GaussianBlur(hsv, (3, 3), 0)

                # Create a mask for the green color
                mask = cv2.inRange(blurred, lower, upper)

                # Find contours in the mask
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    # Sort contours by area, largest first
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)

                    for contour in contours:
                        # Get the minimum enclosing circle
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        #print(radius)

                        if radius > 40:  # Minimum radius threshold
                            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                            cv2.putText(frame, "Object detected", (int(x) - 50, int(y) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            elif detection_method == "MobileNet SSD Detection":
                frame, _ = detect_objects_mobinet(frame, model, class_names, target_class="bottle")

            
           
            frame_new = cv2.resize(frame, (640, 480))
            cv2.imshow('Detection', frame_new)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break


    
#....................................................................................................

# Apporter des modifications au code déja definit dans part2V1 pour qu'il puisse etre utilisé dans l'interface

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
                        calibrated = True
                        motion_detected_event.clear()  # Reset motion detection
                        # Envoyer les résultats du calibrage à la file
                        params = {
                            "ret": ret,
                            "camera_matrix": mtx.tolist(),
                            "dist_coefficients": dist.tolist(),
                            "rotation_vectors": [vec.tolist() for vec in rvecs],
                            "translation_vectors": [vec.tolist() for vec in tvecs],
                        }
                        queue_out.put((frame_resized, params))

            # Envoyer la frame pour visualisation 
            queue_out.put((frame_resized, None))


# Streamlit UI for calibration
def calibration_page():
    st.title("Camera Calibration Interface")

    # Input for URL
    url = st.text_input("Enter the video stream URL:", "http://192.168.137.234:8080/video")

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

        # File pour la communication entre les threads 
        queue_in = queue.Queue()
        queue_out = queue.Queue()

        # Motion detection
        motion_detected_event = threading.Event()
        motion_detector = cv2.createBackgroundSubtractorMOG2(history=700, varThreshold=40, detectShadows=True)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_levels = deque(maxlen=5)  # Rolling average for motion stabilization

        # Lancement du thread qui s'occupe des calculs du calibrage 
        thread = threading.Thread(target=processing_thread,
                                  args=(queue_in, queue_out, checkerboard_width, checkerboard_height, square_size, motion_detected_event))
        thread.daemon = True
        thread.start()

        
        prev_gray_frame = None

        st.subheader("Calibration Results")
        ret1_holder = st.empty()
        K1_holder = st.empty()
        dist1_holder = st.empty()
        rvecs1_holder = st.empty()
        tvecs1_holder = st.empty()

      
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab a frame.")
                break

            frame_resized = cv2.resize(frame, (600, 400))
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

            # Ignorer la detection de mouvement pour la prmière frame récupérée
            if prev_gray_frame is None:
                prev_gray_frame = gray
                continue

            # Detecetr le mouvement en utilisant la diffrence entre les frames 
            diff_frame = cv2.absdiff(prev_gray_frame, gray)
            _, diff_thresh = cv2.threshold(diff_frame, 25, 255, cv2.THRESH_BINARY)

            # Enlever le bruit en utilisant les operations morphologiques 
            motion_mask = motion_detector.apply(gray)
            motion_mask_cleaned = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)

            # Calcul du niveau du mouvement et la moyenne de ce dernier 
            motion_level = np.sum(motion_mask_cleaned) / motion_mask_cleaned.size
            motion_levels.append(motion_level)
            average_motion = sum(motion_levels) / len(motion_levels)

            # Détécter les mouvements 
            if average_motion > 0.1:  
                motion_detected_event.set()

            # Envoyer la frame au thread pour faire les calculs 
            queue_in.put((frame_resized, gray))

            # Afficher l'image avec les corners du cherboard 
            if not queue_out.empty():
                processed_frame, calibration_params = queue_out.get()
                
                cv2.imshow('calibrage', processed_frame)


                # Affichage des informations du calibrage 
                if calibration_params:
                    
                    ret1_holder.write(f"**Ret**: {calibration_params['ret']}")
                    K1_holder.write(f"**Intrinsic Matrix (K1)**: \n{np.array(calibration_params['camera_matrix'])}")
                    dist1_holder.write(f"**Distortion Coefficients (dist1)**: \n{calibration_params['dist_coefficients']}")
                    rvecs1_holder.write(f"**Rotation Vectors (rvecs1)**: \n{np.array(calibration_params['rotation_vectors'])}")
                    tvecs1_holder.write(f"**Translation Vectors (tvecs1)**: \n{np.array(calibration_params['translation_vectors'])}")

            prev_gray_frame = gray.copy()

            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break




#..............................................................................................................

# Apporter des modifications au code de la fonction calibrate_camera_from_video 
# déja definit dans part2V2 pour qu'il puisse etre utilisé dans l'interface

from multiprocessing import Process, Manager

def calibrate_camera_from_video(name, camera_source, rows, cols, square_size):
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"Error: Unable to open camera source: {camera_source}")
        return False, None, None, None, None

    obj_points = []
    img_points = []
    obj_p = np.zeros((rows * cols, 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj_p *= square_size

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading video.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        if ret:
            img_points.append(corners)
            obj_points.append(obj_p)
            cv2.drawChessboardCorners(frame, (cols, rows), corners, ret)

        frame_new = cv2.resize(frame, (640, 480))
        cv2.imshow(name, frame_new)
        if len(img_points) >= 10:
            print("Calibration ready!")
            ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    if len(img_points) < 10:
        print("Not enough images for calibration.")
        return False, None, None, None, None

    return ret, K, dist, rvecs, tvecs

def calibrate_and_store(camera_name, camera_source, checkerboard_width, checkerboard_height, square_size, results):
    ret, K, dist, rvecs, tvecs = calibrate_camera_from_video(camera_name, camera_source, checkerboard_width, checkerboard_height, square_size)
    results[camera_name] = {"ret": ret, "K": K, "dist": dist, "rvecs": rvecs, "tvecs": tvecs}

def position_page():
    st.title("Position Page")

    url1 = st.text_input("Enter the first video stream URL:", "http://192.168.137.234:8080/video")
    url2 = st.text_input("Enter the second video stream URL:", "http://192.168.137.234:8080/video")

    # Calibration parameters
    checkerboard_width = st.number_input("Checkerboard Width:", value=9, step=1)
    checkerboard_height = st.number_input("Checkerboard Height:", value=7, step=1)
    square_size = st.number_input("Square Size (in mm):", value=20, step=1)

    detection_method = st.radio(
        "Choose the detection method:",
        ("Color-based Detection", "MobileNet SSD Detection")
    )

      


    # MobileNet SSD model paths and class names
    prototxt_path = "deploy.prototxt.txt"
    model_path = "mobilenet_iter_73000.caffemodel"
    class_names = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    frame_skip = 5  
    frame_count = 0

    lower = np.array([35, 100, 50])
    upper = np.array([85, 255, 255])

 
            

    

    

    if st.button("Start", key="start"):
         
        manager = Manager()
        calibration_results = manager.dict()
        
        #utilisation des processus pour calibrer les deux caméras en meme temps 
        process1 = Process(target=calibrate_and_store, args=("Camera 1", url1, checkerboard_width, checkerboard_height, square_size, calibration_results))
        process2 = Process(target=calibrate_and_store, args=("Camera 2", url2, checkerboard_width, checkerboard_height, square_size, calibration_results))

        process1.start()
        process2.start()

        process1.join()
        process2.join()

        cam1_results = calibration_results.get("Camera 1", {})
        cam2_results = calibration_results.get("Camera 2", {})

        if cam1_results["ret"] is None or cam2_results["ret"] is None:
            st.error("Calibration failed for one or both cameras.")
            return

        st.success("Calibration completed for both cameras!")

        ret1, K1, dist1, rvecs1, tvecs1 = cam1_results['ret'] , cam1_results['K'], cam1_results['dist'], cam1_results['rvecs'], cam1_results['tvecs']
        ret2, K2, dist2, rvecs2, tvecs2 = cam2_results['ret'] , cam2_results['K'], cam2_results['dist'], cam2_results['rvecs'], cam2_results['tvecs']

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Calibration Results For Caméra 1")
            ret1_holder = st.empty()
            K1_holder = st.empty()
            dist1_holder = st.empty()
            rvecs1_holder = st.empty()
            tvecs1_holder = st.empty()
  
            ret1_holder.write(f"**Ret**: {cam1_results['ret']}")
            K1_holder.write(f"**Intrinsic Matrix (K1)**: \n{cam1_results['K']}")
            dist1_holder.write(f"**Distortion Coefficients (dist1)**: \n{cam1_results['dist']}")
            rvecs1_holder.write(f"**Rotation Vectors (rvecs1)**: \n{cam1_results['rvecs']}")
            tvecs1_holder.write(f"**Translation Vectors (tvecs1)**: \n{cam1_results['tvecs']}")

        with col2:
            st.subheader("Calibration Results For Caméra 2")
            ret2_holder = st.empty()
            K2_holder = st.empty()
            dist2_holder = st.empty()
            rvecs2_holder = st.empty()
            tvecs2_holder = st.empty()

           
            ret2_holder.write(f"**Ret**: {ret2}")
            K2_holder.write(f"**Intrinsic Matrix (K2)**: \n{cam1_results['K']}")
            dist2_holder.write(f"**Distortion Coefficients (dist2)**: \n{cam2_results['dist']}")
            rvecs2_holder.text(f"**Rotation Vectors (rvecs2)**: \n{cam1_results['rvecs']}")
            tvecs2_holder.text(f"**Translation Vectors (tvecs2)**: \n{cam2_results['tvecs']}")

 
            

        cap1 = cv2.VideoCapture(url1)
        cap2 = cv2.VideoCapture(url2)
        
        if not cap1.isOpened() or not cap2.isOpened():
            st.error("Erreur lors de l'ouverture des caméras.")
            exit(0)

        st.subheader("Position Estimation")
        points_3d_placeholder = st.empty()
        new_origin_placeholder = st.empty()
        displacement_placeholder = st.empty()
        points_3d_transformed_placeholder = st.empty()

 
        if detection_method == "MobileNet SSD Detection":
            # Chargement du modèle SSD 
            with st.spinner("Loading Mobinet ssd model..."):
                model = load_model(prototxt_path, model_path)
        while True:
            frame_count += 1

            # Ignorer certains frames pour palier au problème de latence
            if frame_count % frame_skip != 0:
                continue

            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                st.error("Erreur lors de la lecture des caméras.")
                break

            frame1 = cv2.resize(frame1, (600, 400))
            frame2 = cv2.resize(frame2, (600, 400))

            if detection_method == 'Color-based Detection':
                # Process frames (color conversion, blurring, detection, etc.)
                hsv1 = bgr_to_hsv(frame1)
                hsv2 = bgr_to_hsv(frame2)

                frame1_blurred = gaussian_blur(hsv1, 3)
                frame2_blurred = gaussian_blur(hsv2, 3)

                mask1 = in_range(frame1_blurred, lower, upper)
                mask2 = in_range(frame2_blurred, lower, upper)

                contours1 = find_contours(mask1)
                contours2 = find_contours(mask2)

                if contours1 and contours2:
                    # Trie des contours
                    contours1 = sorted(contours1, key=cv2.contourArea, reverse=True)
                    contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)

                    # Trouver les cercles minimums
                    center1, radius1 = min_enclosing_circle(contours1[0])
                    center2, radius2 = min_enclosing_circle(contours2[0])

                    # un seuil pour determiner si il s'agit d'un objet qu'on doit détécter ou pas
                    if radius1 > 40 and radius2 > 40:
                        # Dessiner le cercle sur l'image
                        cv2.circle(frame1, center1, radius1, (255, 0, 0), 2)  # Cercle vert
                        cv2.putText(frame1, "Objet detecte dans la camera 1 ", (center1[0] - 50, center1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        # Dessiner le cercle sur l'image
                        cv2.circle(frame2, center2, radius2, (255, 0, 0), 2)  # Cercle vert
                        cv2.putText(frame2, "Objet detecte dans la camera 2 ", (center2[0] - 50, center2[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                    
                   # Validation de l'écart vertical entre les caméras
                    if center1 is not None and center2 is not None:
                        if not validate_y_coordinates(center1, center2):
                            continue
                        

                        points_3d = calculate_3d_position(center1, center2, K1, K2, rvecs1[0], tvecs1[0], rvecs2[0], tvecs2[0])
                        horizontal_displacement = calculate_horizontal_displacement(center1, center2)
                        # Calcul du nouveau origine (milieu entre les caméras)
                        new_origin = calculate_mid_point(tvecs1[0], tvecs2[0])
                        # Transformation des coordonnées 3D vers le nouveau origine
                        points_3d_transformed = transform_to_new_origin(points_3d, new_origin)
                        
       
                        points_3d_placeholder.write(f"**3D Position:** {points_3d}")
                        displacement_placeholder.write(f"**Horizontal Displacement:** {horizontal_displacement} pixels")
                        new_origin_placeholder.write(f"**New origin:** {new_origin}")
                        points_3d_transformed_placeholder.write(f"**3D Position (nouvel origin):** {points_3d_transformed}")
                        

                        cv2.circle(frame1, tuple(map(int, center1)), int(radius1), (0, 255, 0), 2)
                        cv2.circle(frame2, tuple(map(int, center2)), int(radius2), (0, 255, 0), 2)
            

                    
            elif detection_method == 'MobileNet SSD Detection':
                frame1, centers1 = detect_objects_mobinet(frame1, model , class_names)
                frame2, centers2 = detect_objects_mobinet(frame2, model , class_names)

                if centers1[0] and centers2[0] and centers1[1] and centers2[1]:
                    if not validate_y_coordinates(centers1, centers2):
                        #st.error("Les coordonnées y ne sont pas similaires. Skipping calculations.")
                        continue
    
                            

                    points_3d = calculate_3d_position(centers1, centers2, K1, K2, rvecs1[0], tvecs1[0], rvecs2[0], tvecs2[0])
                    horizontal_displacement = calculate_horizontal_displacement(centers1, centers2)
                    # Calcul du nouveau origine (milieu entre les caméras)
                    new_origin = calculate_mid_point(tvecs1[0], tvecs2[0])
                    # Transformation des coordonnées 3D vers le nouveau origine
                    points_3d_transformed = transform_to_new_origin(points_3d, new_origin)
                        
       
                    points_3d_placeholder.write(f"**3D Position:** {points_3d}")
                    displacement_placeholder.write(f"**Horizontal Displacement:** {horizontal_displacement} pixels")
                    new_origin_placeholder.write(f"**New origin:** {new_origin}")
                    points_3d_transformed_placeholder.write(f"**3D Position (nouvel origin):** {points_3d_transformed}")
                    

    
        
                
            
            cv2.imshow('Caméra 1', frame1)
            cv2.imshow('Caméra 2', frame2)

           
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap1.release()
                cap2.release()
                cv2.destroyAllWindows()
                break

#..................................................................
from deep_sort_realtime.deepsort_tracker import DeepSort # to use the model of track 
from ultralytics import YOLO # to use yolo to track

def load_yolo_model(path ="yolo-Weights/yolov5n.pt"):

    model = YOLO(path) 
    classNames = model.names  
   
    return model, classNames

def improuvemenets_page():
    st.title("Improuvement Page")
    #url = st.text_input("Enter the first video stream URL:", "http://192.168.137.234:8080/video")

    object_tracker = DeepSort()
    

    # just take the index of the object we want to detect , in our case , cell phone => 67
    phone_class_index = 67  

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  
    cap.set(4, 480)  

    if st.button("Start Detecting", key="start_detecting"):
        with st.spinner("Loading Yolo model..."):
            model, classNames = load_yolo_model()
    
        while cap.isOpened():
            success, img = cap.read()

            if not success:
                break
 
            results = model(img, stream=True)


            detections = []

            for r in results:
                # retrieve the box detection result of all objects 
                boxes = r.boxes
                # this to only detect a phone 
                # iterate all the boxes 
                for box in boxes:
                    # retrieve the classe of detection 
                    cls = int(box.cls[0])  

                    # If the detected class is 'cell phone'
                    
                    if cls == phone_class_index:
                        
                        x1, y1, x2, y2 = box.xyxy[0]
               
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # Calculate the center of the bounding box
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        # Add detection to list for DeepSORT (bbox, confidence, class)
                        detections.append(([x1, y1, x2, y2], box.conf[0], cls))

                        # DISPLAY 
              
                        # Draw bounding box 
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        cv2.circle(img, (center_x, center_y), 5, (0, 255, 0), -1) 
                
                        # Display the center position as text on the webcam
                        center_text = f"Center: ({center_x}, {center_y})"
                        org = (x1, y1 - 10)  
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 0.7
                        color = (0, 255, 0)
                        thickness = 2
              
                        cv2.putText(img, center_text, org, font, fontScale, color, thickness)


            tracks = object_tracker.update_tracks(detections, frame=img)

            # Draw a moving dot for each track (phone)
            for track in tracks:

                if not track.is_confirmed():
                        continue
                track_id = track.track_id
                ltrb = track.to_ltrb()

                # Calculate the center of the bounding box for the dot
                center_x = int((ltrb[0] + ltrb[2]) // 2)
                center_y = int((ltrb[1] + ltrb[3]) // 2)

                # Draw the dot at the center of the tracked phone object
                cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)  # Red dot for tracking

    

             # Display the image on webcam with all the added displays 
            cv2.imshow('Webcam', img)

            # press q to quit 
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break






    


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
