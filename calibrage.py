import cv2
import numpy as np
import queue
import threading
from scipy.optimize import least_squares
import time
from collections import deque

# Pour calculer l'erreur de la reprojection 
def reprojection_error(params, objpoints, imgpoints):
    """
    Calcul de l'erreur de reprojection pour ajuster les paramètres intrinsèques
    
    Paramètres :
        params : Liste des paramètres intrinsèques (fx, fy, cx, cy)
        objpoints : Liste des points d'objet en 3D
        imgpoints : Liste des points correspondants en 2D dans l'image
    
    Retour :
        Une liste des erreurs pour chaque point.
    """
    fx, fy, cx, cy = params[:4]
    
    # creation de la matrice avec les paramètres intrinsèques
    intrinsic_matrix = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]], dtype=np.float32)
    
    errors = []

    for objp, imgp in zip(objpoints, imgpoints):
        uv_projected = []

        if len(objp) == 0 or len(imgp) == 0:
            print("Warning: Empty object or image points detected.")
            continue

        for p in objp:
            X, Y, Z = p
            #  Z = 0 pour les plans  2D  ( nous sommes entrain d'utiliser un checkboard de 2D)
            u = intrinsic_matrix[0, 0] * X + intrinsic_matrix[0, 2]
            v = intrinsic_matrix[1, 1] * Y + intrinsic_matrix[1, 2]
            uv_projected.append([u, v])

        uv_projected = np.array(uv_projected)

        if len(uv_projected) != len(imgp):
            print(f"Warning: Mismatch in points. Projected: {len(uv_projected)}, Image: {len(imgp)}")
            continue

        # Calculer la distance euclidienne comme moyen de mesurer l'erreur 
        errors.append(np.linalg.norm(imgp.reshape(-1, 2) - uv_projected, axis=1))

    return np.concatenate(errors)


# Fonction qui calcule les parametres intrinsèques
def calibrate_camera(objpoints, imgpoints, image_shape):

    """
    Calibrage de la caméra pour déterminer les paramètres intrinsèques.
    
    Paramètres :
        objpoints : Points d'objet 3D.
        imgpoints : Points image 2D.
        image_shape : Dimensions de l'image (hauteur, largeur).
    
    Retour :
        La matrice intrinsèque de la caméra.
    """
    h, w = image_shape
    initial_guess = [w / 2, w / 2, w / 2, h / 2]  # Initialisation des parametres 
    result = least_squares(reprojection_error, initial_guess, args=(objpoints, imgpoints))
    fx, fy, cx, cy = result.x
    intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return intrinsic_matrix



# fonction qui detect si la caméra bouge 
def detect_intentional_movement(prev_frame, current_frame, 
                                motion_threshold=15, 
                                stability_threshold=5, 
                                window_size=5):
    
    """
    Détecte les mouvements intentionnels tout en ignorant les petits tremblements.
    
    Paramètres :
        prev_frame : Image précédente en niveaux de gris.
        current_frame : Image actuelle en niveaux de gris.
        motion_threshold : Seuil pour détecter un mouvement significatif.
        stability_threshold : Seuil pour détecter les changements de cadre.
        window_size : Taille de la fenêtre pour lisser les mouvements détectés.
    
    Retour :
        bool : True si un mouvement intentionnel est détecté, sinon False.
    """
    
    #  Calculer la difference entre les deux frame pour voir si il y'a eu un changement significatif 
    frame_diff = cv2.absdiff(prev_frame, current_frame)
    avg_frame_diff = np.mean(frame_diff)

    # Initialisation de  keypoints pour le  Lucas-Kanade Optical Flow
    if not hasattr(detect_intentional_movement, "keypoints_prev"):
        detect_intentional_movement.keypoints_prev = cv2.goodFeaturesToTrack(
            prev_frame, maxCorners=100, qualityLevel=0.3, minDistance=7
        )

    # Calculer l'optical flow en utilisant  Lucas-Kanade
    keypoints_curr, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_frame, current_frame, 
        detect_intentional_movement.keypoints_prev, None
    )

    # Filter les points valides
    valid_keypoints_prev = detect_intentional_movement.keypoints_prev[status == 1]
    valid_keypoints_curr = keypoints_curr[status == 1]

    # Calculer les magnitudes du mouvement  pour les points valides
    motion_vectors = valid_keypoints_curr - valid_keypoints_prev
    magnitudes = np.linalg.norm(motion_vectors, axis=1)
    avg_motion = np.mean(magnitudes)

    # mettre ajour les  keypoints pour la future frame 
    detect_intentional_movement.keypoints_prev = valid_keypoints_curr.reshape(-1, 1, 2)

    # garder un historique des mouvements  
    if not hasattr(detect_intentional_movement, "motion_history"):
        detect_intentional_movement.motion_history = deque(maxlen=window_size)
    detect_intentional_movement.motion_history.append(avg_motion)

    # Smooth mouvement en calculant la moyenne 
    smoothed_motion = np.mean(detect_intentional_movement.motion_history)

    # Detecter un mouvement si i y a eu un changement sigificatif ( qqui depasse le seuli que nous avons defeni) 
    movement_detected = (
        smoothed_motion > motion_threshold or 
        avg_frame_diff > stability_threshold
    )
    return movement_detected




# thread qui gère les calculs pour avoir un affichage de la video plus rapide et "smoth"
def processing_thread(queue_in, queue_out, checkerboard_width, checkerboard_height, square_size):
    objp = np.array([[x, y, 0] for y in range(checkerboard_height) for x in range(checkerboard_width)], dtype=np.float32)
    objp *= square_size

    objpoints = [] #pour stocker les points 3D
    imgpoints = [] #pour les points 2D correspondants 
    intrinsic_matrix = None
    dist_coeffs = np.zeros((4, 1))
    prev_frame_gray = None
    calibration_done = False

    while True:
        if not queue_in.empty():
            frame_resized, gray = queue_in.get()

            ret, corners = cv2.findChessboardCorners(gray, (checkerboard_width, checkerboard_height), None)
            if ret:
                imgpoints.append(corners) # recpérer les point en 2D
                objpoints.append(objp) # récupérer les points en 3D 

                # detecter les internal corners sur la grille 
                frame_resized = cv2.drawChessboardCorners(frame_resized, (checkerboard_width, checkerboard_height), corners, ret)
            
            # si nous avons pas déja calibrer et nnus avons assez de points pour le fiiar on procède au calcul 
            if not calibration_done and len(objpoints) > 0 and len(imgpoints) > 0:
                print("Calibrating camera...")
                image_shape = gray.shape[::-1]
                intrinsic_matrix = calibrate_camera(objpoints, imgpoints, image_shape)
                calibration_done = True
                print("Intrinsic Matrix:\n", intrinsic_matrix)

                # Compute extrinsics
                ret, R_vec, T, _ = cv2.solvePnPRansac(objp, imgpoints[0], intrinsic_matrix, dist_coeffs)
                R, _ = cv2.Rodrigues(R_vec)
            
                print("Extrinsic Parameters - Translation:\n", T)
                print("Extrinsic Parameters - Rotation Matrix:\n", R)
            
            #detecter le mouvement pour refair le calibrage 
            if calibration_done and prev_frame_gray is not None:
                # s'assurer que les deux frames ont le meme shape 
                if prev_frame_gray.shape != gray.shape:
                    prev_frame_gray = gray
                    continue
                # choisir une valeur qui optimse la detection de mouvements de la caméra 
                if detect_intentional_movement(prev_frame_gray, gray, motion_threshold=25, stability_threshold=15.0):
                     print("Significant movement detected. Recalibrating...")
                     calibration_done = False
                     objpoints.clear()
                     imgpoints.clear()
                else:
                    print("No mouvement detected")


                

            # mettre a jour prev_frame_gray pour la prochaine iteration 
            prev_frame_gray = gray
            # enyoyer les resultats au main thread 
            queue_out.put((frame_resized, calibration_done, intrinsic_matrix))


# Main function
def main():
    url = "http://192.168.100.4:8080/video" # url donnée par l'app IP webcamera sur android ( voir l'equivalent pour iphone)
    cap = cv2.VideoCapture(url)

    while not cap.isOpened(): #pour eviter le time out 
        print("Reconnecting to the camera...")
        cap = cv2.VideoCapture(url)
        time.sleep(2)  # attendre avant de reesseyer 
    
    '''
    fps : le nombre de frame par secondes , 
    nous avons besoin de cette var dans le wiatkey par la suite , 
    pour avoir la frame affichée par imshow et la video dans la meme vitese 
    '''
    fps = cap.get(cv2.CAP_PROP_FPS) 
    checkerboard_width = 9 # nombre de colonnes -1 ( on prend le checkboard horizentalement )
    checkerboard_height = 7 # nombre de lignes -1 ( on prend le checkboard horizentalement )
    square_size = 20 # la taille en mm de chaque case 
    

    # necessaires à la communication des threads 
    queue_in = queue.Queue()
    queue_out = queue.Queue()

    # Slancement de processing_thread 
    thread = threading.Thread(target=processing_thread, args=(queue_in, queue_out, checkerboard_width, checkerboard_height, square_size))
    thread.daemon = True
    thread.start()

    while True:
        ret, frame = cap.read() # capturer une frame 
        if not ret:
            print("Error: Failed to grab a frame.")
            break

        frame_resized = cv2.resize(frame, (640, 480)) # resuire le size de la frame 
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY) #convertir en gray scal car c'est le format exigé par findChessboardCorners

        queue_in.put((frame_resized, gray)) # envoyer la frame au thread qui fait les calculs 

        if not queue_out.empty():
            frame_resized, calibration_done, intrinsic_matrix = queue_out.get()
            cv2.imshow('Camera Calibration', frame_resized) # afficher l'image 

        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord('q'):
            print("Key q pressed. Exiting.")
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()
