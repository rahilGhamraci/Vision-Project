'''
Ce fichier contient le code de la partie 3 :
calcul de la postion 3D ,en utilisant la detection par couleurs de l'objet

'''

import cv2
import numpy as np
from multiprocessing import Process, Manager

# PARTIE 1 : Conversion, détection de couleur et flou
from part1V1 import bgr_to_hsv, in_range, find_contours, min_enclosing_circle, gaussian_blur

# PARTIE 2 : Calibration de la caméra
from part2V2 import calibrate_camera_from_video

def calculate_3d_position(p1, p2, K1, K2, rvec1, tvec1, rvec2, tvec2):
    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)

    P1 = np.dot(K1, np.hstack((R1, tvec1)))
    P2 = np.dot(K2, np.hstack((R2, tvec2)))

    p1_h = np.array([[p1[0]], [p1[1]]], dtype=np.float32)
    p2_h = np.array([[p2[0]], [p2[1]]], dtype=np.float32)

    points_4d = cv2.triangulatePoints(P1, P2, p1_h, p2_h)
    points_3d = points_4d[:3] / points_4d[3]
    return points_3d.T

def transform_to_new_origin(points_3d, new_origin):
    """
    Transforme les coordonnées 3D pour les exprimer dans un nouveau système de coordonnées
    dont l'origine est définie par 'new_origin'.

    Arguments :
    points_3d (np.ndarray) : Coordonnées 3D des points dans le système actuel.
    new_origin (np.ndarray) : Coordonnées de la nouvelle origine dans le système actuel.

    Retourne :
    np.ndarray : Coordonnées 3D transformées.
    """
    # Assurer que les coordonnées sont des tableaux 1D
    points_3d = np.asarray(points_3d).flatten()
    new_origin = np.asarray(new_origin).flatten()
    return points_3d - new_origin

def detect_camera_movement(previous_frame, current_frame, change_ratio_threshold=0.01):
    """
    Détecte un mouvement significatif de la caméra en comparant deux images successives.
    Ignore les différences mineures causées par le bruit ou des vibrations naturelles.
    """
    # Calcul de la différence absolue
    diff = cv2.absdiff(previous_frame, current_frame)

    # Conversion en niveaux de gris et application d'un flou gaussien pour réduire le bruit
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    gray_diff = cv2.GaussianBlur(gray_diff, (5, 5), 0)

    # Application d'un seuil binaire stricte
    _, thresh = cv2.threshold(gray_diff, 50, 255, cv2.THRESH_BINARY)

    # Comptage des pixels non zéro et calcul du ratio de changement
    non_zero_count = cv2.countNonZero(thresh)
    frame_area = gray_diff.shape[0] * gray_diff.shape[1]
    change_ratio = non_zero_count / frame_area

    print(f"Non-zero count: {non_zero_count}, Change ratio: {change_ratio:.5f}")
    return change_ratio > change_ratio_threshold

def calculate_mid_point(tvec1, tvec2):
    """
    Calcule le point à mi-distance entre les centres des deux caméras.
    """
    relative_vector = tvec2 - tvec1
    mid_point = tvec1 + relative_vector / 2
    return mid_point

def calculate_horizontal_displacement(center1, center2):
    """
    Calcule l'écart horizontal (b) entre les deux centres des objets dans les deux vues de caméra.
    """
    return abs(center1[0] - center2[0])

# Fonction pour valider si les coordonnées y sont similaires dans les deux caméras
def validate_y_coordinates(center1, center2, tolerance=5):
    """
    Vérifie si les coordonnées y des deux centres sont identiques dans un certain seuil de tolérance.
    """
    return abs(center1[1] - center2[1]) <= tolerance

def calibrate_and_store(camera_name, camera_source, checkerboard_width, checkerboard_height, square_size, results):
    ret, K, dist, rvecs, tvecs = calibrate_camera_from_video(camera_name, camera_source, checkerboard_width, checkerboard_height, square_size)
    results[camera_name] = {"ret": ret, "K": K, "dist": dist, "rvecs": rvecs, "tvecs": tvecs}

#code pour tester la partie 3 dans le terminal 
'''

if __name__ == "__main__":
    # Calibration Parameters
    rows = 7
    cols = 9
    square_size = 20  # en millimètres



    # Camera URLs
    PHONE_CAMERA_URL1 = "http://192.168.100.122:8080/video"
    PHONE_CAMERA_URL2 = "http://192.168.100.122:8080/video"

    manager = Manager()
    calibration_results = manager.dict()

    #utilisation des processus pour calibrer les deux caméras en meme temps 
    process1 = Process(target=calibrate_and_store, args=("Camera 1", PHONE_CAMERA_URL1, rows, cols, square_size, calibration_results))
    process2 = Process(target=calibrate_and_store, args=("Camera 2", PHONE_CAMERA_URL2, rows, cols, square_size, calibration_results))

    process1.start()
    process2.start()

    process1.join()
    process2.join()

    cam1_results = calibration_results.get("Camera 1", {})
    cam2_results = calibration_results.get("Camera 2", {})

    if cam1_results["ret"] is None or cam2_results["ret"] is None:
            print("Calibration failed for one or both cameras.")
            exit(1)

    print("Calibration completed for both cameras!")

    ret1, K1, dist1, rvecs1, tvecs1 = cam1_results['ret'] , cam1_results['K'], cam1_results['dist'], cam1_results['rvecs'], cam1_results['tvecs']
    ret2, K2, dist2, rvecs2, tvecs2 = cam2_results['ret'] , cam2_results['K'], cam2_results['dist'], cam2_results['rvecs'], cam2_results['tvecs']


    # Camera Initialization
    cap1 = cv2.VideoCapture(PHONE_CAMERA_URL1)
    cap2 = cv2.VideoCapture(PHONE_CAMERA_URL2)

    if not cap1.isOpened() or not cap2.isOpened():
      
      print("Erreur lors de l'ouverture des caméras.")
      exit(1)


    # Frame skipping
    frame_skip = 5  
    frame_count = 0

    previous_frame1 = None
    previous_frame2 = None

    movement2 = False
    movement1 = False

    # Flag pour le suivi du calibrage
    recalibration_done = False





    while True:
        frame_count += 1

        # Skip frames
        if frame_count % frame_skip != 0:
            continue

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("Erreur lors de la lecture des caméras.")
            break

        if previous_frame1 is not None and previous_frame2 is not None:
            movement1 = detect_camera_movement(previous_frame1, frame1, change_ratio_threshold=0.4)
            movement2 = detect_camera_movement(previous_frame2, frame2, change_ratio_threshold=0.4)

            if (movement1 or movement2) and not recalibration_done:
                print("Mouvement détecté. Recalibration des caméras...")
                ret1, K1, dist1, rvecs1, tvecs1 = calibrate_camera_from_video(PHONE_CAMERA_URL1, rows, cols, square_size)
                ret2, K2, dist2, rvecs2, tvecs2 = calibrate_camera_from_video(PHONE_CAMERA_URL2, rows, cols, square_size)
                recalibration_done = True  # Mark recalibration as done
                movement2 = False
                movement1 = False
                if not ret1 or not ret2:
                    print("Erreur lors de la recalibration.")
                    break
            elif not movement1 and not movement2:
                recalibration_done = False  # Reset the flag when no movement is detectedq

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
                print('je rentre dans la boucle de detection')
                cv2.circle(frame1, center1, radius1, (255, 0, 0), 2)  # Cercle vert
                cv2.putText(frame1, "Objet detecte dans la camera 1 ", (center1[0] - 50, center1[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                # Dessiner le cercle sur l'image
                cv2.circle(frame2, center2, radius2, (255, 0, 0), 2)  # Cercle vert
                cv2.putText(frame2, "Objet detecte dans la camera 2 ", (center2[0] - 50, center2[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
            # Validation de l'écart vertical entre les caméras
            if validate_y_coordinates(center1, center2):
                print("Les coordonnées y sont similaires.")
            else:
                print("Les coordonnées y ne sont pas similaires.")
                continue
        
            # Calcul de la position 3D
            points_3d = calculate_3d_position(
                center1, center2, K1, K2, rvecs1[0], tvecs1[0], rvecs2[0], tvecs2[0]
            )
            print(f"Position 3D estimée : {points_3d}")

            # Calcul de la nouvelle origine (milieu entre les caméras)
            new_origin = calculate_mid_point(tvecs1[0], tvecs2[0])

            # Transformation des coordonnées 3D vers la nouvelle origine
            points_3d_transformed = transform_to_new_origin(points_3d, new_origin)
            print(f"Position 3D (nouvelle origine) : {points_3d_transformed}")


            # Calcul et affichage de l'écart horizontal
            horizontal_displacement = calculate_horizontal_displacement(center1, center2)
            print(f"Écart horizontal entre les centres : {horizontal_displacement} pixels")

            cv2.circle(frame1, tuple(map(int, center1)), int(radius1), (0, 255, 0), 2)
            cv2.circle(frame2, tuple(map(int, center2)), int(radius2), (0, 255, 0), 2)


    
   
        # Update previous frames
        previous_frame1 = frame1.copy()
        previous_frame2 = frame2.copy()

   
        cv2.imshow('Caméra 1', frame1)
        cv2.imshow('Caméra 2', frame2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


'''


    
