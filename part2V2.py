'''
Ce fichier contient le code pour le calibrage de la caméra 
qui est utilisé dans la partie 3 

'''



import cv2
import numpy as np

# Fonction pour générer les points 3D du checkerboard
def generate_checkerboard_points(rows, cols, square_size):
    points_3D = np.zeros((rows * cols, 3), np.float32)
    points_3D[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points_3D *= square_size
    return points_3D

# Fonction pour effectuer la calibration en temps réel pour une caméra donnée
def calibrate_camera_from_video(camera_source, rows, cols, square_size):
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"Erreur d'ouverture de la caméra {camera_source}.")
        return False, None, None, None, None

    obj_points = []  # Points 3D dans le monde réel
    img_points = []  # Points 2D dans l'image
    obj_p = np.zeros((rows * cols, 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj_p *= square_size

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur lors de la lecture de la vidéo.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        if ret:
            img_points.append(corners)
            obj_points.append(obj_p)
            cv2.drawChessboardCorners(frame, (cols, rows), corners, ret)

        frameNEw =  cv2.resize(frame, (640, 480))
        cv2.imshow("Calibration", frameNEw)
        if len(img_points) >= 10:  # 10 images minimum pour la calibration
            print("Calibration prête !")
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    if len(img_points) < 10:
        print("Pas assez d'images pour la calibration.")
        return False, None, None, None, None

    # Effectuer la calibration
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    return ret, K, dist, rvecs, tvecs

"""
ret: A boolean indicating if the calibration was successful (True) or not (False).
K (Intrinsic matrix)
dist (Distortion coefficients)
rvecs (Rotation vectors)
tvecs (Translation vectors)

"""