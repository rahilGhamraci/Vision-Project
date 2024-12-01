'''
Ce fichier contient le code pour la détection d'un objet par couleurs 

'''

import cv2
import numpy as np

# Conversion BGR vers HSV
def bgr_to_hsv(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

# Application d'un flou gaussien pour réduire le bruit
def gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Détection de couleur avec cv2.inRange (plus rapide et plus robuste)
def in_range(hsv, lower, upper):
    mask = cv2.inRange(hsv, lower, upper)  # Utiliser OpenCV pour la détection de plage
    return mask

# Détection des contours avec cv2.findContours
def find_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Trouver le cercle minimum qui entoure un contour
def min_enclosing_circle(contour):
    center, radius = cv2.minEnclosingCircle(contour)
    return (int(center[0]), int(center[1])), int(radius)

#code utilisé pour tester la detection seule dans le terminal

""" cap = cv2.VideoCapture("http://192.168.226.189:8080/video")  # Indiquer l'index de la caméra (0 pour la caméra par défaut)

# Vérifier si la caméra s'est ouverte correctement
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la caméra")
    exit()

# Plage de couleurs pour détecter le vert (peut être ajustée selon votre objet)
lower = np.array([35, 100, 50])  # Valeurs ajustées
upper = np.array([85, 255, 255])  # Valeurs ajustées


while True:
    # Lire un cadre depuis la caméra
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
                cv2.circle(frame, center, radius, (255, 0, 0), 2)  # Cercle 
                cv2.putText(frame, "Objet detecte", (center[0] - 50, center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Afficher l'image avec le cercle dessiné
    cv2.imshow("Détection d'objet", frame)

    # Quitter si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la caméra et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()   """