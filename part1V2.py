'''
Ce fichier contient le code pour la détection d'un objet en utilisant le modèle mobinet ssd

'''

import cv2

# Chemin vers les fichiers du modèle MobileNet-SSD
prototxt_path = "deploy.prototxt.txt"
model_path = "mobilenet_iter_73000.caffemodel"

# Charger le modèle pré-entraîné
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Classe cible à détecter (par exemple : 'bottle')
target_class = "bottle"

# Liste des classes détectables par MobileNet-SSD
class_names = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Initialiser la capture vidéo
PHONE_CAMERA_URL = "http://192.168.226.189:8080/video"
cap = cv2.VideoCapture(PHONE_CAMERA_URL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prétraiter l'image pour MobileNet-SSD
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)

    # Effectuer la détection
    detections = net.forward()

    detected = False

    # Parcourir les détections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Ignorer les détections avec une faible confiance
        if confidence > 0.5: # Seuil de confiance
            class_id = int(detections[0, 0, i, 1])
            class_name = class_names[class_id]

            # Vérifier si la classe correspond à l'objet cible
            if class_name == target_class:
                detected = True

                # Extraire les coordonnées de la boîte englobante
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                x1, y1, x2, y2 = box.astype("int")

                # Dessiner le rectangle et afficher les informations
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                cv2.putText(frame, f"Position: ({cx}, {cy})", (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, f"{class_name}: {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                break

    # Afficher le flux vidéo
    frame_resized = cv2.resize(frame, (640, 480))
    cv2.imshow("Object Detection - MobileNet-SSD", frame_resized)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()