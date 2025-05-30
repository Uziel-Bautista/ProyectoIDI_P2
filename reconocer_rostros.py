import cv2
import numpy as np

# Cargar el modelo entrenado y nombres
reconocedor = cv2.face.LBPHFaceRecognizer_create()
reconocedor.read('modelo_reconocimiento.xml')

with open('nombres_usuarios.txt', 'r') as f:
    nombres = [linea.strip() for linea in f.readlines()]

# Cargar el detector Haar para rostros
detector_rostro = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Abrir la cámara (0 es la webcam por defecto)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

print("Presiona ESC para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = detector_rostro.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in rostros:
        rostro_roi = gris[y:y+h, x:x+w]
        id_predicho, confianza = reconocedor.predict(rostro_roi)

        # Mostrar el nombre solo si confianza es baja (menor a cierto umbral)
        if confianza < 70:
            nombre = nombres[id_predicho]
            texto = f"{nombre} ({int(confianza)})"
        else:
            texto = "Desconocido"

        # Dibujar rectángulo y texto en la ventana
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, texto, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Reconocimiento Facial", frame)

    # Salir con tecla ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
