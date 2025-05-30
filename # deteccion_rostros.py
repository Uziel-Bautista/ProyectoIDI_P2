import cv2
import os

# Nombre del usuario (puede ser un ID o nombre real)
nombre_usuario = input("Ingresa el nombre del usuario: ")
ruta = f'dataset/{nombre_usuario}'

# Crear carpeta si no existe
if not os.path.exists(ruta):
    os.makedirs(ruta)

# Cargar clasificador Haar
detector_rostro = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Iniciar cámara
camara = cv2.VideoCapture(0)
conteo = 0
max_rostros = 30  # Número de imágenes a capturar

print("[INFO] Iniciando captura. Presiona 'q' para salir.")

while True:
    ret, frame = camara.read()
    if not ret:
        break

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = detector_rostro.detectMultiScale(gris, 1.3, 5)

    for (x, y, w, h) in rostros:
        conteo += 1
        rostro = gris[y:y+h, x:x+w]
        cv2.imwrite(f"{ruta}/rostro_{conteo}.jpg", rostro)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'Captura {conteo}/{max_rostros}', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Captura de rostros', frame)

    # Terminar si se presiona 'q' o si se capturan suficientes imágenes
    if cv2.waitKey(1) & 0xFF == ord('q') or conteo >= max_rostros:
        break

print("[INFO] Captura finalizada.")
camara.release()
cv2.destroyAllWindows()
