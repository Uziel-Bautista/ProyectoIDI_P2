import cv2
import numpy as np
import os

ruta_dataset = 'dataset'            # Carpeta con subcarpetas de usuarios y sus fotos
ruta_modelo = 'modelo_reconocimiento.xml'  # Archivo donde se guardará el modelo entrenado

# Crear el reconocedor LBPH (asegúrate de tener opencv-contrib-python instalado)
reconocedor = cv2.face.LBPHFaceRecognizer_create()

# Cargar clasificador Haar para detección de rostros
detector_rostro = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

imagenes_entrenamiento = []
ids_usuarios = []
nombres = []
id_actual = 0

for carpeta_usuario in os.listdir(ruta_dataset):
    ruta_usuario = os.path.join(ruta_dataset, carpeta_usuario)

    if not os.path.isdir(ruta_usuario):
        continue

    print(f"[INFO] Procesando usuario: {carpeta_usuario}")
    nombres.append(carpeta_usuario)

    for archivo in os.listdir(ruta_usuario):
        ruta_imagen = os.path.join(ruta_usuario, archivo)
        imagen_color = cv2.imread(ruta_imagen)

        if imagen_color is None:
            print(f"[WARN] No se pudo leer la imagen: {ruta_imagen}")
            continue

        # Convertir a escala de grises
        imagen_gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)

        # Detectar rostros en la imagen
        rostros = detector_rostro.detectMultiScale(imagen_gris, scaleFactor=1.1, minNeighbors=5)
        print(f"    Imagen: {archivo} - Rostros detectados: {len(rostros)}")

        for (x, y, w, h) in rostros:
            rostro_recortado = imagen_gris[y:y+h, x:x+w]
            imagenes_entrenamiento.append(rostro_recortado)
            ids_usuarios.append(id_actual)

    id_actual += 1

if len(imagenes_entrenamiento) == 0:
    print("[ERROR] No se detectaron rostros para entrenar. Revisa el dataset.")
    exit(1)

print("[INFO] Entrenando el modelo...")
reconocedor.train(imagenes_entrenamiento, np.array(ids_usuarios))
reconocedor.save(ruta_modelo)

# Guardar los nombres de los usuarios en un archivo para usar luego en el reconocimiento
with open('nombres_usuarios.txt', 'w') as f:
    for nombre in nombres:
        f.write(nombre + '\n')

print("[INFO] Modelo entrenado y guardado en 'modelo_reconocimiento.xml'")
