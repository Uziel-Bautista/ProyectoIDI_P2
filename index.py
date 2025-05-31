import tkinter as tk
from tkinter import ttk
import subprocess

def ejecutar_registro():
    subprocess.run(["python", "deteccion_rostros.py"])

def ejecutar_entrenamiento():
    subprocess.run(["python", "entrenar_modelo.py"])

def ejecutar_reconocimiento():
    subprocess.run(["python", "reconocer_rostros.py"])

# Crear ventana principal
ventana = tk.Tk()
ventana.title("Sistema de Reconocimiento Facial")
ventana.geometry("400x300")
ventana.configure(bg="#1f1f2e")

# Estilo para los botones
style = ttk.Style()
style.configure("TButton", font=("Helvetica", 14), padding=10)
style.theme_use("clam")

titulo = tk.Label(
    ventana, text="Reconocimiento Facial", font=("Helvetica", 20), fg="white", bg="#1f1f2e"
)
titulo.pack(pady=20)

btn_registrar = ttk.Button(ventana, text="Registrar Rostro", command=ejecutar_registro)
btn_registrar.pack(pady=10)

btn_entrenar = ttk.Button(ventana, text="Entrenar Modelo", command=ejecutar_entrenamiento)
btn_entrenar.pack(pady=10)

btn_reconocer = ttk.Button(ventana, text="Reconocer Rostro", command=ejecutar_reconocimiento)
btn_reconocer.pack(pady=10)

ventana.mainloop()
