import cv2
import json
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Cargar el modelo guardado
model = load_model('C:/Users/juan.ochoa/OneDrive - INMEL INGENIERIA SAS/Documentos/Python_Codigos/Clasificar tattoo/modelo_tatuaje.h5')
print("Modelo cargado correctamente.")

# Cargar class_indices
with open('C:/Users/juan.ochoa/OneDrive - INMEL INGENIERIA SAS/Documentos/Python_Codigos/Clasificar tattoo/class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Convertir índices en una lista de clases
classes = list(class_indices.keys())
print("Clases disponibles:", classes)

# Inicializar la cámara
cap = cv2.VideoCapture(0)  # Usa 0 para la cámara predeterminada

if not cap.isOpened():
    print("No se pudo acceder a la cámara.")
    exit()

print("Presiona 'q' para salir.")

frame_count = 0  # Contador de frames
predicted_class = None  # Clase detectada

while True:
    # Leer un frame de la cámara
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el frame de la cámara.")
        break

    # Procesar cada 10 cuadros (para mejorar rendimiento)
    frame_count += 1
    if frame_count % 10 == 0:
        # Redimensionar y preprocesar la imagen para el modelo
        img = cv2.resize(frame, (128, 128))  # Redimensionar al tamaño que espera el modelo
        img_array = img_to_array(img) / 255.0  # Normalizar los valores
        img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión para el modelo

        # Hacer la predicción
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)

        # Obtener la clase detectada
        predicted_class = classes[predicted_class_index[0]]

    # Mostrar la clase en la ventana
    text = f"Clase detectada: {predicted_class}" if predicted_class else "Detectando..."
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar el frame en una ventana
    cv2.imshow("Camara - Clasificador de Tatuajes", frame)

    # Salir si presionas 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()