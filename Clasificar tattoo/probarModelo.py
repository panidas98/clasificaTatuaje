import os
import json
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

ruta_carpeta_imagenes = 'C:/Users/juan.ochoa/clasificaTatuaje/Clasificar tattoo/Tattoo_Types-4'
ruta_destino = 'C:/Users/juan.ochoa/clasificaTatuaje/Clasificar tattoo'
# Cargar el modelo guardado
model = load_model(f'{ruta_destino}/modelo_tatuaje.h5')
print("Modelo cargado correctamente.")

# Cargar class_indices
with open(f'{ruta_destino}/class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Convertir índices en una lista de clases
classes = list(class_indices.keys())
print("Clases disponibles:", classes)

# Ruta de la imagen que quieres probar
# img_path = "C:/Users/juan.ochoa/Downloads/tradi.JPG"
img_path = [f'{ruta_destino}/imagenTradiPrueba.JPG'
            ,f'{ruta_destino}/imagenRealismoPrueba.jpg'
            ,f'{ruta_destino}/acuarelaPrueba.jpg'
]

def procesarImagen(img_path):
    # Cargar y preprocesar la imagen
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)

    # Hacer la predicción
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    # Imprimir la clase predicha
    print("Clase detectada:", classes[predicted_class[0]])

# Recorrer la lista de imagenes
for i in img_path:
    tipo = i.split('/')[-1].split('.')[0]
    print(f'Procesando una imagen de {tipo}')
    procesarImagen(i)