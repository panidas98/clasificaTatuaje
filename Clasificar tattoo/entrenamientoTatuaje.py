import os
import json
import keras
import shutil
import pandas as pd
import tensorflow as tf
# Significa el '3' que solo se mostrarán errores fatales y se ocultarán advertencias y mensajes informativos.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# VALIDAR VERSIONES DE KERAS Y TENSORFLOW
print(keras.__version__)
print(tf.__version__)

# # Esto descarga el dataset, pero pues si ya se descargó se puede comentar este bloque.
# from roboflow import Roboflow
# rf = Roboflow(api_key="mlT63rhUs7OKHz2LKNMC")
# project = rf.workspace("tattoo-d6iux").project("tattoo_types")
# version = project.version(4)
# dataset = version.download("multiclass")

# Preparar los datos
# VALIDAR SI LAS CARPETAS EXISTEN
def count_images_in_directory(directory):
    images = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                full_path = os.path.join(root, file)
                images.append(full_path)
    print(f"Total images in {directory}: {len(images)}")
    # print(images)
    return images

ruta_carpeta_imagenes = 'C:/Users/juan.ochoa/clasificaTatuaje/Clasificar tattoo/Tattoo_Types-4'
ruta_destino = 'C:/Users/juan.ochoa/clasificaTatuaje/Clasificar tattoo'
# Definir las rutas donde están las imagenes de entrenamiento y las válidas.
rutaTrain = f"{ruta_carpeta_imagenes}/train"
rutaValid = f"{ruta_carpeta_imagenes}/valid"

print("Training images:")
a = count_images_in_directory(rutaTrain)

print("\nValidation images:")
b = count_images_in_directory(rutaValid)

# Organizar las imagenes de la carpeta de train

# Ruta al archivo CSV y a la carpeta de imágenes
csv_path = f"{ruta_carpeta_imagenes}/train/_classes.csv"
source_dir = a  # Lista con las rutas de las imágenes

# Cargar el archivo CSV
data = pd.read_csv(csv_path)

# Limpiar nombres de columnas eliminando espacios iniciales si los hay
data.columns = data.columns.str.strip()

# Crear un directorio de destino para las imágenes organizadas
train_dir = f'{ruta_destino}/train_org'
os.makedirs(train_dir, exist_ok=True)

# Procesar cada fila del CSV
for _, row in data.iterrows():
    filename = row['filename']
    # Determinar la clase a partir de las columnas binarias
    classes = row[1:]  # Todas las columnas excepto 'filename'
    target_classes = classes[classes == 1].index.tolist()

    # Validar si existe al menos una clase para la imagen
    if not target_classes:
        print(f"No se encontró clase para la imagen {filename}.")
        continue

    # Buscar la ruta completa de la imagen en la lista de rutas
    image_path = next((path for path in source_dir if filename in path), None)

    if not image_path:
        print(f"No se encontró la imagen {filename} en la lista.")
        continue

    # Mover o copiar la imagen a la carpeta de cada clase correspondiente
    for cls in target_classes:
        class_dir = os.path.join(train_dir, cls)
        os.makedirs(class_dir, exist_ok=True)  # Crear la carpeta si no existe

        dest_path = os.path.join(class_dir, os.path.basename(image_path))
        shutil.copy(image_path, dest_path)  # Cambia `copy` a `move` si deseas mover
        # print(f"Imagen {filename} copiada a {class_dir}.")

# Organizar carpeta de valid por clase
# Ruta al archivo CSV y a la carpeta de imágenes
csv_path = f"{rutaValid}/_classes.csv"
source_dir = b  # Lista con las rutas de las imágenes

# Cargar el archivo CSV
data = pd.read_csv(csv_path)

# Limpiar nombres de columnas eliminando espacios iniciales si los hay
data.columns = data.columns.str.strip()

# Crear un directorio de destino para las imágenes organizadas
valid_dir = f'{ruta_destino}/valid_org'
os.makedirs(valid_dir, exist_ok=True)

# Procesar cada fila del CSV
for _, row in data.iterrows():
    filename = row['filename']
    # Determinar la clase a partir de las columnas binarias
    classes = row[1:]  # Todas las columnas excepto 'filename'
    target_classes = classes[classes == 1].index.tolist()

    # Validar si existe al menos una clase para la imagen
    if not target_classes:
        print(f"No se encontró clase para la imagen {filename}.")
        continue

    # Buscar la ruta completa de la imagen en la lista de rutas
    image_path = next((path for path in source_dir if filename in path), None)

    if not image_path:
        print(f"No se encontró la imagen {filename} en la lista.")
        continue

    # Mover o copiar la imagen a la carpeta de cada clase correspondiente
    for cls in target_classes:
        class_dir = os.path.join(valid_dir, cls)
        os.makedirs(class_dir, exist_ok=True)  # Crear la carpeta si no existe

        dest_path = os.path.join(class_dir, os.path.basename(image_path))
        shutil.copy(image_path, dest_path)  # Cambia `copy` a `move` si deseas mover
        # print(f"Imagen {filename} copiada a {class_dir}.")

# Entrenar el modelo

# Directorios de entrenamiento y validación
train_dir = train_dir
valid_dir = valid_dir

# Crear el generador de datos
train_datagen = ImageDataGenerator(
    rescale=1.0/255,          # Escalar valores de píxeles entre 0 y 1
    rotation_range=20,        # Rotación aleatoria
    width_shift_range=0.2,    # Desplazamiento horizontal
    height_shift_range=0.2,   # Desplazamiento vertical
    shear_range=0.2,          # Deformación en corte
    zoom_range=0.2,           # Zoom aleatorio
    horizontal_flip=True,     # Volteo horizontal
    fill_mode="nearest"       # Rellenar píxeles vacíos
)

valid_datagen = ImageDataGenerator(rescale=1.0/255)

# Generador para el conjunto de entrenamiento (desde carpetas organizadas)
train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(128, 128),    # Tamaño de las imágenes
    batch_size=32,             # Tamaño del lote
    class_mode="categorical"   # Clasificación multicategoría
)

# Generador para el conjunto de validación (desde carpetas organizadas)
valid_generator = valid_datagen.flow_from_directory(
    directory=valid_dir,
    target_size=(128, 128),    # Tamaño de las imágenes
    batch_size=32,             # Tamaño del lote
    class_mode="categorical"   # Clasificación multicategoría
)

# Crear el modelo de red neuronal
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')  # Número de clases dinámico
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(
    train_generator,
    epochs=10,
    validation_data=valid_generator
)

# Guardar el modelo entrenado
model.save(f'{ruta_destino}/modelo_tatuaje.h5')

# Guardar class_indices
with open(f'{ruta_destino}/class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)
print("Class indices guardados en 'class_indices.json'")