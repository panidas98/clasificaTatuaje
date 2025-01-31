import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ruta_carpeta_imagenes = 'C:/Users/juan.ochoa/clasificaTatuaje/Clasificar tattoo/Tattoo_Types-4'
ruta_destino = 'C:/Users/juan.ochoa/clasificaTatuaje/Clasificar tattoo'
valid_dir = "C:\\Users\\juan.ochoa\\clasificaTatuaje\\Clasificar tattoo\\valid_org"

# Cargar el modelo guardado
model = load_model(f'{ruta_destino}/modelo_tatuaje.h5')
model.summary()
print("Modelo cargado correctamente.")

valid_datagen = ImageDataGenerator(rescale=1.0/255)  # Normalización de imágenes

valid_generator = valid_datagen.flow_from_directory(
    directory=valid_dir,
    target_size=(128, 128),    # Asegúrate de que coincide con el tamaño usado en entrenamiento
    batch_size=32,             # Tamaño del lote
    class_mode="categorical",  # Mismo tipo de clasificación que en el entrenamiento
    shuffle=False              # IMPORTANTE: No mezclar para que los índices coincidan con y_true
)

print("✅ Generador de validación creado correctamente.")

# Obtener predicciones del modelo en el conjunto de validación
y_pred = model.predict(valid_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = valid_generator.classes  # Etiquetas reales

# Obtener la matriz de confusión
cm = confusion_matrix(y_true, y_pred_classes)

# Graficar la matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=valid_generator.class_indices.keys(), yticklabels=valid_generator.class_indices.keys())
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()