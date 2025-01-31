import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

ruta_destino = 'C:/Users/juan.ochoa/clasificaTatuaje/Clasificar tattoo'
valid_dir = "C:\\Users\\juan.ochoa\\clasificaTatuaje\\Clasificar tattoo\\valid_org"

# Cargar el modelo guardado
model = load_model(f'{ruta_destino}/modelo_tatuaje.h5')
model.summary()
print("Modelo cargado correctamente.")

# for layer in model.layers:
#     print(layer.name)

# Asegúrate de que el tamaño coincide con el de entrenamiento
dummy_input = np.random.rand(1, 128, 128, 3)  # Ajusta el tamaño de entrada
_ = model.predict(dummy_input)
print('aaaa')

# def grad_cam(model, img_array, layer_name='conv2d'):
#     # Crear un modelo intermedio para obtener las activaciones de la capa deseada
#     grad_model = tf.keras.models.Model(
#         [model.inputs], 
#         [model.get_layer(layer_name).output, model.output]
#     )

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         loss = predictions[:, np.argmax(predictions)]  # Selecciona la clase con mayor probabilidad

#     # Calcula los gradientes de la clase respecto a la salida de la capa convolucional
#     grads = tape.gradient(loss, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Promedio de los gradientes

#     # Multiplica cada canal por la importancia de su gradiente
#     conv_outputs = conv_outputs.numpy()[0]
#     heatmap = np.mean(conv_outputs * pooled_grads, axis=-1)

#     # Normaliza la imagen
#     heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

#     return heatmap

# # 📌 Cargar imagen de prueba
# img_path = f'{ruta_destino}/imagenTradiPrueba.JPG'
# img = image.load_img(img_path, target_size=(128, 128))  # Usa el tamaño que entrenaste
# img_array = np.expand_dims(image.img_to_array(img), axis=0) / 255.0  # Normalizar

# # 📌 Obtener el mapa de calor con Grad-CAM
# _ = model.predict(img_array)  # Ejecuta una predicción para inicializar el modelo
# heatmap = grad_cam(model, img_array, layer_name='conv2d')  # Cambia 'conv2d' por el nombre real de tu capa

# # 📌 Superponer el mapa de calor en la imagen original
# heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))  # Redimensionar al tamaño de la imagen original
# heatmap = np.uint8(255 * heatmap)  # Convertir a valores entre 0 y 255
# heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Aplicar colormap

# # Mezclar con la imagen original
# superimposed_img = cv2.addWeighted(np.array(img), 0.6, heatmap, 0.4, 0)

# # 📌 Mostrar la imagen con el mapa de calor
# plt.figure(figsize=(8, 8))
# plt.imshow(superimposed_img)
# plt.axis("off")
# plt.title("Mapa de Calor Grad-CAM")
# plt.show()