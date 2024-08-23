import numpy as np
import matplotlib.pyplot as plt
from algoritmo.k_means import*
from PIL import Image
from sklearn.utils import shuffle

com = time.time()

imagen = Image.open('data\\Atardecer.jpeg')

array_imagen = np.array(imagen)
array_imagen = np.array(array_imagen, dtype=np.float64) / 255
WIDTH, HEIGHT, d = array_imagen.shape
array_imagen = array_imagen.reshape(-1, d)

num_centroides = 64
num_muestras_entrenar = 1024

muestras = shuffle(array_imagen, random_state=0)[:num_muestras_entrenar]

#Hago el Algoritmo
centroides = correr_algoritmo_varias_veces(muestras, k=num_centroides, num_iters=16, umbral=0.000001, num_veces=16, show=False)
centorides_cercanos = calcular_centroides_cercanos(array_imagen, centroides)


#Recreo la imagen:
"""Recrear la imagen desde el código de libro y las etiquetas"""
array_imagen_comprimida = centroides[centorides_cercanos].reshape((WIDTH, HEIGHT, -1))
array_imagen_comprimida = array_imagen_comprimida * 255
array_imagen_comprimida = array_imagen_comprimida.astype(np.uint8)
imagen_comprimida = Image.fromarray(array_imagen_comprimida)
imagen_comprimida.save('data/IPP2.jpeg')

fin = time.time()
print(f"Tiempo: {fin - com}")

# Implementación muy lenta y con peores resultados
"""centroides = correr_algoritmo_varias_veces(imagen_reordenada, k=num_centroides, num_iters=16, umbral=0.001, num_veces=16, show=True)
centroides_cercanos = calcular_centroides_cercanos(imagen_reordenada, centroides)
for j in range(num_centroides):
    mask_j = (centroides_cercanos == j)
    imagen_reordenada[mask_j] = centroides[j]

array_imagen_comprimida = imagen_reordenada.reshape(WIDTH, HEIGHT, -1)
array_imagen_comprimida = array_imagen_comprimida.astype(np.uint8)
imagen_comprimida = Image.fromarray(array_imagen_comprimida)
imagen_comprimida.save('Atardecer_comprimido_524.jpeg')

"""