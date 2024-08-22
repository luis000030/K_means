import numpy as np
import matplotlib.pyplot as plt
from algoritmo.k_means import*
from PIL import Image

imagen = Image.open('D:\\Programacion\\Python\\ML\\K-means\\data\\Atardecer.jpeg')

array_imagen = np.array(imagen)
WIDTH = array_imagen.shape[0]
HEIGHT = array_imagen.shape[1]
imagen_reordenada = array_imagen.reshape(-1, 3)

num_centroides = 64

centroides = correr_algoritmo_varias_veces(imagen_reordenada, k=num_centroides, num_iters=16, umbral=0.001, num_veces=16, show=True)
centroides_cercanos = calcular_centroides_cercanos(imagen_reordenada, centroides)
for j in range(num_centroides):
    mask_j = (centroides_cercanos == j)
    imagen_reordenada[mask_j] = centroides[j]

array_imagen_comprimida = imagen_reordenada.reshape(WIDTH, HEIGHT, -1)
array_imagen_comprimida = array_imagen_comprimida.astype(np.uint8)
imagen_comprimida = Image.fromarray(array_imagen_comprimida)
imagen_comprimida.save('D:\\Programacion\\Python\\ML\\K-means\\data\\Atardecer_comprimido_524.jpeg')