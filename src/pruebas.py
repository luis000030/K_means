import numpy as np
import matplotlib.pyplot as plt
from algoritmo.k_means import*

A = np.array([[1, 2], [-1, 4], [3, 0], [-2, -1]])
B = np.array([[1, 2], [1, 5], [0, 2]])
# Transponemos la matriz A para que sus columnas se conviertan en filas

c = coste(A, B)
print(c)

"""def centro_aleatorio():
    return np.random.uniform(0, 10, 2)

def puntos_aleatorios_centro(num_puntos: int, radio: float, c):

    puntos = np.tile(c, (num_puntos, 1))
    radios = np.random.uniform(0, radio, num_puntos)
    angulos = np.random.uniform(0, 360, num_puntos)
    angulos = np.deg2rad(angulos)
    puntos[:, 0] = puntos[:, 0] + radios * np.cos(angulos)
    puntos[:, 1] = puntos[:, 1] + radios * np.sin(angulos)

    return puntos

c1 = centro_aleatorio()
c2 = centro_aleatorio()

a1 = puntos_aleatorios_centro(20, 2.25, c1)
a2 = puntos_aleatorios_centro(20, 2.25, c2)

plt.scatter(a1[:, 0], a1[:, 1])
plt.scatter(a2[:, 0], a2[:, 1])

plt.xlim(-1, 11)
plt.ylim(-1, 11)

plt.axhline(y=0, color='black', linewidth=1)  # Eje X
plt.axvline(x=0, color='black', linewidth=1)  # Eje Y

puntos = np.concatenate((a1, a2), axis=0)
centroides = correr_algoritmo_varias_veces(puntos, k=2, num_iters=10, umbral=0.01, num_veces=100, show=True)

plt.scatter(centroides[:, 0], centroides[:, 1])

plt.show()
"""


# Distancia por si lo otro no funciona
"""
def d(p1, B):
    
    Calcula el cuadrado de las distancias de un punto a un conjunto de puntos.
    Args:
        p1: (ndarray Shape (1, n)) El punto al que se calculan las distancias
        B: (ndarray Shape (m, n)) La matriz con m puntos.
    Returns:
        productos (ndarray Shape (m,)) El array con los cuadrados de las distancias.
    
    productos = np.sum((B-p1) * (B-p1), axis=1)
    return productos
"""