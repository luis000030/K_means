import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from skimage import io
import time

com = time.time()

# Paso 1: Cargar la imagen
image = io.imread('data\\Atardecer_colorido.jpeg')
image = np.array(image, dtype=np.float64) / 255  # Normaliza los valores de píxeles entre 0 y 1

# Paso 2: Transformar la imagen en una matriz de datos
w, h, d = image.shape
image_array = np.reshape(image, (w * h, d))

# Paso 3: Aplicar K-means para encontrar los clústeres de colores
n_colors = 512  # Número de colores deseado en la imagen comprimida
image_array_sample = shuffle(image_array, random_state=0)[:1000]  # Toma una muestra de los datos
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
labels = kmeans.predict(image_array)

# Paso 4: Recrear la imagen comprimida
def recreate_image(codebook, labels, w, h):
    """Recrear la imagen desde el código de libro y las etiquetas"""
    return codebook[labels].reshape((w, h, -1))


compressed_image = recreate_image(kmeans.cluster_centers_, labels, w, h)

fin = time.time()
print(f"Tiempo: {fin - com}")

# Paso 5: Visualizar la imagen original y la comprimida
plt.figure(1)
plt.clf()
plt.axis('off')
plt.title('Imagen Original')
plt.imshow(image)

plt.figure(2)
plt.clf()
plt.axis('off')
plt.title('Imagen Comprimida con K-means')
plt.imshow(compressed_image)
plt.show()
