import numpy as np
import time

def d(A, B):
    """
    Calcula el cuadrado de las disntacias de varios puntos a otro conjunto de puntos.
    Args:
        A: (ndarray Shape (n, m)) El primer conjunto de puntos. Son n puntos de dimensión m.
        B: (ndarray Shape (k, m)) El segundo conjunto de puntos. Son k puntos de dimensión m.
    Returns:
        distancias (ndarray Shape (n, k)) La matriz con las distancias. La entrada (i, j) corresponde con
            d(A[i], B[j])^2
    """
    n = A.shape[0]
    k = B.shape[0]
    m = A.shape[1]
    new_A = np.tile(A, (1, k))
    new_A = new_A.reshape(n, k, m)
    new_B = np.tile(B, (n, 1))
    new_B = new_B.reshape(n, k, m)
    C = new_A - new_B
    productos = np.sum(C * C, axis=-1)
    return productos

def inicializar_algoritmo(puntos, k):
    """
    Método que inicializa el algoritmo devolviendo los primeros centroides desde los que se itera.
    Args:
        puntos: (ndarray Shape (n, m)) Matriz con n puntos de dimensión m.
        k: (integer) Número de centroides que se quieren.
    """
    n = puntos.shape[0]
    if k > n:
        raise ValueError("k no puede ser mayor que el número de puntos")
    indices = np.random.choice(n, size=k, replace=False)
    centroides = puntos[indices]

    return centroides

def entrenar_algoritmo_global(puntos, centroides, num_iters, umbral):
    """
    Método que ejecuta la parte del algoritmo de k-means sin la inicialización.
    Args:
        puntos: (ndarray Shape (n, m)) Matriz con n puntos de dimensión m. El algoritmo encuentra los clusters en estos puntos.
        centroides: (ndarray Shape (k, m)) Matriz con k centroides con los que comienza el algoritmo.
        num_iters: (integer) Máximas iteraciones que se hacen en el algoritmo.
        umbral: (scalar) La diferencia del coste a partir del cual el algoritmo para. Es decir, si de una iteración a otra el coste
            mejora menos que el umbral, el algoritmo para.
    Returns:
        centroides: (ndarray Shape (k, m)) Matriz con los k centroides encontrados por el algoritmo.
        cost: (scalar) El coste que encuentra el algoritmo. Este se calcula sumando las distancias de cada punto a su centroide más cercano
            y promediando entre el total de puntos.
    """
    cost = coste(puntos, centroides)
    for i in range(num_iters):
        centroides = entrenar_algoritmo_paso(puntos, centroides)
        coste_nuevo = coste(puntos, centroides)
        aux = cost
        cost = coste_nuevo
        if(aux - coste_nuevo < umbral):
            break
    
    return centroides, cost

def entrenar_algoritmo_paso(puntos, centroides):
    """
    Método que actualiza los centroides una vez. Esto se hace calculando los centroides más cercanos a cada punto y sustituyendo
    estos centroides por unos nuevos que se calculan hayando el centro de los puntos más cercanos a cada centroide.
    Args:
        puntos: (ndarray Shape (n, m)) Matriz con n puntos de dimensión m.
        centroides: (ndarray Shape (k, m)) Matriz con los k centroides de dimensión m.
    Returns:
        centroides_nuevos: (ndarray Shape (k, m)) Matriz con los k nuevos centroides de dimensión m.
    """
    centroides_cerca = calcular_centroides_cercanos(puntos, centroides)
    k = centroides.shape[0]
    for j in range(k):
        mask_j = (centroides_cerca == j)
        puntos_cerca_j = puntos[mask_j]
        if puntos_cerca_j.size !=0:
            centroides[j] = calcular_centro(puntos_cerca_j)
    return centroides

def calcular_centro(puntos):
    """
    Método que calcula el centro de un conjunto de puntos.
    Args:
        puntos: (ndarray Shape (n, m)) Matriz con n puntos de dimensión m.
    Returns:
        centro: (ndarray Shape (m,)) Punto del centro. Se calcula como (Sum_{i=1}^{n} p_i) / n
    """
    n = puntos.shape[0]
    if n == 0:
        return None
    centro = np.sum(puntos, axis=0) / n
    return centro

def calcular_centroides_cercanos(puntos, centroides):
    """
    Método que dados unos puntos y unos centroides, determina para cada punto qué centroide está más cerca.
    Args:
        puntos: (ndarray Shape (n, m)) Matriz con n puntos de dimensión m.
        centroides: (ndarray Shape (k, m)) Matriz con los k centroides de dimensión m.
    Returns:
        centroides_cerca (ndarray Shape (n,)) Array con entradas de enteros que corresponden al índice del centroide más
            cercano.
    """
    distancias = d(puntos, centroides)
    centroides_cerca = np.argmin(distancias, axis=1)
    
    return centroides_cerca

def coste(puntos, centroides):
    """
    Método que calcula el coste en una etapa del algoritmo. Este se calcula sumando las distancias de cada punto a su centroide más cercano
    y promediando entre el total de puntos.
    Args:
        puntos: (ndarray Shape (n, m)) Matriz con n puntos de dimensión m.
        centroides: (ndarray Shape (k, m)) Matriz con los k centroides con los que se quiere calcular el coste.
    Returns:
        coste : (scalar) El coste.
    """
    dist = d(puntos, centroides)
    coste = np.min(dist, axis = 1).sum() / puntos.shape[0]
    return coste

def correr_algoritmo(puntos, k, num_iters, umbral):
    """
    Método que ejecuta el algoritmo de k-means entero.
    Args:
        puntos: (ndarray Shape (n, m)) Matriz con n puntos de dimensión m. El algoritmo encuentra los clusters en estos puntos.
        k: (integer) Número de clusters que se quieren encontrar.
        num_iters: (integer) Máximas iteraciones que se hacen en el algoritmo.
        umbral: (scalar) La diferencia del coste a partir del cual el algoritmo para. Es decir, si de una iteración a otra el coste
            mejora menos que el umbral, el algoritmo para.
    Returns:
        centroides: (ndarray Shape (k, m)) Matriz con los k centroides encontrados por el algoritmo.
        coste: (scalar) El coste que encuentra el algoritmo. Este se calcula sumando las distancias de cada punto a su centroide más cercano
            y promediando entre el total de puntos.
    """
    centroides = inicializar_algoritmo(puntos, k)
    centroides, coste = entrenar_algoritmo_global(puntos, centroides, num_iters, umbral)
    return centroides, coste

def correr_algoritmo_varias_veces(puntos, k, num_iters, umbral, num_veces, show):
    """
    Método que ejecuta el algoritmo de k-means entero varias veces y se queda con el mejor.
    Args:
        puntos: (ndarray Shape (n, m)) Matriz con n puntos de dimensión m. El algoritmo encuentra los clusters en estos puntos.
        k: (integer) Número de clusters que se quieren encontrar.
        num_iters: (integer) Máximas iteraciones que se hacen en el algoritmo.
        umbral: (scalar) La diferencia del coste a partir del cual el algoritmo para. Es decir, si de una iteración a otra el coste
            mejora menos que el umbral, el algoritmo para.
        num_veces: (integer) El número de veces que se ejecuta el algoritmo.
        show: (boolean) Si quieres que se imprima por pantalla la mejoría del coste.
    Returns:
        centroides: (ndarray Shape (k, m)) Matriz con los k centroides encontrados por el algoritmo.
    """
    centroides, coste = correr_algoritmo(puntos, k, num_iters, umbral)
    for i in range(num_veces - 1):
        prin = time.time()
        nuevos_centroides, nuevo_coste = correr_algoritmo(puntos, k, num_iters, umbral)
        fin = time.time()
        if show == True:
            print(f"Tiempo: {fin - prin}")
        if nuevo_coste < coste:
            centroides = nuevos_centroides
            if show == True:
                print(f"Coste actual: {coste}. Nuevo coste: {nuevo_coste}")
            coste = nuevo_coste

    return centroides