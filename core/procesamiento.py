import os
import cv2
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from core.analisis import calcular_porcentajes



def encontrar_mejor_k(img, k_min=2, k_max=10):

    # =========================
    # FEATURES (EXG + RGB)
    # =========================
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    R = img_rgb[:, :, 0].astype(float)
    G = img_rgb[:, :, 1].astype(float)
    B = img_rgb[:, :, 2].astype(float)

    exg = 2*G - R - B
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX)

    features = np.dstack((img, exg_norm))
    features = features.reshape(-1, 4)

    # =========================
    # MUESTREO DINÁMICO (OPTIMIZADO)
    # =========================
    total_pixeles = features.shape[0]

    sample_size = int(total_pixeles * 0.03)  # 3%

    sample_size = max(2000, sample_size)     # mínimo
    sample_size = min(8000, sample_size)     # máximo

    sample_size = min(sample_size, total_pixeles)  # evitar error

    idx = np.random.choice(total_pixeles, size=sample_size, replace=False)
    sample = features[idx]

    # =========================
    # BÚSQUEDA DE k ÓPTIMO
    # =========================
    mejores_k = k_min
    mejor_score = -1

    scores = []

    for k in range(k_min, k_max + 1):

        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = kmeans.fit_predict(sample)

        score = silhouette_score(sample, labels)

        scores.append(score)

        print(f"k={k} → score={score:.4f}")

        if score > mejor_score:
            mejor_score = score
            mejores_k = k

    print(f"\nMejor k: {mejores_k}")

    return mejores_k, scores



# =========================
# 🌱 CALCULAR AMBIENTES (KMeans)
# =========================
def calcular_ambientes(img, k=3):
    """
    Segmenta la imagen en k ambientes usando KMeans.
    """

    #Suavizado (reduce ruido)
    k_blur = max(3, int(k // 2) * 2 + 1)  # asegurar impar
    img_gauss = cv2.GaussianBlur(img, (k_blur, k_blur), 0)

    # Reorganizar imagen (pixeles x 3)
    img_reshape = img_gauss.reshape(-1, 3)

    # KMeans
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = kmeans.fit_predict(img_reshape)

    # Reconstruir mapa
    mapa = labels.reshape(img.shape[:2])

    # =========================
    # Ordenar clusters por brillo
    # =========================
    centroides = kmeans.cluster_centers_
    orden = np.argsort(np.sum(centroides, axis=1))

    mapa_ordenado = np.zeros_like(mapa)

    for i, idx in enumerate(orden):
        print(i)
        mapa_ordenado[mapa == idx] = i

    mapa = mapa_ordenado

    # =========================
    # Porcentajes
    # =========================
    porcentajes_dict = calcular_porcentajes(mapa)

    # Convertir a lista ordenada
    porcentajes = [porcentajes_dict.get(i, 0) for i in range(k)]

    # =========================
    # Colormap dinámico
    # =========================
    colores = generar_colores(k)
    cmap = ListedColormap(colores)

    return mapa.astype(np.uint8), porcentajes, cmap

def calcular_ambientes_exg(img, k=3):

    # =========================
    # 1. EXG
    # =========================
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    R = img_rgb[:, :, 0].astype(float)
    G = img_rgb[:, :, 1].astype(float)
    B = img_rgb[:, :, 2].astype(float)

    exg = 2 * G - R - B

    # Normalizar EXG
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX)

    # =========================
    # 2. SUAVIZADO
    # =========================
    img_gauss = cv2.GaussianBlur(img, (5, 5), 0)

    # =========================
    # 3. FEATURE STACK
    # =========================
    # Unimos RGB + EXG
    features = np.dstack((img_gauss, exg_norm))

    # reshape → (pixeles, 4)
    features = features.reshape(-1, 4)

    # =========================
    # 4. KMEANS
    # =========================
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = kmeans.fit_predict(features)

    mapa = labels.reshape(img.shape[:2])

    # =========================
    # 5. ORDENAR POR VEGETACIÓN (clave)
    # =========================
    centroides = kmeans.cluster_centers_

    # usar canal EXG (última columna)
    orden = np.argsort(centroides[:, 3])  # 👈 importante

    mapa_ordenado = np.zeros_like(mapa)

    for i, idx in enumerate(orden):
        mapa_ordenado[mapa == idx] = i

    mapa = mapa_ordenado

    # =========================
    # 6. PORCENTAJES
    # =========================
    unique, counts = np.unique(mapa, return_counts=True)
    porcentajes = counts / counts.sum() * 100

    # =========================
    # 7. COLORMAP
    # =========================
    colores = generar_colores(k)
    cmap = ListedColormap(colores)

    return mapa.astype(np.uint8), porcentajes, cmap

def calcular_ambientes_auto(img):

    k_optimo, scores = encontrar_mejor_k(img, 2, 10)

    print("Usando k =", k_optimo)

    mapa, porcentajes, cmap = calcular_ambientes_exg(img, k_optimo)

    return mapa, porcentajes, cmap, k_optimo, scores

# =========================
# 🎨 GENERAR COLORES DINÁMICOS
# =========================
def generar_colores(k):
    
    cmap = LinearSegmentedColormap.from_list(
        "verde_amarillo_rojo",
        ["green", "yellow", "red"]
    )

    return [cmap(i / (k - 1))[:3] for i in range(k)]

# =========================
# 💾 GENERAR MAPAS PARA VARIOS K
# =========================
def generar_mapas_k(img, carpeta_salida="data/resultados"):
    """
    Genera mapas para múltiples valores de k y los guarda.
    """

    os.makedirs(carpeta_salida, exist_ok=True)

    for k in range(3, 100, 6):  # 3, 9, 15 ... 99

        print(f"Procesando k = {k}")

        mapa, _, cmap = calcular_ambientes(img, k)

        # Convertir a imagen
        mapa_color = cmap(mapa)[:, :, :3]
        mapa_color = (mapa_color * 255).astype(np.uint8)
        mapa_color = cv2.cvtColor(mapa_color, cv2.COLOR_RGB2BGR)

        # Guardar
        nombre = f"mapa_k_{k}.png"
        ruta = os.path.join(carpeta_salida, nombre)

        cv2.imwrite(ruta, mapa_color)

    print("Proceso terminado ✔")

# =========================
# 🌱 ÍNDICE EXG
# =========================
def calcular_exg(img):
    """
    Calcula el índice de vegetación EXG.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    R = img_rgb[:, :, 0].astype(float)
    G = img_rgb[:, :, 1].astype(float)
    B = img_rgb[:, :, 2].astype(float)

    exg = (2 * G - R - B) - (1.4 * R - G)

    return exg


# =========================
# 🔥 MAPA DE CALOR
# =========================
def mapa_calor_exg(img):
    """
    Genera un mapa de calor basado en EXG.
    """
    exg = calcular_exg(img)

    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX)
    exg_norm = exg_norm.astype(np.uint8)

    heatmap = cv2.applyColorMap(exg_norm, cv2.COLORMAP_JET)

    return heatmap


# =========================
# ✂️ SEGMENTACIÓN BINARIA (OPCIONAL)
# =========================
def segmentar_otsu(img):
    """
    Segmentación automática con Otsu.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return mask

def dibujar_contornos(img, mapa):

    contornos_img = img.copy()

    # recorrer cada ambiente
    for i in np.unique(mapa):

        # máscara binaria del ambiente
        mask = (mapa == i).astype(np.uint8) * 255

        # encontrar contornos
        contornos, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # dibujar contornos
        cv2.drawContours(contornos_img, contornos, -1, (0, 0, 0), 2)

    return contornos_img