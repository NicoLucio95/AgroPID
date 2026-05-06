import numpy as np
import cv2


# =========================
# 📊 PORCENTAJES POR CLASE
# =========================
def calcular_porcentajes(mapa):
    """
    Calcula el porcentaje de cada clase en un mapa segmentado.
    """
    unique, counts = np.unique(mapa, return_counts=True)
    porcentajes = counts / counts.sum() * 100
    return dict(zip(unique, porcentajes))


# =========================
# 🌱 INDICE EXG (VEGETACIÓN)
# =========================
def calcular_exg(img):
    """
    Calcula el índice de vegetación EXG.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    R = img_rgb[:, :, 0].astype(float)
    G = img_rgb[:, :, 1].astype(float)
    B = img_rgb[:, :, 2].astype(float)

    exg = 2 * G - R - B

    return exg


# =========================
# 🎨 NORMALIZAR IMAGEN
# =========================
def normalizar_imagen(img):
    """
    Normaliza una imagen a rango 0–255.
    """
    norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


# =========================
# 🔥 MAPA DE CALOR
# =========================
def crear_mapa_calor(img):
    """
    Genera un heatmap a partir del EXG.
    """
    exg = calcular_exg(img)
    exg_norm = normalizar_imagen(exg)

    heatmap = cv2.applyColorMap(exg_norm, cv2.COLORMAP_JET)

    return heatmap


# =========================
# 📈 VARIANZA POR CLASE
# =========================
def varianza_por_clase(img, mapa):
    """
    Calcula la variabilidad (varianza) dentro de cada clase.
    """
    resultados = {}

    for clase in np.unique(mapa):
        pixels = img[mapa == clase]

        if len(pixels) > 0:
            var = np.var(pixels)
            resultados[int(clase)] = var

    return resultados


# =========================
# 📊 PROMEDIO POR CLASE
# =========================
def promedio_por_clase(img, mapa):
    """
    Calcula el valor medio RGB por clase.
    """
    resultados = {}

    for clase in np.unique(mapa):
        pixels = img[mapa == clase]

        if len(pixels) > 0:
            mean = np.mean(pixels, axis=0)
            resultados[int(clase)] = mean.tolist()

    return resultados


# =========================
# 🧠 SCORE SIMPLE DE CLUSTERING
# =========================
def score_homogeneidad(img, mapa):
    """
    Evalúa qué tan homogéneas son las clases (menor varianza = mejor).
    """
    total_var = 0

    for clase in np.unique(mapa):
        pixels = img[mapa == clase]

        if len(pixels) > 0:
            total_var += np.var(pixels)

    return total_var


# =========================
# 🎯 SELECCIÓN DEL MEJOR K
# =========================
def elegir_mejor_k(resultados_k):
    """
    Recibe un diccionario:
    {k: score}
    y devuelve el mejor k (menor score)
    """
    return min(resultados_k, key=resultados_k.get)

def calcular_cobertura(img):

    # =========================
    # 1. EXG
    # =========================
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    R = img_rgb[:, :, 0].astype(float)
    G = img_rgb[:, :, 1].astype(float)
    B = img_rgb[:, :, 2].astype(float)

    exg = 2*G - R - B

    # =========================
    # 2. NORMALIZAR
    # =========================
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX)
    exg_norm = exg_norm.astype("uint8")

    # =========================
    # 3. UMBRAL AUTOMÁTICO
    # =========================
    _, mask = cv2.threshold(
        exg_norm, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # =========================
    # 4. COBERTURA
    # =========================
    vegetacion = np.sum(mask == 255)
    total = mask.size

    cobertura = (vegetacion / total) * 100

    return cobertura, mask

def overlay_cobertura(img, mask):

    color_mask = np.zeros_like(img)
    color_mask[mask == 255] = [0, 255, 0]  # verde

    overlay = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)

    return overlay

def detectar_estres(img):

    # =========================
    # EXG
    # =========================
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    R = img_rgb[:, :, 0].astype(float)
    G = img_rgb[:, :, 1].astype(float)
    B = img_rgb[:, :, 2].astype(float)

    exg = 2*G - R - B

    # =========================
    # NORMALIZAR
    # =========================
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX)
    exg_norm = exg_norm.astype("uint8")

    # =========================
    # CLASIFICACIÓN
    # =========================
    estres = np.zeros_like(exg_norm)

    # umbrales (ajustables)
    estres[exg_norm > 170] = 2   # sano
    estres[(exg_norm > 100) & (exg_norm <= 170)] = 1  # medio
    estres[exg_norm <= 100] = 0  # estrés

    return estres, exg_norm

def mapa_estres_color(estres):

    mapa = np.zeros((estres.shape[0], estres.shape[1], 3), dtype=np.uint8)

    # rojo → estrés
    mapa[estres == 0] = [0, 0, 255]

    # amarillo → medio
    mapa[estres == 1] = [0, 255, 255]

    # verde → sano
    mapa[estres == 2] = [0, 255, 0]

    return mapa