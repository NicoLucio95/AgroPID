import cv2
import numpy as np


# =========================
# 🎨 CREAR OVERLAY
# =========================
def crear_overlay(img, mapa, alpha=0.6):
    """
    Mezcla imagen original con mapa usando transparencia.
    """

    if img is None or mapa is None:
        return None

    # Asegurar mismo tamaño
    mapa = cv2.resize(mapa, (img.shape[1], img.shape[0]))

    beta = 1 - alpha

    overlay = cv2.addWeighted(img, alpha, mapa, beta, 0)

    return overlay


# =========================
# 🔄 CONVERTIR BGR → RGB
# =========================
def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# =========================
# 🔄 CONVERTIR RGB → BGR
# =========================
def rgb_to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# =========================
# 📏 REDIMENSIONAR IMAGEN
# =========================
def resize_proporcional(img, max_width=700, max_height=500):
    """
    Redimensiona manteniendo proporción.
    """

    h, w = img.shape[:2]

    scale = min(max_width / w, max_height / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(img, (new_w, new_h))


# =========================
# 🎨 APLICAR COLORMAP
# =========================
def aplicar_colormap(mapa, cmap):
    """
    Convierte un mapa de clases en imagen color.
    """

    mapa_color = cmap(mapa)[:, :, :3]
    mapa_color = (mapa_color * 255).astype(np.uint8)

    return rgb_to_bgr(mapa_color)


# =========================
# 🔥 NORMALIZAR IMAGEN
# =========================
def normalizar(img):
    """
    Normaliza imagen a rango 0–255.
    """

    norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    return norm.astype(np.uint8)


# =========================
# 🌈 HEATMAP GENERICO
# =========================
def crear_heatmap(img_gray):
    """
    Aplica colormap JET a una imagen en escala de grises.
    """

    img_norm = normalizar(img_gray)

    heatmap = cv2.applyColorMap(img_norm, cv2.COLORMAP_JET)

    return heatmap


# =========================
# ✂️ RECORTAR IMAGEN
# =========================
def recortar(img, x_min, y_min, x_max, y_max):
    """
    Recorta una región de la imagen.
    """
    return img[y_min:y_max, x_min:x_max]


# =========================
# 📊 OBTENER PIXEL
# =========================
def obtener_pixel(img, x, y):
    """
    Devuelve el valor RGB de un pixel.
    """
    return img[y, x]


# =========================
# 🎯 MASCARA POR COLOR
# =========================
def aplicar_mascara(img, mask):
    """
    Aplica una máscara binaria a una imagen.
    """
    return cv2.bitwise_and(img, img, mask=mask)


# =========================
# 🧱 CONVERTIR A GRIS
# =========================
def a_gris(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# =========================
# 🧪 DEBUG (MOSTRAR INFO)
# =========================
def info_imagen(img, nombre="Imagen"):
    """
    Imprime información útil de la imagen.
    """
    print(f"{nombre}:")
    print(f"  shape: {img.shape}")
    print(f"  dtype: {img.dtype}")
    print(f"  min: {img.min()}  max: {img.max()}")