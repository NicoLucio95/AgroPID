import cv2
import glob
import re
import os

from tkinter import filedialog

from core.procesamiento import calcular_ambientes, generar_mapas_k


# =========================
# 📂 CARGAR IMAGEN + PROCESAR
# =========================
def cargar_imagen_procesada():

    ruta = filedialog.askopenfilename(
        title="Seleccionar imagen",
        filetypes=[("Imagenes", "*.jpg *.png *.tif")]
    )

    if not ruta:
        return None

    img = cv2.imread(ruta)

    if img is None:
        print("Error al cargar la imagen")
        return None

    print("Imagen cargada:", ruta)

    return {
        "ruta": ruta,
        "img": img,
    }


# =========================
# 📁 OBTENER MAPAS ORDENADOS
# =========================
def obtener_mapas(carpeta="data/resultados"):

    if not os.path.exists(carpeta):
        return []

    archivos = glob.glob(os.path.join(carpeta, "mapa_k_*.png"))

    def extraer_k(ruta):
        match = re.search(r"mapa_k_(\d+)", ruta)
        return int(match.group(1)) if match else 0

    lista_ordenada = sorted(archivos, key=extraer_k)

    return lista_ordenada


# =========================
# 📖 CARGAR MAPA INDIVIDUAL
# =========================
def cargar_mapa(ruta):

    if not os.path.exists(ruta):
        return None

    img = cv2.imread(ruta)

    return img


# =========================
# 💾 GUARDAR IMAGEN
# =========================
def guardar_imagen(img):

    if img is None:
        return

    ruta = filedialog.asksaveasfilename(
        title="Guardar imagen",
        defaultextension=".png",
        filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")]
    )

    if not ruta:
        return

    cv2.imwrite(ruta, img)
    print("Imagen guardada en:", ruta)


# =========================
# 🔢 EXTRAER K DESDE NOMBRE
# =========================
def extraer_k_de_ruta(ruta):
    match = re.search(r"mapa_k_(\d+)", ruta)
    return int(match.group(1)) if match else None


# =========================
# 📁 CREAR CARPETA SI NO EXISTE
# =========================
def asegurar_carpeta(carpeta):
    os.makedirs(carpeta, exist_ok=True)