import cv2
import numpy as np
import tkinter as tk
import logging

from PIL import Image, ImageTk

from utils.io_utils import cargar_imagen_procesada
from utils.image_utils import crear_overlay

from core.analisis import (
    calcular_cobertura,
    overlay_cobertura,
    detectar_estres,
    mapa_estres_color
)

from core.procesamiento import mapa_calor_exg

logging.basicConfig(level=logging.INFO)


class CultivoApp:

    def __init__(self, root):

        self.root = root
        self.root.title("AgroPID - Análisis de Cultivo")
        self.root.geometry("1000x600")

        # =========================
        # VARIABLES
        # =========================
        self.img = None
        self.mapa = None
        self.overlay = None
        self.alpha = 0.4

        self.img_original = None
        self.img_tk = None

        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # =========================
        # CANVAS
        # =========================
        self.canvas = tk.Canvas(root, bg="#111")
        self.canvas.pack(side="right", expand=True, fill="both")

        self.canvas.bind("<MouseWheel>", self.zoom_mouse)
        self.canvas.bind("<ButtonPress-1>", self.iniciar_pan)
        self.canvas.bind("<B1-Motion>", self.mover_pan)

        # =========================
        # PANEL
        # =========================
        panel = tk.Frame(root, width=260, bg="#2c3e50")
        panel.pack(side="left", fill="y")
        panel.pack_propagate(False)

        tk.Label(panel, text="Cultivo", fg="white", bg="#2c3e50",
                 font=("Segoe UI", 16, "bold")).pack(pady=20)

        # =========================
        # BOTONES
        # =========================
        def btn(texto, cmd):
            b = tk.Label(panel, text=texto, bg="#34495e",
                         fg="white", pady=8, cursor="hand2")
            b.pack(fill="x", padx=15, pady=5)
            b.bind("<Button-1>", lambda e: cmd())
            return b

        btn("📂 Cargar Imagen", self.cargar_imagen)
        btn("🔥 Mapa de vigor (ExG)", self.mostrar_exg)
        btn("🌿 Cobertura vegetal", self.mostrar_cobertura)
        btn("⚠️ Estrés del cultivo", self.mostrar_estres)
        btn("💾 Guardar", self.guardar)

        # =========================
        # SLIDER
        # =========================
        tk.Label(panel, text="Transparencia", bg="#2c3e50", fg="white").pack()

        self.slider = tk.Scale(panel, from_=0, to=1, resolution=0.05,
                               orient="horizontal",
                               command=self.cambiar_alpha)

        self.slider.set(self.alpha)
        self.slider.pack(fill="x", padx=15)

        # =========================
        # INFO
        # =========================
        self.label_info = tk.Label(panel, text="-",
                                   bg="#2c3e50", fg="white")
        self.label_info.pack(pady=10)

    # =========================
    # FUNCIONES
    # =========================

    def cargar_imagen(self):
        data = cargar_imagen_procesada()
        if data is None:
            return

        self.img = data["img"]
        self.mostrar(self.img)

    def mostrar(self, img):
        self.img_original = img
        self.render()

    def render(self):
        if self.img_original is None:
            return

        img = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w*self.zoom), int(h*self.zoom)))

        img = Image.fromarray(img)
        self.img_tk = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y,
                                 anchor="nw", image=self.img_tk)

    def zoom_mouse(self, e):
        self.zoom *= 1.1 if e.delta > 0 else 0.9
        self.zoom = max(0.2, min(self.zoom, 5))
        self.render()

    def iniciar_pan(self, e):
        self.start_x = e.x
        self.start_y = e.y

    def mover_pan(self, e):
        dx = e.x - self.start_x
        dy = e.y - self.start_y

        self.offset_x += dx
        self.offset_y += dy

        self.start_x = e.x
        self.start_y = e.y

        self.render()

    def cambiar_alpha(self, val):
        self.alpha = float(val)
        self.actualizar_overlay()

    def actualizar_overlay(self):
        if self.img is None or self.mapa is None:
            return

        self.overlay = crear_overlay(self.img, self.mapa, self.alpha)
        self.mostrar(self.overlay)

    # =========================
    # ANÁLISIS
    # =========================

    def mostrar_exg(self):
        if self.img is None:
            return

        self.mapa = mapa_calor_exg(self.img)
        self.actualizar_overlay()
        self.label_info.config(text="Mapa de vigor (ExG)")

    def mostrar_cobertura(self):
        if self.img is None:
            return

        cobertura, mask = calcular_cobertura(self.img)
        self.mapa = overlay_cobertura(self.img, mask)

        self.mostrar(self.mapa)
        self.label_info.config(text=f"Cobertura: {cobertura:.2f}%")

    def mostrar_estres(self):
        if self.img is None:
            return

        estres, _ = detectar_estres(self.img)
        self.mapa = mapa_estres_color(estres)

        self.actualizar_overlay()
        self.label_info.config(text="Mapa de estrés")

    def guardar(self):
        if self.overlay is None:
            return

        from tkinter import filedialog

        ruta = filedialog.asksaveasfilename(defaultextension=".png")
        if ruta:
            cv2.imwrite(ruta, self.overlay)