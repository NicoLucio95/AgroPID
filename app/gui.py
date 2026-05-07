import re
import cv2
import glob
import logging
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt

from PIL import Image, ImageTk

from utils.io_utils import cargar_imagen_procesada
from utils.image_utils import crear_overlay

from core.analisis import calcular_cobertura,overlay_cobertura, detectar_estres,mapa_estres_color
from core.procesamiento import calcular_exg,mapa_calor_exg,calcular_ambientes_exg,calcular_ambientes_auto, calcular_ambientes, dibujar_contornos

logging.basicConfig(level=logging.INFO)

class AgroApp:

    def __init__(self, root):

        self.root = root
        self.root.title("AgroPID - Análisis de Ambientes")
        self.root.geometry("1000x600")

        # =========================
        # COLORES
        # =========================
        BG_MAIN = "#1e1e1e"
        BG_PANEL = "#2c3e50"
        BTN_COLOR = "#34495e"
        BTN_HOVER = "#3d566e"
        TEXT_COLOR = "#ecf0f1"
        ACCENT = "#1abc9c"

        self.root.configure(bg=BG_MAIN)

        # =========================
        # VARIABLES
        # =========================
        self.img = None
        self.mapa_procesado = None
        self.overlay = None
        #self.scores_cache = None
        #self.k_optimo_cache = None
        self.auto_cache = None

        self.lista_mapas = []
        self.indice_actual = 0

        self.zoom = 1.0
        self.img_original = None
        self.img_tk = None

        self.offset_x = 0
        self.offset_y = 0

        self.mostrar_contornos = False
        self.capa_contornos = None

        self.alpha = 0  # ✅ antes del slider

        # =========================
        # CANVAS
        # =========================
        self.canvas = tk.Canvas(root, bg="#111111", highlightthickness=0)
        self.canvas.pack(side="right", expand=True, fill="both")

        self.canvas.bind("<MouseWheel>", self.zoom_mouse)
        self.canvas.bind("<ButtonPress-1>", self.iniciar_pan)
        self.canvas.bind("<B1-Motion>", self.mover_pan)

        self.canvas.focus_set()  # importante

        # =========================
        # PANEL IZQUIERDO
        # =========================
        panel = tk.Frame(root, width=260, bg=BG_PANEL)
        panel.pack(side="left", fill="y")
        panel.pack_propagate(False)

        # =========================
        # TÍTULO
        # =========================
        tk.Label(panel,
                text="AgroPID",
                bg=BG_PANEL,
                fg=ACCENT,
                font=("Segoe UI", 16, "bold")
        ).pack(pady=20)

        self.modo = tk.StringVar(value="auto")

        tk.Label(panel, text="Modo", bg=BG_PANEL, fg=TEXT_COLOR).pack()

        tk.Radiobutton(panel, text="Automático", variable=self.modo,
                    value="auto", bg=BG_PANEL, fg=TEXT_COLOR,
                    selectcolor=BG_PANEL).pack(anchor="w", padx=15)

        tk.Radiobutton(panel, text="Manual", variable=self.modo,
                    value="manual", bg=BG_PANEL, fg=TEXT_COLOR,
                    selectcolor=BG_PANEL).pack(anchor="w", padx=15)
        
        tk.Label(panel, text="Número de ambientes (k)",
         bg=BG_PANEL, fg=TEXT_COLOR).pack(pady=5)

        self.slider_k = tk.Scale(
            panel,
            from_=2, to=10,
            orient="horizontal",
            bg=BG_PANEL,
            fg=TEXT_COLOR,
            highlightthickness=0
        )
        self.slider_k.set(4)
        self.slider_k.pack(padx=15, fill="x")

        # =========================
        # BOTONES MODERNOS
        # =========================
        def crear_boton(texto, comando):

            btn = tk.Label(
                panel,
                text=texto,
                bg=BTN_COLOR,
                fg=TEXT_COLOR,
                font=("Segoe UI", 10),
                padx=10,
                pady=8,
                cursor="hand2"
            )

            btn.pack(fill="x", padx=15, pady=5)

            btn.bind("<Enter>", lambda e: btn.config(bg=BTN_HOVER))
            btn.bind("<Leave>", lambda e: btn.config(bg=BTN_COLOR))
            btn.bind("<Button-1>", lambda e: comando())

            return btn

        crear_boton("📂 Cargar Imagen", self.cargar_imagen)
        crear_boton("🔥 Mapa de calor", self.mostrar_exg)
        crear_boton("🌱 Ambientes", self.limites_ambientes)
        crear_boton("📊 Límites", self.limites)

        self.btn_contornos = crear_boton("Mostrar límites", self.toggle_contornos)
        crear_boton("💾 Guardar Resultado", self.guardar_imagen)

        # =========================
        # SLIDER
        # =========================
        tk.Label(panel, text="Transparencia",
                bg=BG_PANEL, fg=TEXT_COLOR).pack(pady=10)

        self.slider = tk.Scale(
            panel,
            from_=0, to=1,
            resolution=0.05,
            orient="horizontal",
            bg=BG_PANEL,
            fg=TEXT_COLOR,
            highlightthickness=0,
            troughcolor="#555",
            command=self.actualizar_transparencia
        )

        self.slider.set(self.alpha)
        self.slider.pack(padx=15, fill="x")

        # =========================
        # INFO
        # =========================
        self.label_info = tk.Label(
            panel,
            text="Ambientes: -",
            bg=BG_PANEL,
            fg=TEXT_COLOR,
            justify="left"
        )
        self.label_info.pack(pady=15)

    def cargar_imagen(self):

        data = cargar_imagen_procesada()

        if data is None:
            return
        
        self.img = data["img"]
        
        # limpiar estados anteriores
        self.mapa_color = None
        self.lista_mapas = []
        self.porcentajes = []
        self.overlay = None

        # 🔥 limpiar cache de clustering automático
        self.scores_cache = None
        self.k_optimo_cache = None

        # mostrar imagen original
        self.mostrar_imagen(self.img)

    def actualizar_info(self):

        texto = "Ambientes:\n"
        for i, p in enumerate(self.porcentajes):
            texto += f"Zona {i}: {p:.2f}%\n"

        self.info.config(text=texto)

    def actualizar_transparencia(self, val):
        self.alpha = float(val)
        self.actualizar_overlay()

    def actualizar_overlay(self):

        # ❌ No hay imagen base
        if self.img is None:
            logging.error("[actualizar_overlay] No hay imagen cargada")
            return

        # ⚠️ No hay mapa → fallback
        if self.mapa_procesado is None:
            logging.warning("[actualizar_overlay] No hay mapa_procesado, mostrando imagen original")

            self.overlay = self.img.copy()
            self.mostrar_imagen(self.overlay, reset_view=False)
            return

        # ✅ Caso normal
        overlay = crear_overlay(self.img, self.mapa_procesado, self.alpha)

        # contornos opcionales
        if self.mostrar_contornos and self.capa_contornos is not None:
            overlay = self.superponer_contornos(overlay, self.capa_contornos)

        self.overlay = overlay
        self.mostrar_imagen(self.overlay, reset_view=False)

    def mostrar_imagen(self, img, reset_view=True):
        self.img_original = img

        if reset_view:
            self.zoom = 1.0
            self.offset_x = 0
            self.offset_y = 0

        self.renderizar_imagen()
    
    def renderizar_imagen(self):
        if self.img_original is None:
            return

        img_rgb = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2RGB)

        h, w = img_rgb.shape[:2]

        new_w = int(w * self.zoom)
        new_h = int(h * self.zoom)

        img_resized = cv2.resize(img_rgb, (new_w, new_h))

        img_pil = Image.fromarray(img_resized)
        self.img_tk = ImageTk.PhotoImage(img_pil)

        self.canvas.delete("all")

        self.canvas.create_image(
            self.offset_x,
            self.offset_y,
            anchor="nw",
            image=self.img_tk
        )

    def zoom_mouse(self, event):
        factor = 1.1 if event.delta > 0 else 0.9
        self.zoom *= factor

        # límites
        self.zoom = max(0.1, min(self.zoom, 5))

        self.renderizar_imagen()

    def iniciar_pan(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def mover_pan(self, event):
        dx = event.x - self.start_x
        dy = event.y - self.start_y

        self.offset_x += dx
        self.offset_y += dy

        self.start_x = event.x
        self.start_y = event.y

        self.renderizar_imagen()
    
    def mostrar_mapa_actual(self):

        if not self.lista_mapas:
            return

        ruta = self.lista_mapas[self.indice_actual]

        # Extraer k
        match = re.search(r"mapa_k_(\d+)", ruta)
        k = match.group(1) if match else "-"

        self.label_k.config(
            text=f"k = {k} ({self.indice_actual+1}/{len(self.lista_mapas)})"
        )

        self.root.title(f"AgroPID - k = {k}")

        mapa = cv2.imread(ruta)

        # Guardar para overlay
        self.mapa_color = mapa

        # Mostrar overlay actualizado
        self.actualizar_overlay()

    def mostrar_exg_cluster(self):

        if self.img is None:
            return

        mapa, porcentajes, cmap = calcular_ambientes_exg(self.img, k=5)

        mapa_color = cmap(mapa)[:, :, :3]
        mapa_color = (mapa_color * 255).astype(np.uint8)
        mapa_color = cv2.cvtColor(mapa_color, cv2.COLOR_RGB2BGR)

        overlay = cv2.addWeighted(self.img, 0.6, mapa_color, 0.4, 0)

        self.mostrar_imagen(overlay)

    def mostrar_exg(self):

        if self.img is None:
            logging.warning("[mostrar_exg] self.img es None")
            return

        #exg = calcular_exg(self.img)
        self.mapa_procesado = mapa_calor_exg(self.img)

        self.mostrar_imagen(self.mapa_procesado)

    def siguiente(self):

        if not self.lista_mapas:
            return

        self.indice_actual = (self.indice_actual + 1) % len(self.lista_mapas)
        self.mostrar_mapa_actual()

    def anterior(self):

        if not self.lista_mapas:
            return

        self.indice_actual = (self.indice_actual - 1) % len(self.lista_mapas)
        self.mostrar_mapa_actual()

    def guardar_imagen(self):

        if self.mapa_procesado is None:
            logging.warning("[guardar_imagen] No hay imagen para guardar")
            return

        from tkinter import filedialog

        ruta = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")]
        )

        if ruta:
            cv2.imwrite(ruta, self.mapa_procesado)

    def limites_ambientes(self):

        if self.img is None:
            logging.warning("[limite_ambientes] No hay imagen")
            return

        # =========================
        # MODO AUTOMÁTICO
        # =========================
        if self.modo.get() == "auto":

            if self.auto_cache is not None:
                print("Usando resultados cacheados")

                mapa = self.auto_cache["mapa"]
                porcentajes = self.auto_cache["porcentajes"]
                cmap = self.auto_cache["cmap"]
                k = self.auto_cache["k"]
                scores = self.auto_cache["scores"]

            else:
                mapa, porcentajes, cmap, k, scores = calcular_ambientes_auto(self.img)

                self.auto_cache = {
                    "mapa": mapa,
                    "porcentajes": porcentajes,
                    "cmap": cmap,
                    "k": k,
                    "scores": scores
                }

        # =========================
        # MODO MANUAL
        # =========================
        else:
            k = self.slider_k.get()
            mapa, porcentajes, cmap = calcular_ambientes(self.img, k)

        # =========================
        # FILTRADO POR TAMAÑO MÍNIMO
        # =========================
        min_area = 00  # 🔥 ajustable luego con slider

        print("Creando copia del mapa")
        print("Filtro de area " + str(min_area))

        try:
            mapa_filtrado = np.copy(mapa)
        except Exception as e:
            print(f"Error al crear mapa: {e}")
            return

        # =========================
        # LISTA PARA GUARDAR ÁREAS
        # =========================
        areas_regiones = []

        # contador global
        total_regiones_eliminadas = 0

        for clase in np.unique(mapa):

            print(f"\n==============================")
            print(f"Analizando la clase {clase}")

            # contador por clase
            regiones_eliminadas = 0

            mask = (mapa == clase).astype(np.uint8)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask,
                connectivity=8
            )

            print(f"Regiones detectadas: {num_labels - 1}")

            for i in range(1, num_labels):  # 0 es fondo

                area = stats[i, cv2.CC_STAT_AREA]

                # =========================
                # GUARDAR ÁREA
                # =========================
                areas_regiones.append(area)

                porcentaje = (i / (num_labels - 1)) * 100

                if area < min_area:

                    regiones_eliminadas += 1
                    total_regiones_eliminadas += 1

                    region = (labels == i)

                    # buscar vecinos
                    dilatada = cv2.dilate(
                        region.astype(np.uint8),
                        np.ones((3,3), np.uint8)
                    )

                    vecinos = mapa[dilatada.astype(bool)]

                    # excluir la misma clase
                    vecinos = vecinos[vecinos != clase]

                    if len(vecinos) > 0:

                        nueva_clase = np.bincount(vecinos).argmax()
                        mapa_filtrado[region] = nueva_clase


            print(f"Regiones eliminadas en clase {clase}: {regiones_eliminadas}")

        print("\n==============================")
        print(f"TOTAL DE REGIONES ELIMINADAS: {total_regiones_eliminadas}")

        # =========================
        # GRÁFICO
        # =========================
        plt.figure(figsize=(12,6))

        plt.hist(
            areas_regiones,
            bins=30,
            #range=(0, 50)
        )

        plt.xlabel("Número de región")
        plt.ylabel("Área de región (px)")
        plt.title("Área de cada región detectada")

        plt.grid(True)

        plt.show()

        # =========================
        # USAR MAPA FILTRADO
        # =========================
        mapa = mapa_filtrado

        # =========================
        # VISUALIZACIÓN
        # =========================
        mapa_color = cmap(mapa)[:, :, :3]
        mapa_color = (mapa_color * 255).astype(np.uint8)

        self.mapa_procesado = cv2.cvtColor(
            mapa_color,
            cv2.COLOR_RGB2BGR
        )

        self.mostrar_imagen(self.mapa_procesado)

        self.label_info.config(
            text=f"Ambientes: {k}"
        )

        # contornos
        self.capa_contornos = self.crear_capa_contornos(mapa)

        self.mapa_color = mapa_color

        self.actualizar_overlay()

    def crear_capa_contornos(self,mapa):

        h, w = mapa.shape

        grosor = max(1, int(min(h, w) * 0.002))  

        capa = np.zeros((h, w, 3), dtype=np.uint8)

        for i in np.unique(mapa):
            mask = (mapa == i).astype(np.uint8) * 255

            contornos, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            cv2.drawContours(capa, contornos, -1, (255, 255, 255), grosor)

        return capa

    def limites(self):

        if self.img is None:
            logging.warning("[limite] No hay imagen")
            return

        # =========================
        # CACHE
        # =========================
        if self.auto_cache is not None:
            print("Usando resultados cacheados")

            mapa = self.auto_cache["mapa"]
            porcentajes = self.auto_cache["porcentajes"]
            cmap = self.auto_cache["cmap"]
            k = self.auto_cache["k"]
            scores = self.auto_cache["scores"]

        else:
            mapa, porcentajes, cmap, k, scores = calcular_ambientes_auto(self.img)

            self.auto_cache = {
                "mapa": mapa,
                "porcentajes": porcentajes,
                "cmap": cmap,
                "k": k,
                "scores": scores
            }

        # =========================
        # VISUALIZACIÓN
        # =========================
        mapa_color = cmap(mapa)[:, :, :3]
        mapa_color = (mapa_color * 255).astype(np.uint8)
        mapa_procesado = cv2.cvtColor(mapa_color, cv2.COLOR_RGB2BGR)

        # guardar en la app
        self.mapa_procesado = mapa_procesado
        self.capa_contornos = self.crear_capa_contornos(mapa)

        # usar overlay central
        self.actualizar_overlay()

    def superponer_contornos(self, img, capa_contornos):

        gray = cv2.cvtColor(capa_contornos, cv2.COLOR_BGR2GRAY)

        # máscara: solo donde hay contorno
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        contornos_color = capa_contornos.copy()

        resultado = img.copy()
        resultado[mask > 0] = contornos_color[mask > 0]

        return resultado
    
    def toggle_contornos(self):
        
        self.mostrar_contornos = not self.mostrar_contornos

        if self.mostrar_contornos:
            self.btn_contornos.config(text="Ocultar límites")
        else:
            self.btn_contornos.config(text="Mostrar límites")

        self.actualizar_overlay()

    def calcular_cobertura(self):

        if self.img is None:
            return

        cobertura, mask = calcular_cobertura(self.img)

        overlay = overlay_cobertura(self.img, mask)

        self.mostrar_imagen(overlay)

        self.info.config(text=f"Cobertura vegetal: {cobertura:.2f}%")
    
    def mostrar_estres(self):

        if self.img is None:
            return

        estres, _ = detectar_estres(self.img)

        mapa = mapa_estres_color(estres)

        overlay = cv2.addWeighted(self.img, 0.6, mapa, 0.4, 0)

        self.mostrar_imagen(overlay)

    def calcular_ambientes(self):
        

        if self.img is None:
            return
        
        mapa_procesado, porcentajes, cmap = calcular_ambientes(self.img)

        # guardar resultados
        self.porcentajes = porcentajes

        # convertir mapa a imagen color
        mapa_color = cmap(mapa_procesado)[:, :, :3]
        mapa_color = (mapa_color * 255).astype(np.uint8)
        mapa_color = cv2.cvtColor(mapa_color, cv2.COLOR_RGB2BGR)

        self.mapa_color = mapa_color

        # mostrar overlay
        self.actualizar_overlay()

        # actualizar info en UI
        #self.actualizar_info()

    def crear_boton(self, parent, texto, comando):

        btn = tk.Label(
            parent,
            text=texto,
            bg=self.BTN_COLOR,
            fg=self.TEXT_COLOR,
            font=("Segoe UI", 10),
            padx=10,
            pady=8,
            cursor="hand2"
        )

        btn.pack(fill="x", padx=15, pady=5)

        btn.bind("<Enter>", lambda e: btn.config(bg=self.BTN_HOVER))
        btn.bind("<Leave>", lambda e: btn.config(bg=self.BTN_COLOR))
        btn.bind("<Button-1>", lambda e: comando())

        return btn
