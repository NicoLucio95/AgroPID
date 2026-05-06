import os
import tkinter as tk

from app.gui import AgroApp
from PIL import Image, ImageTk, ImageEnhance
from app.gui_cultivo import CultivoApp


class MainMenu:

    def __init__(self, root):

        self.root = root
        self.root.title("Procesamiento de Imágenes Digitales Agrícolas")
        self.root.geometry("400x400")
        self.root.resizable(False, False)

        # =========================
        # FONDO
        # =========================
        ruta = os.path.join(os.path.dirname(__file__), "..", "assets", "fondo.jpg")
        ruta = os.path.abspath(ruta)

        self.bg_image = Image.open(ruta).resize((400, 400))

        # oscurecer para contraste
        enhancer = ImageEnhance.Brightness(self.bg_image)
        self.bg_image = enhancer.enhance(0.5)

        self.bg_photo = ImageTk.PhotoImage(self.bg_image)

        self.bg_label = tk.Label(root, image=self.bg_photo)
        self.bg_label.place(relwidth=1, relheight=1)

        # =========================
        # CONTENEDOR CENTRAL
        # =========================
        container = tk.Frame(root, bg="#000000", bd=0)
        container.place(relx=0.5, rely=0.5, anchor="center")

        # =========================
        # TITULO
        # =========================
        titulo = tk.Label(
            container,
            text="PIDAgro",
            font=("Segoe UI", 24, "bold"),
            fg="white",
            bg="#000000"
        )
        titulo.pack(pady=(10, 20))

        # =========================
        # ESTILO BOTONES
        # =========================
        def crear_boton(texto, comando):
            btn = tk.Label(
                container,
                text=texto,
                font=("Segoe UI", 11),
                fg="black",
                bg="white",
                width=22,
                height=2,
                cursor="hand2"
            )

            btn.pack(pady=8)

            # hover
            btn.bind("<Enter>", lambda e: btn.config(bg="#e6e6e6"))
            btn.bind("<Leave>", lambda e: btn.config(bg="white"))

            # click
            btn.bind("<Button-1>", lambda e: comando())

            return btn

        # =========================
        # BOTONES
        # =========================
        crear_boton("Ambientes", self.abrir_analizador)
        crear_boton("Cultivo", self.mapa_calor)
        crear_boton("❌ Salir", self.root.quit)

    # =========================
    # FUNCIONES
    # =========================
    def abrir_analizador(self):
        self.root.withdraw()
        nueva = tk.Toplevel()
        AgroApp(nueva)
        nueva.protocol("WM_DELETE_WINDOW", lambda: self.volver_menu(nueva))

    def mapa_calor(self):
        self.root.withdraw()
        nueva = tk.Toplevel()
        CultivoApp(nueva)
        nueva.protocol("WM_DELETE_WINDOW", lambda: self.volver_menu(nueva))

    def volver_menu(self, ventana):
        ventana.destroy()
        self.root.deiconify()