import tkinter as tk
from app.main_menu import MainMenu

if __name__ == "__main__":
    root = tk.Tk()
    app = MainMenu(root)
    root.mainloop()