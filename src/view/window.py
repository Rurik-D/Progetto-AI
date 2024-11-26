from tkinter import *
import customtkinter as ctk
from os import path

class Window:
    """
        This class manage all the graphics options of
        the window.
    """
    def __init__(self, root):
        self.__dark_theme = True
        self.__set_window(root)


    def __set_window(self, root):
        """
            Set window values and options.
        """
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")


        screen_width = root.winfo_screenwidth()  # Larghezza dello schermo
        screen_height = root.winfo_screenheight()  # Altezza dello schermo


        root.geometry(f"1000x650+{(screen_width-1000)//2}+{(screen_height-650)//2}")
        root.maxsize(1100, 650)
        root.minsize(700, 450)
        root.iconbitmap(path.abspath(".") + "\\src\\resources\\images\\sudoku.ico")
        root.title("SolveDoku!")


    def switchTheme(self):
        """
            Switch theme from dark to light and vice versa.
        """
        theme_color = "blue"

        if not self.__dark_theme :
            theme = "dark"
            self.__dark_theme = True
        else:
            theme = "light"
            self.__dark_theme = False
        
        ctk.set_appearance_mode(theme)
        ctk.set_default_color_theme(theme_color)
