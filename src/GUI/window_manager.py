from tkinter import *
import customtkinter as ctk


class Wnd_manager:
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

        root.geometry("900x550")
        root.maxsize(1100, 650)
        root.minsize(700, 450)
        root.iconbitmap("img\\sudoku.ico")
        root.title("SolveDoku!")


    def switch_theme(self):
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
