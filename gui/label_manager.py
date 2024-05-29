from tkinter import *
import customtkinter as ctk
from PIL import Image


class Lbl_manager:
    """
        This class manage all the GUI's labels.
    """
    def __init__(self, root):
        self.__root = root
        self.title_lbl = ctk.CTkLabel(root, text="")
        self.loadedImg_lbl= ctk.CTkLabel(root, text="")

        self.set_title()


    def set_title(self):
        img = ctk.CTkImage(light_image=Image.open('img\\title_light-no_bg.png'),
                           dark_image=Image.open('img\\title_dark-no_bg.png'),
                           size=(500, 100))

        self.title_lbl = ctk.CTkLabel(self.__root, text='', image=img)

    def set_image_label(self, path):
        img = ctk.CTkImage(light_image=Image.open(path),
                    dark_image=Image.open(path),
                    size=(400, 400))
        self.loadedImg_lbl = ctk.CTkLabel(self.__root, text='', image=img)

    def hide_title(self):
        """
            Hide the title label.
        """
        self.title_lbl.place_forget()


    def hide_loadedImg(self):
        self.loadedImg_lbl.place_forget()


    def show_title(self):
        """
            Show the title label.
        """
        self.title_lbl.place(relx=0.5, rely=0.18, anchor=ctk.CENTER)

    
    def show_loadedImg(self):
        self.loadedImg_lbl.place(relx=0.7, rely=0.5, anchor=ctk.CENTER)

