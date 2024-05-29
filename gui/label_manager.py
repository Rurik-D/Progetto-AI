from tkinter import *
import customtkinter as ctk
from PIL import Image


class Lbl_manager:
    """
        This class manage all the GUI's labels.
    """
    def __init__(self, root):

        self.title_lbl = ctk.CTkLabel(root, text="")

        img = ctk.CTkImage(light_image=Image.open('title_light-no_bg.png'),
                           dark_image=Image.open('title_dark-no_bg.png'),
                           size=(500, 100))

        self.title_lbl = ctk.CTkLabel(root, text='', image=img)


    def hide_title(self):
        
        self.title_lbl.place_forget()


    def show_title(self):
        self.title_lbl.place(relx=0.5, rely=0.15, anchor=ctk.CENTER)
