from tkinter import *
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading

class Lbl_manager:
    """
        This class manage all the GUI's labels.
    """
    def __init__(self, root):
        self.__root = root
        self.title_lbl = ctk.CTkLabel(root, text='')
        self.theme_lbl = ctk.CTkLabel(root, text='')
        self.credits_lbl= ctk.CTkLabel(root, text="Emanuele D'Agostino\tAlessandro Buccioli\tGiuseppe Borracci")
        self.loadedImg_lbl = ctk.CTkLabel(root, text='')
        self.scanBar_lbl = ctk.CTkLabel(root, text='')


        self.set_images()


    def set_images(self):
        """
            Set the images of theme and title.
        """
        title = ctk.CTkImage(light_image=Image.open('img\\title_light-no_bg.png'),
                             dark_image=Image.open('img\\title_dark-no_bg.png'),
                             size=(500, 100))

        theme = ctk.CTkImage(light_image=Image.open('img\\light.png'),
                             dark_image=Image.open('img\\moon.png'),
                             size=(30, 30))

        self.title_lbl = ctk.CTkLabel(self.__root, text='', image=title)
        self.theme_lbl = ctk.CTkLabel(self.__root, text='', image=theme)


    def show_menu_graphics(self):
        """
            Show all the graphics of the main menu.
        """
        self.title_lbl.place(relx=0.5, rely=0.18, anchor=ctk.CENTER)
        self.credits_lbl.place(relx=0.5, rely=0.97, anchor=ctk.CENTER)
        self.theme_lbl.place(relx=0.96, rely=0.06, anchor=ctk.CENTER)


    def set_loadedImg(self, image):
        """
            Set the image by the loaded path.
        """
        image = cv2_to_pil_image(image)

        tk_img = ctk.CTkImage(light_image=image,
                            dark_image=image,
                            size=(400, 400))
        self.loadedImg_lbl = ctk.CTkLabel(self.__root, text='', height=415, fg_color="gray", 
                                          corner_radius=8, image=tk_img)


    def show_loadedImg(self):
        """
            Show the loaded image.
        """
        self.loadedImg_lbl.place(relx=0.7, rely=0.5, anchor=ctk.CENTER)


    def start_scanning(self):
        self.title_lbl = ctk.CTkLabel(self.loadedImg_lbl, text='', fg_color="#01decf")

    
    def move_scanner(self):
        pass



    def hide_all(self):
        """
            Hide all the graphics.
        """
        self.loadedImg_lbl.place_forget()
        self.title_lbl.place_forget()
        self.credits_lbl.place_forget()
        self.theme_lbl.place_forget()



def cv2_to_pil_image(cv2_image):
    # Converti l'immagine da BGR a RGB
    cv2_image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    # Converti l'immagine in un oggetto PIL
    pil_image = Image.fromarray(cv2_image_rgb)
    return pil_image
