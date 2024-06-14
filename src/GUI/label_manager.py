import customtkinter as ctk
from PIL import Image
import cv2
from scanner_effect import Scanner

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
        self.scan = Scanner()

        self.set_images()


    def set_images(self):
        """
            Set the images of theme and title.
        """
        title = ctk.CTkImage(light_image=Image.open('gui\\img\\title_light-no_bg.png'),
                             dark_image=Image.open('gui\\img\\title_dark-no_bg.png'),
                             size=(500, 100))

        theme = ctk.CTkImage(light_image=Image.open('gui\\img\\light.png'),
                             dark_image=Image.open('gui\\img\\moon.png'),
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
        self.scan.setLabel(self.loadedImg_lbl)
        self.loadedImg_lbl.place(relx=0.7, rely=0.5, anchor=ctk.CENTER)


    def scanning_switch(self, stop=False):
        if self.scan.moving or stop:
            self.scan.stop()
        else:
            self.scan.packScanner()
            self.scan.start()

    def hide_all(self):
        """
            Hide all the graphics.
        """
        self.loadedImg_lbl.place_forget()
        self.title_lbl.place_forget()
        self.credits_lbl.place_forget()
        self.theme_lbl.place_forget()


def cv2_to_pil_image(cv2_image):
    cv2_image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv2_image_rgb)
    return pil_image
