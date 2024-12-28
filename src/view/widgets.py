import customtkinter as ctk
from PIL import Image
from os import path
IMAGES_PATH = path.abspath(".") + "\\src\\resources\\images\\"

class Widgets:
    """
        This class creates all the widget used in the GUI.
    """
    def __init__(self, root):
        self.root = root
        corner = 15
        W = 150
        H = 50

        # Main menu
        self.start_btn = ctk.CTkButton(root, height=H, width=W, corner_radius=corner)
        self.settings_btn = ctk.CTkButton(root,height=H, width=W, corner_radius=corner)
        self.exit_btn = ctk.CTkButton(root, height=H, width=W, corner_radius=corner)
        
        # Choose-image menu
        self.load_btn = ctk.CTkButton(root, height=H, width=W, corner_radius=corner)
        self.back_btn = ctk.CTkButton(root, height=H, width=W, corner_radius=corner)
        
        # Solve menu
        self.solve_btn = ctk.CTkButton(root, height=H, width=W, corner_radius=corner)
        self.change_btn = ctk.CTkButton(root, height=H, width=W, corner_radius=corner)
        self.mainM_btn = ctk.CTkButton(root, height=H, width=W, corner_radius=corner)
        
        # Settings menu
        #self.lang_btn = ctk.CTkButton(root, height=H, width=W, corner_radius=corner)
        lingue = ["Italian", "English"]
        lingua_var = ctk.StringVar(value="English")  # Valore predefinito
        
        # Menù a tendina per selezionare la lingua

        self.lang_btn = ctk.CTkOptionMenu(root, height=H, width=W, variable=lingua_var, values=lingue, corner_radius=corner)
        # Theme switch
        self.theme_switch = ctk.CTkSwitch(root, text="", height=20, width=30)


        titleImg = ctk.CTkImage(light_image=Image.open(IMAGES_PATH + "title_light-no_bg.png"),
                                dark_image=Image.open(IMAGES_PATH + "title_dark-no_bg.png"),
                                size=(500, 100))

        themeImg = ctk.CTkImage(light_image=Image.open(IMAGES_PATH + "light.png"),
                                dark_image=Image.open(IMAGES_PATH + "moon.png"),
                                size=(30, 30))
        
        self.title_lbl = ctk.CTkLabel(root, text='', image=titleImg)
        self.theme_lbl = ctk.CTkLabel(root, text='', image=themeImg)

        self.credits_lbl= ctk.CTkLabel(root, text="Emanuele D'Agostino\tAlessandro Buccioli\tGiuseppe Borracci")
        self.chosenImg_lbl = ctk.CTkLabel(root, text='', height=415, fg_color="gray", corner_radius=8)
