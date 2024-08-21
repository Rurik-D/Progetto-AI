import customtkinter as ctk
from util.language import load_language
from PIL import Image
from os import path


class Widgets:
    def __init__(self, root):
        lang = load_language("en")

        corner = 15
        W = 150
        H = 50

        # Main menu
        self.start_btn = ctk.CTkButton(root, height=H, width=W, corner_radius=corner, text=lang['start'])
        self.settings_btn = ctk.CTkButton(root,height=H, width=W, corner_radius=corner, text=lang['settings'])
        self.exit_btn = ctk.CTkButton(root, height=H, width=W, corner_radius=corner, text=lang['exit'])
        
        # Choose-image menu
        self.load_btn = ctk.CTkButton(root, height=H, width=W, corner_radius=corner, text=lang['load'])
        self.back_btn = ctk.CTkButton(root, height=H, width=W, corner_radius=corner, text=lang['back'])
        
        # Solve menu
        self.solve_btn = ctk.CTkButton(root, height=H, width=W, corner_radius=corner, text=lang['solve'])
        self.change_btn = ctk.CTkButton(root, height=H, width=W, corner_radius=corner, text=lang['change'])
        self.mainM_btn = ctk.CTkButton(root, height=H, width=W, corner_radius=corner, text=lang['mainMenu'])
        
        # Settings menu
        self.lang_btn = ctk.CTkButton(root, height=H, width=W, corner_radius=corner, text=lang['lang'])
        
        # Theme switch
        self.theme_switch = ctk.CTkSwitch(root, text="", height=20, width=30)


        titleImg = ctk.CTkImage(light_image=Image.open(path.abspath(".") + "\\src\\resources\\images\\title_light-no_bg.png"),
                                dark_image=Image.open(path.abspath(".") + "\\src\\resources\\images\\title_dark-no_bg.png"),
                                size=(500, 100))

        themeImg = ctk.CTkImage(light_image=Image.open(path.abspath(".") + "\\src\\resources\\images\\light.png"),
                                dark_image=Image.open(path.abspath(".") + "\\src\\resources\\images\\moon.png"),
                                size=(30, 30))
        
        self.title_lbl = ctk.CTkLabel(root, text='', image=titleImg)
        self.theme_lbl = ctk.CTkLabel(root, text='', image=themeImg)

        self.credits_lbl= ctk.CTkLabel(root, text="Emanuele D'Agostino\tAlessandro Buccioli\tGiuseppe Borracci")
        self.chosenImg_lbl = ctk.CTkLabel(root, text='', height=415, fg_color="gray", corner_radius=8)


