from tkinter import filedialog
from tkinter import messagebox
import customtkinter as ctk
from language import load_language
from sys import path

path.insert(1, 'C:\\Users\\halle\\Desktop\\Progetto-AI')

import sud


class Btn_manager:
    """
        This class manage all the GUI's buttons.
    """
    def __init__(self, root, wnd_man, lbl_man):
        corner_rad = 15
        W = 150
        H = 50
        
        self.__lbl_man = lbl_man
        self.lang = "en"
        self.currLang = load_language(self.lang)


        # Main menu
        self.start_btn = ctk.CTkButton(root, height=H, width=W,
                                       corner_radius=corner_rad, command=self.imgLoad_menu)
        self.settings_btn = ctk.CTkButton(root,height=H, width=W,
                                          corner_radius=corner_rad, command=self.setts_menu)
        self.exit_btn = ctk.CTkButton(root, height=H, width=W,
                                      corner_radius=corner_rad, command=root.destroy)
        
        # Image load menu
        self.load_btn = ctk.CTkButton(root, height=H, width=W,
                                      corner_radius=corner_rad, command=self.load_image)
        self.back_btn = ctk.CTkButton(root, height=H, width=W,
                                      corner_radius=corner_rad, command=self.main_menu)
        
        # Solve menu
        self.solve_btn = ctk.CTkButton(root, height=H, width=W,
                                       corner_radius=corner_rad)
        self.change_btn = ctk.CTkButton(root, height=H, width=W,
                                        corner_radius=corner_rad, command=self.load_image)
        self.mainM_btn = ctk.CTkButton(root, height=H, width=W,
                                       corner_radius=corner_rad, command=self.main_menu)
        
        # Settings menu
        self.lang_btn = ctk.CTkButton(root, height=H, width=W,
                                       corner_radius=corner_rad, command=self.change_lang)
        
        # Theme switch
        self.theme_switch = ctk.CTkSwitch(root, text="", height=20, width=30, 
                                          command=wnd_man.switch_theme)

        self.update_btn_lang()


    def main_menu(self):
        """
            Show main menu buttons.
        """
        self.hide_all()
        self.__lbl_man.hide_all()
        self.__lbl_man.show_menu_graphics()
        self.start_btn.place(relx=0.5, rely=0.45, anchor=ctk.CENTER)
        self.settings_btn.place(relx=0.5, rely=0.6, anchor=ctk.CENTER)
        self.exit_btn.place(relx=0.5, rely=0.75, anchor=ctk.CENTER)
        self.theme_switch.place(relx=0.92, rely=0.06, anchor=ctk.CENTER)


    def imgLoad_menu(self):
        """
            Show the image loading menu's buttons.
        """
        self.hide_all()
        self.load_btn.place(relx=0.5, rely=0.45, anchor=ctk.CENTER)
        self.back_btn.place(relx=0.5, rely=0.6, anchor=ctk.CENTER)
        self.theme_switch.place(relx=0.92, rely=0.06, anchor=ctk.CENTER)


    def solve_menu(self, imgPath):
        """
            Show solving menu's buttons.
        """
        self.hide_all()
        self.__lbl_man.hide_all()
        self.__lbl_man.set_loadedImg(imgPath)
        self.__lbl_man.show_loadedImg()
        self.solve_btn.place(relx=0.2, rely=0.35, anchor=ctk.CENTER)
        self.change_btn.place(relx=0.2, rely=0.5, anchor=ctk.CENTER)
        self.mainM_btn.place(relx=0.2, rely=0.65, anchor=ctk.CENTER)


    def setts_menu(self):
        """
            Show settings menu's buttons.
        """
        self.hide_all()
        self.lang_btn.place(relx=0.5, rely=0.45, anchor=ctk.CENTER)
        self.back_btn .place(relx=0.5, rely=0.6, anchor=ctk.CENTER)
        self.theme_switch.place(relx=0.92, rely=0.06, anchor=ctk.CENTER)


    def load_image(self):
        """
            Show solving menu's buttons.
        """
        extensions = ('png', 'jpg', 'jpeg')
        file_path = filedialog.askopenfilename(title=self.currLang['selectFile'])

        # Checks if a file has been selected
        if file_path:
            if file_path.split('.')[-1] in extensions:
                grid, _, isGrid= sud.getGrid(file_path)

                if isGrid:
                    self.solve_menu(grid)
                else:
                    messagebox.showwarning(self.currLang['adv'], self.currLang['selectImg']) 
            else:
                messagebox.showwarning(self.currLang['adv'], self.currLang['selectImg'])


    def change_lang(self):
        self.lang = 'en' if self.lang == 'it' else 'it'
        self.currLang = load_language(self.lang)
        self.update_btn_lang()


    def update_btn_lang(self):
        """
            Loads the new languange and updates al the buttons's text
        """
        self.start_btn.configure(text=self.currLang ['start'])
        self.settings_btn.configure(text=self.currLang ['settings'])
        self.exit_btn.configure(text=self.currLang ['exit'])
        self.load_btn.configure(text=self.currLang ['load'])
        self.back_btn.configure(text=self.currLang ['back'])
        self.solve_btn.configure(text=self.currLang ['solve'])
        self.change_btn.configure(text=self.currLang ['change'])
        self.mainM_btn.configure(text=self.currLang ['mainMenu'])
        self.lang_btn.configure(text=self.currLang ['lang'])


    def hide_all(self):
        """
            Hide all the buttons.
        """
        self.start_btn.place_forget()
        self.settings_btn.place_forget()
        self.exit_btn.place_forget()
        self.load_btn.place_forget()
        self.back_btn.place_forget()
        self.solve_btn.place_forget()
        self.change_btn.place_forget()
        self.mainM_btn.place_forget()
        self.lang_btn.place_forget()
        self.theme_switch.place_forget()
