from tkinter import *
import customtkinter as ctk

class Btn_manager:
    """
        This class manage all the GUI's buttons.
    """
    def __init__(self, root, wnd_man):
        self.start_btn = ctk.CTkButton(root, 
                                       text="Start",
                                       height=50,
                                       width=150,
                                       corner_radius=15,
                                       command=wnd_man.switch_theme)
        self.settings_btn = ctk.CTkButton(root, 
                                          text="Settings",
                                          height=50,
                                          width=150,
                                          corner_radius=15)
        self.exit_btn = ctk.CTkButton(root, 
                                      text="Exit",
                                      height=50,
                                      width=150,
                                      corner_radius=15,
                                      command=lambda: self.hide_all())


    def main_menu(self):
        """
            Show main menu buttons.
        """
        self.start_btn.place(relx=0.5, rely=0.4, anchor=ctk.CENTER)
        self.settings_btn.place(relx=0.5, rely=0.55, anchor=ctk.CENTER)
        self.exit_btn.place(relx=0.5, rely=0.7, anchor=ctk.CENTER)


    def hide_all(self):
        """
            Hide all the buttons.
        """
        self.start_btn.place_forget()
        self.settings_btn.place_forget()
        self.exit_btn.place_forget()





