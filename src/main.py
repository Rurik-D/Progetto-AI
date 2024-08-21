from tkinter import *
import customtkinter as ctk
from controller.main_controller import MainController
from os import _exit


class Main:
    """
        Custom tkinter main window
    """
    def __init__(self):
        self.root = ctk.CTk()
        self.menuCtrl = MenuController(self.root)

        self.menuCtrl.switchToMainMenu()

        self.root.protocol("WM_DELETE_WINDOW", self.onClose)

        self.root.mainloop()

    def onClose(self):
        self.root.destroy()
        _exit(0)
        

Main()