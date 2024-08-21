from tkinter import *
import customtkinter as ctk
from controller.window_manager import Wnd_manager
from controller.main_controller import MenuController
from os import _exit


class Main:
    """
        Custom tkinter main window
    """
    def __init__(self):
        self.root = ctk.CTk()
        self.wnd_man = Wnd_manager(self.root)
        self.menuCtrl = MenuController(self.root, self.wnd_man)

        self.menuCtrl.switchToMainMenu()

        self.root.protocol("WM_DELETE_WINDOW", self.onClose)

        self.root.mainloop()

    def onClose(self):
        self.root.destroy()
        _exit(0)
        

Main()