from sys import path as sysPath
from os import path, _exit
import customtkinter as ctk

sysPath.append((path.abspath(".") + "\\src\\model"))

from controller.main_controller import MainController


class Main:
    """
        Custom tkinter main window. Here is where the
        code starts.
    """
    def __init__(self):
        self.root = ctk.CTk()
        self.menuCtrl = MainController(self.root)
        self.root.mainloop()

Main()
