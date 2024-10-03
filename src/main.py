import sys
from os import path

sys.path.append((path.abspath(".") + "\\src\\model"))

print(path.abspath(".") + "\\src\\model")


from tkinter import *
import customtkinter as ctk
from controller.main_controller import MainController
from os import _exit



class Main:
    """
        Custom tkinter main window. Here is where the
        code starts.
    """
    def __init__(self):
        self.root = ctk.CTk()
        self.menuCtrl = MainController(self.root)
        self.root.protocol("WM_DELETE_WINDOW", self.onClose)
        self.root.mainloop()

    def onClose(self):
        """
            Method runned when the window is closed.
        """
        self.root.destroy()
        _exit(0)
        

Main()
