from tkinter import *
import customtkinter as ctk
from button_manager import Btn_manager
from label_manager import Lbl_manager
from window_manager import Wnd_manager
from time import sleep
from os import _exit

class Root:
    """
        Custom tkinter main window
    """
    def __init__(self):
        self.root = ctk.CTk()
        self.wnd_man = Wnd_manager(self.root)
        self.lbl_man = Lbl_manager(self.root)
        self.btn_man = Btn_manager(self.root, self.wnd_man, self.lbl_man)

        self.btn_man.main_menu()

        self.root.protocol("WM_DELETE_WINDOW", self.onClose)

        self.root.mainloop()

    def onClose(self):
        self.root.destroy()
        _exit(0)
        

