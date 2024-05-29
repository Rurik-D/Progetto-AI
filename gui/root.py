from tkinter import *
import customtkinter as ctk
from button_manager import Btn_manager
from label_manager import Lbl_manager
from window_manager import Wnd_manager



class Root:
    """
        Custom tkinter main window
    """
    def __init__(self):
        self.__root = ctk.CTk()

        wnd_man = Wnd_manager(self.__root)
        lbl_man = Lbl_manager(self.__root)
        btn_man = Btn_manager(self.__root, wnd_man, lbl_man)

        btn_man.main_menu()

        self.__root.mainloop()







