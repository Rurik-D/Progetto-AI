from tkinter import filedialog
from tkinter import messagebox
import customtkinter as ctk

class Btn_manager:
    """
        This class manage all the GUI's buttons.
    """
    def __init__(self, root, wnd_man, lbl_man):
        corner_rad = 15
        W = 150
        H = 50
        
        self.__lbl_man = lbl_man

        # Main menu
        self.start_btn = ctk.CTkButton(root, 
                                       text="Start",
                                       height=H,
                                       width=W,
                                       corner_radius=corner_rad,
                                       command=self.imgLoad_menu)
        self.settings_btn = ctk.CTkButton(root, 
                                          text="Settings",
                                          height=H,
                                          width=W,
                                          corner_radius=corner_rad,
                                          command=wnd_man.switch_theme)
        self.exit_btn = ctk.CTkButton(root, 
                                      text="Exit",
                                      height=H,
                                      width=W,
                                      corner_radius=corner_rad,
                                      command=root.destroy)
        
        # Image load menu
        self.load_btn = ctk.CTkButton(root, 
                                      text="Load",
                                      height=H,
                                      width=W,
                                      corner_radius=corner_rad,
                                      command=self.load_image)
        self.back_btn = ctk.CTkButton(root, 
                                      text="Back",
                                      height=H,
                                      width=W,
                                      corner_radius=corner_rad,
                                      command=self.main_menu)
        
        # Solve menu
        self.solve_btn = ctk.CTkButton(root, 
                                       text="Solve!",
                                       height=H,
                                       width=W,
                                       corner_radius=corner_rad)
        self.change_btn = ctk.CTkButton(root, 
                                        text="Change",
                                        height=H,
                                        width=W,
                                        corner_radius=corner_rad,
                                        command=self.load_image)
        self.mainM_btn = ctk.CTkButton(root, 
                                       text="Main Menu",
                                       height=H,
                                       width=W,
                                       corner_radius=corner_rad,
                                       command=self.main_menu)


    def main_menu(self):
        """
            Show main menu buttons.
        """
        self.hide_all()
        self.__lbl_man.show_title()
        self.__lbl_man.hide_loadedImg()
        self.start_btn.place(relx=0.5, rely=0.45, anchor=ctk.CENTER)
        self.settings_btn.place(relx=0.5, rely=0.6, anchor=ctk.CENTER)
        self.exit_btn.place(relx=0.5, rely=0.75, anchor=ctk.CENTER)


    def imgLoad_menu(self):
        """
            Show the image loading menu's buttons.
        """
        self.hide_all()
        self.load_btn.place(relx=0.5, rely=0.45, anchor=ctk.CENTER)
        self.back_btn.place(relx=0.5, rely=0.6, anchor=ctk.CENTER)


    def solve_menu(self, imgPath):
        """
            Show solving menu's buttons.
        """
        self.hide_all()
        self.__lbl_man.hide_title()
        self.__lbl_man.hide_loadedImg()
        self.__lbl_man.set_image_label(imgPath)
        self.__lbl_man.show_loadedImg()
        self.solve_btn.place(relx=0.2, rely=0.35, anchor=ctk.CENTER)
        self.change_btn.place(relx=0.2, rely=0.5, anchor=ctk.CENTER)
        self.mainM_btn.place(relx=0.2, rely=0.65, anchor=ctk.CENTER)


    def load_image(self):
        """
            Show solving menu's buttons.
        """
        extensions = ('png', 'jpg', 'jpeg')
        file_path = filedialog.askopenfilename(title="Select a file")

        # Checks if a file has been selected
        if file_path:
            if file_path.split('.')[-1] in extensions:
                self.solve_menu(file_path)
            else:
                messagebox.showwarning("Advice", "Select an image!")


    def setts_menu(self):
        """
            Show settings menu's buttons.
        """
        pass


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


