from tkinter import filedialog
from tkinter import messagebox
import customtkinter as ctk
from util.language import load_language
from util.image_converter import cv2_to_pil_image
from model.grid import Grid
from view.widgets import Widgets
from view.scanner_effect import ScannerEffect


class MenuController:
    """
        This class manage all the GUI's buttons.
    """
    def __init__(self, root, wnd_man):       
        self.wnd_man = wnd_man
        self.root = root
        self.lang = "en"
        self.currLang = load_language(self.lang)
        self.scanEffect = ScannerEffect()

        self.wdgt = Widgets(root)

        self.setButtonsCommand()


    def setButtonsCommand(self):
        # Main menu
        self.wdgt.start_btn.configure(command=self.switchToChooseImageMenu)
        self.wdgt.settings_btn.configure(command=self.switchToSettingsMenu)
        self.wdgt.exit_btn.configure(command=self.root.destroy)
        
        # Image load menu
        self.wdgt.load_btn.configure(command=self.openChooseImageWindow)
        self.wdgt.back_btn.configure(command=self.switchToMainMenu)
        
        # Solve menu
        self.wdgt.solve_btn.configure(command=self.scanning_switch)
        self.wdgt.change_btn.configure(command=self.openChooseImageWindow)
        self.wdgt.mainM_btn.configure(command=self.switchToMainMenu)
        
        # Settings menu
        self.wdgt.lang_btn.configure(command=self.change_lang)
        
        # Theme switch
        self.wdgt.theme_switch.configure(command=self.wnd_man.switch_theme)

        
    def switchToMainMenu(self):
        """
            Show main menu buttons.
        """
        self.hideAllButtons()
        self.scanning_switch(stop=True)
        self.hideAllLabels()
        self.show_menu_graphics()
        self.wdgt.start_btn.place(relx=0.5, rely=0.45, anchor=ctk.CENTER)
        self.wdgt.settings_btn.place(relx=0.5, rely=0.6, anchor=ctk.CENTER)
        self.wdgt.exit_btn.place(relx=0.5, rely=0.75, anchor=ctk.CENTER)
        self.wdgt.theme_switch.place(relx=0.92, rely=0.06, anchor=ctk.CENTER)


    def switchToChooseImageMenu(self):
        """
            Show the image loading menu's buttons.
        """
        self.hideAllButtons()
        self.wdgt.load_btn.place(relx=0.5, rely=0.45, anchor=ctk.CENTER)
        self.wdgt.back_btn.place(relx=0.5, rely=0.6, anchor=ctk.CENTER)
        self.wdgt.theme_switch.place(relx=0.92, rely=0.06, anchor=ctk.CENTER)


    def switchToSolveMenu(self, imgPath):
        """
            Show solving menu's buttons.
        """
        self.hideAllButtons()
        self.hideAllLabels()
        self.setChosenImg(imgPath)
        self.showChosenImg()
        self.wdgt.solve_btn.place(relx=0.2, rely=0.35, anchor=ctk.CENTER)
        self.wdgt.change_btn.place(relx=0.2, rely=0.5, anchor=ctk.CENTER)
        self.wdgt.mainM_btn.place(relx=0.2, rely=0.65, anchor=ctk.CENTER)


    def switchToSettingsMenu(self):
        """
            Show settings menu's buttons.
        """
        self.hideAllButtons()
        self.wdgt.lang_btn.place(relx=0.5, rely=0.45, anchor=ctk.CENTER)
        self.wdgt.back_btn .place(relx=0.5, rely=0.6, anchor=ctk.CENTER)
        self.wdgt.theme_switch.place(relx=0.92, rely=0.06, anchor=ctk.CENTER)


    def show_menu_graphics(self):
        """
            Show all the graphics of the main menu.
        """
        self.wdgt.title_lbl.place(relx=0.5, rely=0.18, anchor=ctk.CENTER)
        self.wdgt.credits_lbl.place(relx=0.5, rely=0.97, anchor=ctk.CENTER)
        self.wdgt.theme_lbl.place(relx=0.96, rely=0.06, anchor=ctk.CENTER)


    def openChooseImageWindow(self):
        """
            Show solving menu's buttons.
        """
        self.scanning_switch(stop=True)
        file_path = filedialog.askopenfilename(title=self.currLang['selectFile'])
        # Checks if a file has been selected
        if file_path:
            if file_path.split('.')[-1] in ('png', 'jpg', 'jpeg'):
                grid = Grid(file_path)

                if grid.isGrid:
                    self.switchToSolveMenu(grid.warped)
                else:
                    messagebox.showwarning(self.currLang['adv'], self.currLang['selectImg']) 
            else:
                messagebox.showwarning(self.currLang['adv'], self.currLang['selectImg'])
                

    def setChosenImg(self, image):
        """
            Set the image by the loaded path.
        """
        image = cv2_to_pil_image(image)

        tk_img = ctk.CTkImage(light_image=image, dark_image=image, size=(400, 400))
        self.wdgt.chosenImg_lbl = ctk.CTkLabel(self.root, text='', height=415, fg_color="gray", corner_radius=8, image=tk_img)


    def showChosenImg(self):
        """
            Show the loaded image.
        """
        self.scanEffect.setLabels(self.wdgt.chosenImg_lbl)
        self.wdgt.chosenImg_lbl.place(relx=0.7, rely=0.5, anchor=ctk.CENTER)


    def scanning_switch(self, stop=False):
        if self.scanEffect.moving or stop:
            self.scanEffect.stop()
        else:
            self.scanEffect.packScanner()
            self.scanEffect.start()


    def change_lang(self):
        self.lang = 'en' if self.lang == 'it' else 'it'
        self.currLang = load_language(self.lang)
        self.update_btn_lang()


    def update_btn_lang(self):
        """
            Loads the new languange and updates al the buttons's text
        """
        self.wdgt.start_btn.configure(text=self.currLang ['start'])
        self.wdgt.settings_btn.configure(text=self.currLang ['settings'])
        self.wdgt.exit_btn.configure(text=self.currLang ['exit'])
        self.wdgt.load_btn.configure(text=self.currLang ['load'])
        self.wdgt.back_btn.configure(text=self.currLang ['back'])
        self.wdgt.solve_btn.configure(text=self.currLang ['solve'])
        self.wdgt.change_btn.configure(text=self.currLang ['change'])
        self.wdgt.mainM_btn.configure(text=self.currLang ['mainMenu'])
        self.wdgt.lang_btn.configure(text=self.currLang ['lang'])


    def hideAllButtons(self):
        """
            Hide all the buttons.
        """
        self.wdgt.start_btn.place_forget()
        self.wdgt.settings_btn.place_forget()
        self.wdgt.exit_btn.place_forget()
        self.wdgt.load_btn.place_forget()
        self.wdgt.back_btn.place_forget()
        self.wdgt.solve_btn.place_forget()
        self.wdgt.change_btn.place_forget()
        self.wdgt.mainM_btn.place_forget()
        self.wdgt.lang_btn.place_forget()
        self.wdgt.theme_switch.place_forget()

    
    def hideAllLabels(self):
        """
            Hide all the graphics.
        """
        self.wdgt.chosenImg_lbl.place_forget()
        self.wdgt.title_lbl.place_forget()
        self.wdgt.credits_lbl.place_forget()
        self.wdgt.theme_lbl.place_forget()
