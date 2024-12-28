from tkinter import filedialog, messagebox
from threading import Thread
from PIL import Image
from util.language import Language
from util.image_converter import cv2_to_pil_image
from model.grid import Grid
from view.widgets import Widgets
from view.window import Window
from model.digits import get_solved_sudoku
import customtkinter as ctk

class MainController:
    """
        This class manage all the GUI's widgets.
    """
    def __init__(self, root):       
        self.root = root
        self.wndMan = Window(root)
        self.wdgt = Widgets(root)
        self.lang = Language()
        self.selectedGrid = None
        self.solverThread = None
        self.runSolution = False

        self.setButtonsCommand()
        self.updateBtnLang()
        self.switchToMainMenu()


    def setButtonsCommand(self):
        """
            Sets the command of each button in the window.
        """
        # Main menu
        self.wdgt.start_btn.configure(command=self.switchToChooseImageMenu)
        self.wdgt.settings_btn.configure(command=self.switchToSettingsMenu)
        self.wdgt.exit_btn.configure(command=self.checkConfirm)
        self.root.protocol("WM_DELETE_WINDOW", self.checkConfirm)
        # Image load menu
        self.wdgt.load_btn.configure(command=self.openChooseImageWindow)
        self.wdgt.back_btn.configure(command=self.switchToMainMenu)
        # Solve menu
        self.wdgt.solve_btn.configure(command=self.startSudokuSolving)
        self.wdgt.change_btn.configure(command=self.changeSudoku)
        self.wdgt.mainM_btn.configure(command=self.switchToMainMenu)
        # Settings menu
        self.wdgt.lang_btn.configure(command=self.swapLanguage)
        # Theme switch
        self.wdgt.theme_switch.configure(command=self.wndMan.switchTheme)
        
    
    def checkConfirm(self):
        risposta = messagebox.askyesno(self.lang.langMap['conf'], self.lang.langMap['exitQuestion'])
        if risposta:
            self.root.destroy()
            
    def switchToMainMenu(self):
        """
            Hides all the buttons and labels, showing only the widgets on the main
            menù screen.
        """
        self.hideAllButtons()
        self.hideAllLabels()

        
        self.wdgt.solve_btn.configure(state="normal")
        self.wdgt.change_btn.configure(state="normal")
        self.wdgt.mainM_btn.configure(state="normal")

        self.wdgt.start_btn.place(relx=0.5, rely=0.45, anchor=ctk.CENTER)
        self.wdgt.settings_btn.place(relx=0.5, rely=0.6, anchor=ctk.CENTER)
        self.wdgt.exit_btn.place(relx=0.5, rely=0.75, anchor=ctk.CENTER)
        self.wdgt.theme_switch.place(relx=0.92, rely=0.06, anchor=ctk.CENTER)
        self.wdgt.title_lbl.place(relx=0.5, rely=0.18, anchor=ctk.CENTER)
        self.wdgt.credits_lbl.place(relx=0.5, rely=0.97, anchor=ctk.CENTER)
        self.wdgt.theme_lbl.place(relx=0.96, rely=0.06, anchor=ctk.CENTER)

    def switchToChooseImageMenu(self):
        """
            Hides all the buttons, showing only the widgets on the choose image
            menù screen.
        """
        self.hideAllButtons()
        self.wdgt.load_btn.place(relx=0.5, rely=0.45, anchor=ctk.CENTER)
        self.wdgt.back_btn.place(relx=0.5, rely=0.6, anchor=ctk.CENTER)
        self.wdgt.theme_switch.place(relx=0.92, rely=0.06, anchor=ctk.CENTER)

    def switchToSolveMenu(self, imgPath:str):
        """
            Hides all the buttons and labels, showing only the widgets on the solve
            menù screen.
        """
        self.hideAllButtons()
        self.hideAllLabels()
        self.updateChoosenImageLabel(imgPath)
        self.wdgt.chosenImg_lbl.place(relx=0.7, rely=0.5, anchor=ctk.CENTER)
        self.wdgt.solve_btn.place(relx=0.2, rely=0.35, anchor=ctk.CENTER)
        self.wdgt.change_btn.place(relx=0.2, rely=0.5, anchor=ctk.CENTER)
        self.wdgt.mainM_btn.place(relx=0.2, rely=0.65, anchor=ctk.CENTER)

    def switchToSettingsMenu(self):
        """
            Hides all the buttons, showing only the widgets on the settings
            menù screen.
        """
        self.hideAllButtons()
        self.wdgt.lang_btn.place(relx=0.5, rely=0.45, anchor=ctk.CENTER)
        self.wdgt.back_btn.place(relx=0.5, rely=0.6, anchor=ctk.CENTER)
        self.wdgt.theme_switch.place(relx=0.92, rely=0.06, anchor=ctk.CENTER)


    def changeSudoku(self):
        """
            Open the file-explorer window, allowing to change the currently selected
            image.
        """
        self.openChooseImageWindow()
        self.runSolution = False

        self.wdgt.solve_btn.configure(state="normal")
        self.wdgt.change_btn.configure(state="normal")
        self.wdgt.mainM_btn.configure(state="normal")




    def openChooseImageWindow(self):
        """
            Open a file-explorer window which allows the user to select the
            image of a Sudoku that want to solve.
            If the image is not a Sudoku, or the file is not an image, an error
            message appears.
        """
        file_path = filedialog.askopenfilename(title=self.lang.langMap['selectFile'])
       
        if file_path != "":
            self.selectedGrid = Grid(file_path)

            if self.isSudoku(file_path):
                self.switchToSolveMenu(self.selectedGrid.warped)
            else:
                messagebox.showwarning(self.lang.langMap['adv'], self.lang.langMap['selectImg'])



    def startSudokuSolving(self):
        """
            Starts sudoku loading and solving effect threads.
        """
        self.wdgt.solve_btn.configure(state="disabled")
        self.wdgt.change_btn.configure(state="disabled")
        self.wdgt.mainM_btn.configure(state="disabled")
        self.runSolution = True


        self.solverThread = Thread(target=self.solveAndUpdate)
        self.solverThread.start()

    def solveAndUpdate(self):
        """
            Solves the sudoku and updates the sudoku label image by blocking 
            the scanner effect.
        """
        solvedTuple= get_solved_sudoku(self.selectedGrid)
        if solvedTuple[1]:
            solvedSdk = solvedTuple[0]
            self.updateSudokuImageLabel(solvedSdk)

        else:
            messagebox.showwarning(self.lang.langMap['err'], self.lang.langMap['sudokuError'])
            self.wdgt.solve_btn.configure(state="normal")

        self.wdgt.change_btn.configure(state="normal")
        self.wdgt.mainM_btn.configure(state="normal")


        


        
    def updateSudokuImageLabel(self, solvedSdk):
        """
            Converts the NumPy array of the solved sudoku to a PIL image, creates 
            a CTkImage object from the PIL image, update the label image and 
            assign the image to the label to prevent it from being garbage collected.
        """
        new_image = Image.fromarray(solvedSdk)
        new_image = new_image.resize((400, 400))
        new_ctk_image = ctk.CTkImage(light_image=new_image, size=(400, 400))


        self.wdgt.chosenImg_lbl.configure(image=new_ctk_image)
        self.wdgt.chosenImg_lbl.image = new_ctk_image



    def swapLanguage(self, scelta):
        """
            Switch from English to Italian and vice versa.
        """
        
        self.lang.swapLanguage(scelta)
        self.updateBtnLang()

    def updateBtnLang(self):
        """
            Loads the new languange and updates al the buttons's text.
        """
        self.wdgt.start_btn.configure(text=self.lang.langMap['start'])
        self.wdgt.settings_btn.configure(text=self.lang.langMap['settings'])
        self.wdgt.exit_btn.configure(text=self.lang.langMap['exit'])
        self.wdgt.load_btn.configure(text=self.lang.langMap['load'])
        self.wdgt.back_btn.configure(text=self.lang.langMap['back'])
        self.wdgt.solve_btn.configure(text=self.lang.langMap['solve'])
        self.wdgt.change_btn.configure(text=self.lang.langMap['change'])
        self.wdgt.mainM_btn.configure(text=self.lang.langMap['mainMenu'])

    def updateChoosenImageLabel(self, imgPath:str):
        """
            Converts the path to an image, then sets the chosenImg_lbl with
            the chosen image.
        """
        image = cv2_to_pil_image(imgPath)

        tk_img = ctk.CTkImage(light_image=image, dark_image=image, size=(400, 400))
        self.wdgt.chosenImg_lbl = ctk.CTkLabel(self.root, text='', height=415, fg_color="gray", corner_radius=8, image=tk_img)

    def isSudoku(self, file_path:str):
        return file_path and file_path.split('.')[-1].lower() in ('png', 'jpg', 'jpeg') and self.selectedGrid.isGrid        
       

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
