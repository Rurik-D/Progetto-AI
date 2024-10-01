import customtkinter as ctk
from threading import Thread
from time import sleep

class ScannerEffect:
    def __init__(self):
        self.vScannerLbl = None
        self.hScannerLbl = None
        self.moving = False
        self.scanThread = None
        self.color = None
        self.speed = 0.04
        self.delay = 0.04
        self.color_speed = int("000800", 16)

        self.colorReset()


    def setScannerLabels(self, container:ctk.CTkLabel):
        self.vScannerLbl = ctk.CTkLabel(container, text='', fg_color=f"#{self.getColor()}", height=400, width=1, corner_radius=6)
        self.hScannerLbl = ctk.CTkLabel(container, text='', fg_color=f"#{self.getColor()}", height=1, width=400, corner_radius=2)


    def getColor(self) -> str:
        return hex(self.color)[2:].rjust(6, '0')

    def colorReset(self):
        self.color = 122575

    def packScanner(self):
        self.vScannerLbl.place(relx=0.025, rely=0.5, anchor=ctk.CENTER)
        self.hScannerLbl.place(relx=0.5, rely=0.025, anchor=ctk.CENTER)


    def start(self):
        self.moving = True
        self.colorReset()
        self.scanThread = Thread(target=self.move_scanner)
        self.scanThread.start()

    def stop(self):
        if self.moving:
            self.moving = False

    def move_scanner(self):
        position = 0.025
        color_direction = -self.color_speed
        direction = self.speed
        while self.moving:
            if position >= 0.95:
                direction = -self.speed
            elif position <= 0.025:
                direction = self.speed

            if self.color <= 90000:
                color_direction = self.color_speed
            elif self.color >= 130000:
                color_direction = -self.color_speed

            position += direction
            self.color += color_direction

            self.updateLabels(position)

            sleep(self.delay)

        self.vScannerLbl.place_forget()
        self.hScannerLbl.place_forget()


    def scanning_switch(self, stop=False):
        if self.moving or stop:
            self.stop()
        else:
            self.packScanner()
            self.start()

    def updateLabels(self, pos:float):
        self.vScannerLbl.place_forget()
        self.hScannerLbl.place_forget()
        self.vScannerLbl.configure(fg_color=f"#{self.getColor()}")
        self.hScannerLbl.configure(fg_color=f"#{self.getColor()}")
        self.vScannerLbl.place(relx=pos, rely=0.5, anchor=ctk.CENTER)
        self.hScannerLbl.place(relx=0.5, rely=pos, anchor=ctk.CENTER)
    

