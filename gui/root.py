from tkinter import *
import customtkinter as ctk

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

root = ctk.CTk()
root.geometry("700x500")

switch = False

def btn_function():
    global switch

    if not switch:
        lbl.configure(text = "Tronci >>>>>> Mancini")
        switch = True
    else:
        lbl.configure(text = "")
        switch = False


btn = ctk.CTkButton(root, 
                    text="This is not a button", 
                    command=btn_function,
                    height=100,
                    width=200,
                    font=("Helvetica", 20),
                    text_color="black",
                    fg_color="#1ca9da",
                    hover_color="#1286b9",
                    corner_radius=50,
                    bg_color="#000000",
                    border_width=5,
                    border_color="white",
                    border_spacing=50,
                    state="normal") # "disabled"
                    

btn.place(relx=0.5, rely=0.5, anchor=ctk.CENTER)

lbl = ctk.CTkLabel(root, text="")
lbl.place(relx=0.5, rely=0.7, anchor=ctk.CENTER)


root.mainloop()

