from grid import Grid

from tkinter import filedialog

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision import transforms

from PIL import Image


def select_file(filepath=None):
    if filepath != None:
        return filepath
    else:
        filepath = filedialog.askopenfilename(title="Select the dataset file")
        return filepath

def choose_device(feedback=False):
    device = ("cuda" if torch.cuda.is_available()
              else "mps"
              if torch.backends.mps.is_available()
              else "cpu")
    
    if feedback:
        print("Device in use:", device)
    
    return device

device = choose_device()
image_path = select_file('C:\\Users\\giuse\\Desktop\\Progetto-AI\\aug\\_288_6294564.jpeg')

class OurCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
            )
        
        self.mlp = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x
    
model = OurCNN()
model.load_state_dict(torch.load('C:\\Users\\giuse\\Desktop\\Progetto-AI\\src\\model\\ai_models\\digits_rec(v2).pth',torch.device('cpu')))
model.eval()

grid = Grid('C:\\Users\\giuse\\Desktop\\Progetto-AI\\Images\\Sudoku\\_290_2832444.jpeg')


def zoomCells(warped, dst_points):
    for point in dst_points.tolist():

        punto = (int(point[0]),int(point[1]))
        cv2.circle(warped, punto, 5, (0, 255, 0), -1)
    
    rows, cols = warped.shape[:2]
    array_sudoku = []
    for x in range(0, rows-rows//9+1, (rows//9)):
        riga_sud = []
        for y in range(0, cols-cols//9+1, (cols//9)):
            M = np.float32([[9, 0, -y*9], [0, 9, -x*9]])
            dst_image = cv2.warpAffine(warped, M, (cols, rows))
            gray = cv2.cvtColor(dst_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.bitwise_not(gray)
            # Applica una soglia per binarizzare l'immagine
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Trova i contorni nell'immagine
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Trova il contorno più grande (presumibilmente il bordo esterno della cella)
            c = max(contours, key=cv2.contourArea)

            # Crea una maschera nera
            mask = np.zeros_like(gray)

            # Disegna un riempimento del contorno della cella
            cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)

            # Erodi la maschera per escludere il bordo esterno (e lasciare solo il numero)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=2)

            # Applica la maschera all'immagine originale
            result = cv2.bitwise_and(dst_image, dst_image, mask=mask)

            # Riempie l'esterno del numero con bianco
            result[mask == 0] = 255

            # Mostra l'immagine risultante
            if len(array_sudoku) == 6:
                result = cv2.resize(result, (200, 200), interpolation=cv2.INTER_LINEAR)
                gray_image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
                image = clahe.apply(gray_image)
                image = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY)
                image = cv2.bitwise_not(image[1])
                cv2.imshow("image", image)
                cv2.waitKey(0)
            predict = digits_rec(result)
            riga_sud.append(predict)
        array_sudoku.append(riga_sud)
    #print("Vecchio array: [[0, 5, 0, 6, 8, 0, 0, 6, 0], [2, 0, 0, 0, 0, 0, 0, 0, 5], [0, 0, 1, 0, 0, 7, 0, 0, 0], [5, 0, 0, 2, 0, 0, 5, 0, 0], [4, 0, 0, 0, 0, 0, 0, 0, 3], [0, 0, 3, 0, 0, 4, 0, 0, 2], [0, 5, 0, 7, 0, 0, 3, 0, 0], [8, 0, 0, 0, 0, 0, 0, 0, 1], [0, 9, 0, 0, 4, 5, 0, 7, 0]]\n\n")
    print(array_sudoku)

def digits_rec(image_path):

    # image = Image.open(image_path)

    # image = cv2.imread(image_path, 0)

    image = cv2.resize(image_path, (200,200), interpolation=cv2.INTER_LINEAR)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    image = clahe.apply(gray_image)
    image = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY)
    image = cv2.bitwise_not(image[1])

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # medianBlur = cv2.medianBlur(blurred,5)
    # th = cv2.adaptiveThreshold(medianBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                            cv2.THRESH_BINARY, 11, 2)
    # _, image = cv2.threshold(th, 160, 255, cv2.THRESH_BINARY_INV)
    
    #cv2.imshow("img", image)
    #cv2.waitKey(0)
    image = cv2.resize(image, (28,28), interpolation=cv2.INTER_AREA)

    # Applica le trasformazioni all'immagine
    transform = transforms.Compose([
        #transforms.Resize((28, 28)),
        #transforms.Grayscale(num_output_channels=1),# Ridimensiona l'immagine alle dimensioni di input del modello
        transforms.ToTensor()    # Converte l'immagine in un tensore
        #transforms.Lambda(invert_colors)
        
    ])

    # Applica le trasformazioni e aggiunge una dimensione di batch
    image_tensor = transform(image).unsqueeze(0)

    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

zoomCells(grid.warped, grid.dstPoints)

