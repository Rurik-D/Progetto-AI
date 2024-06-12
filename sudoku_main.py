from sud import getGrid 
import cv2 
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_path = 'C:\\Users\\giuse\\Desktop\\Progetto-AI\\aug\\_288_6294564.jpeg'
class OurCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            # nn.Conv2d(1, 5, 3),
            # nn.ReLU(),
            # nn.Conv2d(5, 10, 3),
            # nn.ReLU(),
            # nn.Conv2d(10, 1, 3),
            # nn.ReLU()
            nn.Conv2d(1, 32, 3, padding=1),  # Primo strato convoluzionale
            nn.BatchNorm2d(32),               # Batch normalization
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # Primo strato di pooling
            nn.Conv2d(32, 64, 3, padding=1),  # Secondo strato convoluzionale
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # Secondo strato di pooling
            nn.Conv2d(64, 128, 3, padding=1), # Terzo strato convoluzionale
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                # Terzo strato di pooling
            )
        self.mlp = nn.Sequential(
            # nn.Linear(22*22,10),
            # nn.ReLU(),
            # nn.Linear(10,10)
            nn.Linear(128 * 3 * 3, 256),      # Primo strato lineare
            nn.ReLU(),
            nn.Dropout(0.5),                  # Dropout
            nn.Linear(256, 128),              # Secondo strato lineare
            nn.ReLU(),
            nn.Dropout(0.5),                  # Dropout
            nn.Linear(128, 10)                # Strato di output
        )

    def forward(self, x):
        x = self.cnn(x)
        #print(x.shape)
        #x = torch.flatten(x,1)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.mlp(x)
        return x
    
model = OurCNN().to(device)
model.load_state_dict(torch.load('C:\\Users\\giuse\\Desktop\\Progetto-AI\\Models\\\digits_rec.pth'))
model.eval()

warped, dst_points = getGrid('C:\\Users\\giuse\\Desktop\\Progetto-AI\\aug\\_288_6294564.jpeg')

def zoomCells(warped, dst_points):
    for point in dst_points.tolist():

        punto = (point[0],point[1])

        cv2.circle(warped, punto, 5, (0, 255, 0), -1)
    
    rows, cols = warped.shape[:2]

    for x in range(0, rows-rows//9, rows//9):

        for y in range(0, cols-cols//9, cols//9):

            M = np.float32([[9, 0, -y*9], [0, 9, -x*9]])
            dst_image = cv2.warpAffine(warped, M, (cols, rows))
            digits_rec(dst_image)
            # cv2.imshow("image", dst_image)
            # cv2.waitKey(0)

def digits_rec(image_path):

    # image = Image.open(image_path)

    # image = cv2.imread(image_path, 0)

    image = cv2.resize(image_path, (200,200), interpolation=cv2.INTER_LINEAR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(gray_image.shape[:])

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    image = clahe.apply(gray_image)

    image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
    image = cv2.bitwise_not(image[1])
    cv2.imshow("image-clay", image)
    cv2.waitKey(0)
    image = cv2.resize(image, (28,28), interpolation=cv2.INTER_AREA)
    print(image.shape[:])
    cv2.imshow("image-resize", image)
    cv2.waitKey(0)
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
    print("Classe predetta:", predicted.item())

zoomCells(warped, dst_points)

