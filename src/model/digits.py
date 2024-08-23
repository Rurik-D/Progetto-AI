from grid import Grid

from tkinter import filedialog

import cv2
import numpy as np

import torch
from torch import nn
from torchvision import transforms

DATABASE_PATH = 'C:\\Users\\giuse\\Desktop\\Progetto-AI\\src\\model\\ai_models\\digits_rec(v2).pth'
IMAGE_PATH = 'C:\\Users\\giuse\\Desktop\\Progetto-AI\\aug\\_288_6294564.jpeg'
GRID_PATH = 'C:\\Users\\giuse\\Desktop\\Progetto-AI\\Images\\Sudoku\\_290_2832444.jpeg'

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

def dst_points_drawer(warped, dstPoints):
    for point in dstPoints.tolist():
        X = int(point[0])
        Y = int(point[1])
        coordinates = (X, Y)
        DOT_SIZE = 5
        GREEN = (0, 255, 0)
        cv2.circle(warped, coordinates, DOT_SIZE, GREEN, -1)

def BGR2GRAY_selective(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    return gray_image

def clahe_equalizer(image):
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    image = clahe.apply(image)
    return image

def filters_applier(raw_image):
    IMAGE_DEFAULT_SIZE = (200, 200)
    resized_image = cv2.resize(raw_image, IMAGE_DEFAULT_SIZE, interpolation=cv2.INTER_LINEAR)
    gray_image = BGR2GRAY_selective(resized_image)
    equalized_image = clahe_equalizer(gray_image)
    thrs_image = cv2.threshold(equalized_image, 90, 255, cv2.THRESH_BINARY)
    preprocessed_image = cv2.bitwise_not(thrs_image[1])
    return preprocessed_image

def cell_cleaner(img):
    rows = np.shape(img)[0]
    for i in range(rows):
        #Floodfilling the outermost layer
        cv2.floodFill(img, None, (0, i), 0)
        cv2.floodFill(img, None, (i, 0), 0)
        cv2.floodFill(img, None, (rows-1, i), 0)
        cv2.floodFill(img, None, (i, rows-1), 0)
        #Floodfilling the second outermost layer
        cv2.floodFill(img, None, (1, i), 1)
        cv2.floodFill(img, None, (i, 1), 1)
        cv2.floodFill(img, None, (rows - 2, i), 1)
        cv2.floodFill(img, None, (i, rows - 2), 1)
    
    #Finding the bounding box of the number in the cell
    rowtop = None
    
    rowbottom = None
    colleft = None
    colright = None
    thresholdBottom = 50
    thresholdTop = 50
    thresholdLeft = 50
    thresholdRight = 50
    center = rows // 2
    for i in range(center, rows):
        if rowbottom is None:
            temp = img[i]
            if sum(temp) < thresholdBottom or i == rows-1:
                rowbottom = i
        if rowtop is None:
            temp = img[rows-i-1]
            if sum(temp) < thresholdTop or i == rows-1:
                rowtop = rows-i-1
        if colright is None:
            temp = img[:, i]
            if sum(temp) < thresholdRight or i == rows-1:
                colright = i
        if colleft is None:
            temp = img[:, rows-i-1]
            if sum(temp) < thresholdLeft or i == rows-1:
                colleft = rows-i-1

    # Centering the bounding box's contents
    newimg = np.zeros(np.shape(img))
    
    startatX = (rows + colleft - colright)//2
    startatY = (rows - rowbottom + rowtop)//2
    for y in range(startatY, (rows + rowbottom - rowtop)//2):
        for x in range(startatX, (rows - colleft + colright)//2):
            newimg[y, x] = img[rowtop + y - startatY, colleft + x - startatX]
    return np.float32(newimg)

def zoomCells(warped, dst_points):
    rows, cols = warped.shape[:2]
    array_sudoku = []
    
    ROW_SIZE = rows-rows//9
    CELL_LENGTH = rows//9
    COL_SIZE = cols-cols//9
    CELL_HEIGHT = cols//9
    
    ZOOM_LEVEL = 9
    
    for x in range(0, ROW_SIZE+1, CELL_LENGTH):
        riga_sud = []
        X_traslation = -x*ZOOM_LEVEL
        for y in range(0, COL_SIZE+1, CELL_HEIGHT):
            
            Y_traslation = -y*ZOOM_LEVEL
            M = np.float32([[ZOOM_LEVEL, 0, Y_traslation], [0, ZOOM_LEVEL, X_traslation]])
            dst_image = cv2.warpAffine(warped, M, (cols, rows))
        
            filtered_image = filters_applier(dst_image)
            
            cleaned_image = cell_cleaner(filtered_image)

            predict = digits_rec(cleaned_image)
            riga_sud.append(predict)
        array_sudoku.append(riga_sud)
    print(array_sudoku)

def digits_rec(image_path):
    MODEL_IMAGE_SIZE = (28,28)
    image = cv2.resize(image_path, MODEL_IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    # Applica le trasformazioni all'immagine
    transform = transforms.Compose([transforms.ToTensor()])

    # Applica le trasformazioni e aggiunge una dimensione di batch
    image_tensor = transform(image).unsqueeze(0)

    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

device = choose_device()
image_path = select_file(IMAGE_PATH)

model = OurCNN()
model.load_state_dict(torch.load(DATABASE_PATH,torch.device(device)))
model.eval()

grid = Grid(GRID_PATH)

dst_points_drawer(grid.warped, grid.dstPoints)
zoomCells(grid.warped, grid.dstPoints)
