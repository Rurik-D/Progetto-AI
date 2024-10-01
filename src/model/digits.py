from grid import Grid
from sdk_solver import *
from tkinter import filedialog

import cv2
import numpy as np

import torch
from torch import nn
from torchvision import transforms

DATABASE_PATH = 'C:\\Users\\giuse\\Desktop\\Progetto-AI\\src\\model\\ai_models\\digits_rec(v2).pth'
IMAGE_PATH = 'C:\\Users\\giuse\\Desktop\\Progetto-AI\\Images\\Sudoku\\_53_9638115.jpeg'
GRID_PATH = 'C:\\Users\\giuse\\Desktop\\Progetto-AI\\Images\\Sudoku\\_53_9638115.jpeg'

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
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(9,9))
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


def clean_board(warped):
    rows, cols = warped.shape[:2]
    CELL_LENGTH = rows//9
    CELL_HEIGHT = cols//9

    # Invert the image
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    inverted_image = cv2.bitwise_not(warped)

    # Apply adaptive thresholding to make the grid lines more pronounced
    thresh = cv2.adaptiveThreshold(inverted_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_img = cv2.dilate(thresh, kernel, iterations=1)


    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (CELL_LENGTH-1, 1))
    detect_horizontal = cv2.morphologyEx(dilated_img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, CELL_HEIGHT-1))
    detect_vertical = cv2.morphologyEx(dilated_img, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Combine the horizontal and vertical lines
    grid_lines = cv2.add(detect_horizontal, detect_vertical)
    ##cv2.imshow("grif_lines", grid_lines)
    ##cv2.waitKey(0)
    # Invert grid lines to create a mask
    mask = cv2.bitwise_not(grid_lines)
    ##cv2.imshow("mask", mask)
    ##cv2.waitKey(0)

    # Use the mask to remove lines from the original inverted image
    result = cv2.bitwise_and(inverted_image, mask)

    # Apply additional morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    cleaned_result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    # Invert the result back to original grayscale
    final_result = cv2.bitwise_not(cleaned_result)

    # Optionally, apply blurring to further smooth the result
    final_result = cv2.medianBlur(final_result, 3)

    return final_result


def zoomCells(warped, dst_points):
    # cv2.imshow("wp", warped)
    # cv2.waitKey(0)
    rows, cols = warped.shape[:2]
    cleaned_image = clean_board(warped)
    # cv2.imshow("clean warped", cleaned_image)
    # cv2.waitKey(0)
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
            dst_image = cv2.warpAffine(cleaned_image, M, (cols, rows))
            # cv2.imshow("dst", dst_image)
            # cv2.waitKey(0)
            filtered_image = filters_applier(dst_image)
            #cleaned_image = cell_cleaner(filtered_image)
            # cv2.imshow("filtered_image", filtered_image)
            # cv2.waitKey(0)
            predict = digits_rec(filtered_image)
            riga_sud.append(predict)
        array_sudoku.append(riga_sud)
    return array_sudoku

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

model = OurCNN().to(device)
model.load_state_dict(torch.load(DATABASE_PATH,torch.device(device)))
model.eval()

grid = Grid(GRID_PATH)

dst_points_drawer(grid.warped, grid.dstPoints)

def print_sudoku():
    void_grid = cv2.imread("C:\\Users\\giuse\\Desktop\\Progetto-AI\\Images\\void_grid.png")
    sudoku = zoomCells(grid.warped, grid.dstPoints)
    
    board = np.array(sudoku)
     
    if solve_sudoku(board):
        current_point = [5,5]
        for riga in range(9):
            for numero in range(9): 
                text_point = (current_point[0] + 7, current_point[1] + 30) 
                if sudoku[riga][numero]==board[riga][numero]:
                    #printa nero
                    
                    cv2.putText(void_grid, str(board[riga][numero]), text_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    
                else:
                    #printa verde
                    cv2.putText(void_grid, str(board[riga][numero]), text_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                current_point[0] += 39
            current_point[1] += 39
            current_point[0] = 5
    else:
        print("Nessuna soluzione trovata.")      
    cv2.imshow("griglia temp", void_grid)
    cv2.waitKey(0)
    return void_grid
print_sudoku()