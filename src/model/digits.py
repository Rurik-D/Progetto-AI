from model_trainer.digits_model import OurCNN
from sdk_solver import solve_sudoku
from torchvision import transforms
from os.path import abspath
import numpy as np
import cv2
import torch
import time

DATABASE_PATH = abspath(".") + f"\\src\\model\\model_trainer\\digits_model.pth"

def choose_device(feedback=False):
    device = ("cuda" if torch.cuda.is_available()
              else "mps"
              if torch.backends.mps.is_available()
              else "cpu")
    
    if feedback:
        print("Device in use:", device)
    
    return device
    
device = choose_device()
model = OurCNN().to(device)
model.load_state_dict(torch.load(DATABASE_PATH,torch.device(device), weights_only=False))
model.eval()


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

# def filter_test(*args):                                                          TODO PER IMMAGINI RELAZIONE
#     for img in args:
#         cv2.imshow(f"{img}", img)
#         cv2.waitKey(0)

def filters_applier(raw_image):
    IMAGE_DEFAULT_SIZE = (200, 200)
    gray_image = BGR2GRAY_selective(raw_image)
    equalized_image = clahe_equalizer(gray_image)
    thrs_image = cv2.threshold(equalized_image, 90, 255, cv2.THRESH_BINARY)
    preprocessed_image = cv2.bitwise_not(thrs_image[1])
    preprocessed_image = cv2.dilate(preprocessed_image, np.ones((7, 7), np.uint8), iterations=2)
    resized_image = cv2.resize(preprocessed_image, IMAGE_DEFAULT_SIZE, interpolation=cv2.INTER_LINEAR)
    #filter_test(raw_image, gray_image, equalized_image, thrs_image[1], preprocessed_image, resized_image)
    return resized_image

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
    detect_horizontal = cv2.morphologyEx(dilated_img, cv2.MORPH_OPEN, horizontal_kernel, iterations=3)

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, CELL_HEIGHT-1))
    detect_vertical = cv2.morphologyEx(dilated_img, cv2.MORPH_OPEN, vertical_kernel, iterations=3)

    # Combine the horizontal and vertical lines
    grid_lines = cv2.add(detect_horizontal, detect_vertical)

    # Invert grid lines to create a mask
    mask = cv2.bitwise_not(grid_lines)

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


def zoomCells(warped):
    rows, cols = warped.shape[:2]
    cleaned_image = clean_board(warped)
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
            filtered_image = filters_applier(dst_image)
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


def draw_canvas():
    SUDOKU_SIZE = 720
    canvas = np.zeros((SUDOKU_SIZE, SUDOKU_SIZE, 3), dtype='uint8')
    white_canvas = cv2.bitwise_not(canvas)
    return white_canvas

def draw_empty_grid(partial_grid, grid_frames=1, margin=5):
    SUDOKU_SIZE = 720
    BLACK = (0, 0, 0)
    CELLS_PER_SIDE = 9
    if grid_frames>CELLS_PER_SIDE:
        empty_grid = partial_grid
        return empty_grid
    else:
        cell_size=SUDOKU_SIZE//grid_frames
        for row in range(0, SUDOKU_SIZE+1, cell_size):
            for col in range(cell_size, SUDOKU_SIZE+1, cell_size):
                row_coordinates = (row, row)
                col_coordinates = (col, col)
                cv2.rectangle(partial_grid, row_coordinates, col_coordinates, BLACK, margin)
        return draw_empty_grid(partial_grid, grid_frames*3, margin//2)

def color_selector(preset_digit):
    GREEN = (12, 92, 13)
    BLACK = (0, 0, 0)
    EMPTY_CELL = int(0)
    if preset_digit == EMPTY_CELL:
        return GREEN
    else:
        return BLACK

def fill_sudoku(empty_grid, solved_sudoku, unsolved_sudoku):

    START_OFFSET = 4
    CELLS_PER_SIDE = 9
    HOR_OFFSET = 16
    VER_OFFSET = 62
    CELLS_DISTANCE = 80
    
    current_point = [START_OFFSET, START_OFFSET]
    for row in range(CELLS_PER_SIDE):
        for number in range(CELLS_PER_SIDE):
            preset_digit = unsolved_sudoku[row][number]
            solution_digit = str(solved_sudoku[row][number])
            text_point = current_point[0] + HOR_OFFSET, current_point[1] + VER_OFFSET
            cv2.putText(empty_grid, solution_digit, text_point, cv2.FONT_HERSHEY_SIMPLEX,
                    2, color_selector(preset_digit), 3)
            current_point[0] += CELLS_DISTANCE
        current_point[1] += CELLS_DISTANCE
        current_point[0] = START_OFFSET


def get_solved_sudoku(grid):
    try:
        empty_grid = draw_empty_grid(draw_canvas())
        sudoku = zoomCells(grid.warped)
        unsolved_sudoku = np.array(sudoku)
        solved_sudoku = unsolved_sudoku.copy()

        if solve_sudoku(solved_sudoku, time.time()):
            fill_sudoku(empty_grid, solved_sudoku, unsolved_sudoku)
            
            if not is_valid(solved_sudoku):
                return (None, False)
        else:
            return (None, False)
    except:
        return (None, False)
    
    # solved_sudoku = empty_grid # SCOMMENTARE PER TEST!!!!!!!!!!!!!
    return (solved_sudoku, True)

def is_valid(solved):
    rowError = any(np.unique(row).size != row.size for row in solved)
    colError = any(np.unique(col).size != col.size for col in solved.T)
    return not rowError and not colError







