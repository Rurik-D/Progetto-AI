import cv2
import numpy as np
def applyFilters(image):
    """
        Applies following filters:
            - gray scale
            - median blur
            - adaptive thresholding (get a binary image)
            - reverses the image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    medianBlur = cv2.medianBlur(blurred,5)
    th = cv2.adaptiveThreshold(medianBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                            cv2.THRESH_BINARY, 11, 2)
    _, binary = cv2.threshold(th, 128, 255, cv2.THRESH_BINARY_INV)

    return binary

def offset(image, x, y):
    BLACK = 0
    WHITE = 1
    y1 = y + 10
    off = 0
    if image[x][y1] == WHITE:
        return 0
    while image[x][y1] == BLACK:
        y1 += 1
        off += 1
    return off
    

# 1. Carica l'immagine
image = cv2.imread('C:\\Users\\giuse\\Desktop\\Progetto-AI\\Images\\digit-sudoku\\sd3.png')

img2 = applyFilters(image)
# 4. Trova i contorni
contours, _ = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 5. Trova il contorno più grande (che dovrebbe essere il Sudoku)
largest_contour = max(contours, key=cv2.contourArea)

# 6. Approssima il contorno con un poligono con 4 lati
epsilon = 0.02 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)

# 7. Controlla se l'approssimazione ha 4 punti
if len(approx) == 4:
    # 8. Ordina gli angoli
    pts = approx.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Converti a interi
    rect = rect.astype(int)

    # 9. Disegna i punti trovati sull'immagine originale
    for x, y in rect:
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

    orLine = rect[1][0] - rect[0][0]
    cella = orLine // 9

    x = rect[0][0]
    off = offset(image, rect[0][0], rect[0][1])
    for i in range(0, 10):
    
        cv2.circle(image, (x , rect[0][1]), 5, (0, 255, 0), -1)
        x += cella
    # Mostra l'immagine con gli angoli trovati
    cv2.imshow("Sudoku Angoli", image)
    cv2.waitKey(0)
else:
    print("Il contorno non ha 4 lati")

cv2.destroyAllWindows()
