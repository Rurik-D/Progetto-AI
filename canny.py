import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carica l'immagine
img = cv2.imread('C:\\Users\\giuse\\Desktop\\Progetto-AI\\aug\\_0_9607135.jpeg', cv2.IMREAD_GRAYSCALE)

# Verifica se l'immagine è stata caricata correttamente
if img is None:
    raise ValueError("Immagine non trovata o il percorso è errato")

# Applica una leggera sfocatura gaussiana per ridurre il rumore
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Applica una soglia binaria per migliorare il contrasto
_, thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

# Rileva i bordi utilizzando Canny
edges = cv2.Canny(thresholded, 50, 150)

# Trova i contorni nell'immagine
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Trova il contorno più grande
sudoku_contour = None
max_area = 0
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 1000:  # Filtra i contorni più piccoli
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and area > max_area:  # Cerca il contorno con quattro lati e area massima
            sudoku_contour = approx
            max_area = area

# Verifica se è stato trovato un contorno valido
if sudoku_contour is not None:
    points = sudoku_contour.reshape(4, 2)
    
    # Ordina i punti in ordine (top-left, top-right, bottom-right, bottom-left)
    points = sorted(points, key=lambda x: x[0])
    top_left, bottom_left = sorted(points[:2], key=lambda x: x[1])
    top_right, bottom_right = sorted(points[2:], key=lambda x: x[1])
    rect = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

    # Coordinate della griglia del Sudoku rettificata
    side = max(
        np.linalg.norm(top_right - top_left),
        np.linalg.norm(bottom_right - bottom_left),
        np.linalg.norm(bottom_right - top_right),
        np.linalg.norm(bottom_left - top_left)
    )
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

    # Calcola la trasformazione prospettica
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (int(side), int(side)))

    # Visualizza l'immagine rettificata
    plt.imshow(warped, cmap='gray')
    plt.title('Warped Sudoku Grid')
    plt.axis('off')
    plt.show()

    cv2.imshow("Original Image", img)
    cv2.imshow("Warped Image", warped)
else:
    print("Contorno del Sudoku non trovato")

cv2.waitKey(0)
cv2.destroyAllWindows()