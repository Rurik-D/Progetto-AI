import cv2

from tkinter import filedialog
import numpy as np
def sudoku_filter(img):
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    eq_img = clahe.apply(img)

    img = cv2.threshold(eq_img, 120, 255, cv2.THRESH_BINARY)

    return img

def display_keypoints(img, keypoints, descriptor):
    cv2.drawKeypoints(img, keypoints, img, (51, 164, 200),
                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imshow("Keypoints of the Image", img)
    cv2.waitKey(0)

def akaze(img, feedback=False):
    akaze = cv2.AKAZE.create(threshold=0.003, nOctaves=10)
    keypoints, descriptor = akaze.detectAndCompute(img, None)
    if feedback:
        display_keypoints(img, keypoints, descriptor)
    return keypoints, descriptor

def select_file(filepath=None):
    if filepath != None:
        return filepath
    else:
        filepath = filedialog.askopenfilename(title="Select the image")
        return filepath

base_image = cv2.imread(select_file())
new_image = cv2.imread(select_file())

#new_image = sudoku_filter(new_image)

base_kpts, base_des = akaze(base_image)
new_kpts, new_des = akaze(new_image, True)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(base_des, new_des, k=2)

good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)
print(good)



# Crea una nuova lista di liste di match
good_matches = [[m] for m in good]

# Disegna i "buoni" match
final_image = cv2.drawMatchesKnn(base_image, base_kpts, new_image, new_kpts, good_matches,None,flags=2)

# Estrai le posizioni dei buoni match
src_pts = np.float32([ base_kpts[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ new_kpts[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

# Trova l'omografia
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

# Ottieni le dimensioni dell'immagine di riferimento
h,w = base_image.shape[:2]

# Usa l'omografia per allineare le immagini
result = cv2.warpPerspective(new_image, M, (w, h))

# Combina le immagini
final = cv2.addWeighted(base_image, 0.7, result, 0.3, 0)

cv2.imshow("Keypoints of the Image", final)
cv2.waitKey(0)