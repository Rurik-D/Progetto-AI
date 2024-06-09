import cv2

from tkinter import filedialog

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

new_image = sudoku_filter(new_image)

base_kpts, base_des = akaze(base_image)
new_kpts, new_des = akaze(new_image)

bf = cv2.BFMatcher()
matches = bf.knnMatch(base_des, new_des, k=2)

good = []
for m,n in matches:
    if m.distance < 0.75*n.distance: #0.75 valore arbitrario
        good.append([m])
