import cv2

from tkinter import filedialog

def sudoku_filter(img):
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    eq_img = clahe.apply(img)

    img = cv2.threshold(eq_img, 120, 255, cv2.THRESH_BINARY)

    return img

def display_keypoints(img, keypoints, descriptor, feedback=False):
    if feedback:
        cv2.drawKeypoints(img, keypoints, img, (51, 164, 200),
                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        cv2.imshow("Keypoints of the Image", img)
        cv2.waitKey(0)

def akaze(img):
    akaze = cv2.AKAZE.create(threshold=0.003, nOctaves=10)
    keypoints, descriptor = akaze.detectAndCompute(img, None)
    display_keypoints(img, keypoints, descriptor, True)
    

def select_file(filepath=None):
    if filepath != None:
        return filepath
    else:
        filepath = filedialog.askopenfilename(title="Select the image")
        return filepath

image_fixed = cv2.imread(select_file("C:\\Users\\giuse\\Desktop\\Progetto-AI\\aug\\_6_2331578.jpeg"))

akaze(image_fixed)
#new_image = cv2.imread(select_file(), 0)

new_image = cv2.imread(select_file(), 0)
filtered_img = sudoku_filter(new_image)
akaze(filtered_img)
