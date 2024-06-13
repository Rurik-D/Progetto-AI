import cv2
from tkinter import filedialog
import numpy as np

def sudoku_filter(img):
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    
    img = cv2.addWeighted(img, 0.9, img, 0.5, 0)
    
    img = cv2.medianBlur(img, 3)
    
    trs = cv2.adaptiveThreshold(img, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    
    img = cv2.cvtColor(trs, cv2.COLOR_GRAY2BGR)
    return img

def sudoku_filter_v2(img):
    img_grey = img
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_grey = cv2.medianBlur(img_grey, 5)
    edges = cv2.Laplacian(img_grey, cv2.CV_8U, ksize=5)
    
    ret, thresholded = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)
    
    color_img = cv2.bilateralFilter(img, 10, 250, 250)
    skt = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
    output = cv2.bitwise_and(color_img, skt)
    return output

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

base_image =  sudoku_filter(cv2.imread(select_file()))
new_image = sudoku_filter(cv2.imread(select_file()))
new_image = cv2.resize(new_image, (600, 600), interpolation=cv2.INTER_LINEAR)
base_image = cv2.resize(base_image, (600, 600), interpolation=cv2.INTER_LINEAR)

# let's create a feature extractor
orb = cv2.ORB_create()

# compute the features in both images
kpt1, desc1 = orb.detectAndCompute(base_image,None)
kpt2, desc2 = orb.detectAndCompute(new_image,None)

# create a matcher and match the keypoints
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = matcher.knnMatch(desc1,desc2,k=2)

# perform the ratio test:
# a match is correct if the
# ratio betwee the two closest points of a match
# is below a certain threshold
good_matches = []
for m,n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# check if we have at least 4 points
# REMEMBER: the two images are called queryImg and trainingImg
# so queryIdx is a point belonging to query Img,
# while trainingIdx is a point belonging to trainingImg
# In our case, queryImg is the left image, while trainingImg is the right image
if len(good_matches) > 4:
    # convert the points to float32
    src_points = np.float32([kpt1[m.queryIdx].pt for m in good_matches])
    dst_points = np.float32([kpt2[m.trainIdx].pt for m in good_matches])

    # compute the homography matrix
    M, mask = cv2.findHomography(src_points,dst_points)

    # transform the left image and stitch it together with the
    # right image
    dst = cv2.warpPerspective(base_image,M,(base_image.shape[1] + new_image.shape[1], base_image.shape[0]))
    dst[0:new_image.shape[0],0:new_image.shape[1]] = new_image.copy()

    cv2.namedWindow('Panorama',cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Panorama',dst)
    cv2.waitKey(0)
else:
    print("Not enough points")

