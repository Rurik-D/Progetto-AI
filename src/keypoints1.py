import cv2
import numpy as np
from tkinter import filedialog

def show_image(image, window_name="Image"):
    if image is None or image.shape[0] <= 0 or image.shape[1] <= 0:
        print("L'immagine non può essere visualizzata.")
    else:
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
def select_file(filepath=None):
    if filepath != None:
        return filepath
    else:
        filepath = filedialog.askopenfilename(title="Select the image")
        return filepath

def akaze(img, feedback=False):
    akaze = cv2.AKAZE_create(threshold=0.003, nOctaves=10)
    keypoints, descriptor = akaze.detectAndCompute(img, None)
    if feedback:
        display_keypoints(img, keypoints, descriptor)
    return keypoints, descriptor

def display_keypoints(img, keypoints, descriptor):
    cv2.drawKeypoints(img, keypoints, img, (51, 164, 200),
                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Keypoints of the Image", img)
    cv2.waitKey(0)

def warp_image(base_image, new_image):
    # Applica AKAZE per trovare keypoints e descrittori
    base_kpts, base_des = akaze(base_image)
    new_kpts, new_des = akaze(new_image)

    # Matcher dei descrittori
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(base_des, new_des, k=2)

    # Filtra i buoni match
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(good)

    # Disegna i buoni match
    good_matches = [[m] for m in good]
    final_image = cv2.drawMatchesKnn(base_image, base_kpts, new_image, new_kpts, good_matches, None, flags=2)

    # Estrai le posizioni dei buoni match
    src_pts = np.float32([base_kpts[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([new_kpts[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Trova l'omografia
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Verifica se la matrice di trasformazione è stata trovata correttamente
    if M is None:
        print("Impossibile trovare la matrice di trasformazione.")
        return None

    # Verifica e converte la matrice di trasformazione M
    if M.dtype != np.float32 and M.dtype != np.float64:
        M = M.astype(np.float32)

    if M.shape != (3, 3):
        raise ValueError("La matrice di trasformazione M non è 3x3")

    # Applica l'omografia per allineare le immagini
    h, w = base_image.shape[:2]
    result = cv2.warpPerspective(new_image, M, (w, h))

    # Combina le immagini
    final = cv2.addWeighted(base_image, 0.7, result, 0.3, 0)

    return final

if __name__ == "__main__":
    base_image = cv2.imread(select_file())
    new_image = cv2.imread(select_file())

    final_image = warp_image(base_image, new_image)
    show_image(final_image, window_name="Image")