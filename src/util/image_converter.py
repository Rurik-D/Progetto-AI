import cv2
from PIL import Image


def cv2_to_pil_image(cv2_image:str) -> Image:
    cv2_image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv2_image_rgb)
    return pil_image
