import glob
import cv2
import numpy as np
import os

def create_soft_mask(image_path, 
                      thresh, 
                      blur_size=21,
                      morph_size=5):
    
    img = cv2.imread(image_path, 0)
    
    _, binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
    softed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    blurred = cv2.GaussianBlur(softed, (blur_size, blur_size), 0)
    
    return cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


os.makedirs("output_fg", exist_ok=True)
os.makedirs("output_light", exist_ok=True)
os.makedirs("output_bg", exist_ok=True)

for img_path in glob.glob("images/*"):
    print(f"Processing: {img_path}")
    
    light_mask = create_soft_mask(img_path, thresh=250, blur_size=31, morph_size=7)
    fg_mask = create_soft_mask(img_path, thresh=100, blur_size=51, morph_size=9)
    
    img = cv2.imread(img_path, 0)
    _, bg_mask = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY_INV)
    
    filename = os.path.basename(img_path)
    cv2.imwrite(f"output_fg/{filename}", fg_mask)
    cv2.imwrite(f"output_light/{filename}", light_mask)
    cv2.imwrite(f"output_bg/{filename}", bg_mask)