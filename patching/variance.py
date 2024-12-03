import cv2
import numpy as np
import os

def save_patch(p, num, out_dir='dark_patches'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(os.path.join(out_dir, f"patch_{num}.png"), p)
def crop_patches(img, p_size=(128, 128), out_dir='dark_patches'):
    h, w = img.shape
    ph, pw = p_size
    num = 0
    total_intensity = 0
    total_patches = 0

    for y in range(0, h, ph):
        for x in range(0, w, pw):
            p = img[y:y + ph, x:x + pw]
            if p.shape[0] != ph or p.shape[1] != pw:
                continue
            total_intensity += np.mean(p)
            total_patches += 1
            #calculating threshold 
    avg_intensity = total_intensity / total_patches if total_patches > 0 else 0

    for y in range(0, h, ph):
        for x in range(0, w, pw):
            p = img[y:y + ph, x:x + pw]
            if p.shape[0] != ph or p.shape[1] != pw:
                continue
            m = np.mean(p)
            if m < avg_intensity:
                save_patch(p, num, out_dir)
                num += 1


img_path = 'A-global-magnetogram-of-the-Sun-with-white-and-back-showing-regions-where-the-magnetic.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (512, 512))
crop_patches(img, p_size=(128, 128), out_dir='dark_patches')
