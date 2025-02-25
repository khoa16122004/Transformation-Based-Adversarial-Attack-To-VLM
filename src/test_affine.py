import numpy as np
import cv2


def apply_gamma_correction(img, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)

def apply_gaussian_blur(img, sigma, ksize=3):
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

def apply_affine_transform(img, params):
    # """
    #     s_x, s_y \in [0.8, 1.5]
    #     shx, shy \in [-0.2, 0.2]
    #     theta \in [-np.pi/6, np.pi/6]
    # """
    h, w = img.shape[:2]
    center_x, center_y = w / 2, h / 2
    [sx, sy, shx, shy, theta, sigma, gamma] = params
    affine_matrix = np.array([
        [sx * np.cos(theta) + shx * np.sin(theta), -sy * np.sin(theta) + shx * np.cos(theta), (1 - (sx * np.cos(theta) + shx * np.sin(theta))) * center_x + (sy * np.sin(theta) - shx * np.cos(theta)) * center_y],
        [sx * np.sin(theta) + shy * np.cos(theta),  sy * np.cos(theta) + shy * np.sin(theta), (1 - (sx * np.sin(theta) + shy * np.cos(theta))) * center_x + (-sy * np.cos(theta) - shy * np.sin(theta)) * center_y]
    ], dtype=np.float32)

    transformed_img = cv2.warpAffine(img, affine_matrix, (w, h), flags=cv2.INTER_LINEAR)
    transformed_img = apply_gamma_correction(transformed_img, gamma)
    transformed_img = apply_gaussian_blur(transformed_img, sigma)
    
    return transformed_img

img = cv2.resize(cv2.imread('images/lionsea.jpg'), (224, 224))
cv2.imwrite("original.jpg", img)  
params =  [ 0.81228038 , 0.82101978 , 0,  0 ,       0, 0.25, 1.5]
transformed_img = apply_affine_transform(img, params)

cv2.imwrite("output.jpg", transformed_img)  
