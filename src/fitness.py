from PIL import Image
from torch import nn
import cv2
import numpy as np

class Fitness:
    def __init__(self, vlm, img_cle, c_gt, c_tar="A dog runing in the grass."):
        self.vlm = vlm
        self.c_groundtruth = self.vlm.text_encode(c_gt)
        self.c_target = self.vlm.text_encode(c_tar)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.img_cle = img_cle
    
    
    def apply_gamma_correction(self, img, gamma):
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(img, table)

    def apply_gaussian_blur(self, img, sigma, ksize=3):
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)
        
    def apply_affine_transform(self, params):
        # """
        #     s_x, s_y \in [0.8, 1.5]
        #     shx, shy \in [-0.2, 0.2]
        #     theta \in [-np.pi/6, np.pi/6]
        # """
        [sx, sy, shx, shy, theta, sigma, gamma] = params.tolist()
        
        h, w = self.img_cle.shape[:2]
        
        center_x, center_y = w / 2, h / 2

        affine_matrix = np.array([
            [sx * np.cos(theta) + shx * np.sin(theta), -sy * np.sin(theta) + shx * np.cos(theta), (1 - (sx * np.cos(theta) + shx * np.sin(theta))) * center_x + (sy * np.sin(theta) - shx * np.cos(theta)) * center_y],
            [sx * np.sin(theta) + shy * np.cos(theta),  sy * np.cos(theta) + shy * np.sin(theta), (1 - (sx * np.sin(theta) + shy * np.cos(theta))) * center_x + (-sy * np.cos(theta) - shy * np.sin(theta)) * center_y]
        ], dtype=np.float32)

        transformed_img = cv2.warpAffine(self.img_cle, affine_matrix, (w, h), flags=cv2.INTER_LINEAR)
        transformed_img = self.apply_gamma_correction(transformed_img, gamma)
        transformed_img = self.apply_gaussian_blur(transformed_img, sigma)
        return transformed_img   

    def IG_IT_fitness(self, X):
        imgs = [Image.fromarray(self.apply_affine_transform(x)) for x in X]
        img_adv_features = self.vlm.image_encode(imgs)
        IG_cos = self.cos(img_adv_features, self.c_groundtruth) # min
        IT_cos = self.cos(img_adv_features, self.c_target) # max
        fitness = IG_cos - IT_cos
        return fitness.cpu().numpy()

    
    

            


        