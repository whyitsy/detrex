import numpy as np
import torch
from detectron2 import transforms as T
from detectron2.data import transforms as T


# 定义椒盐噪声Transform
class AddSaltPepperNoise(T.Transform):
    def __init__(self, salt_prob=0.02, pepper_prob=0.02, p=0.5):
        """
        添加椒盐噪声
        
        参数:
            salt_prob: 盐噪声(白色)概率
            pepper_prob: 椒噪声(黑色)概率
            p: 应用噪声的概率
        """
        super().__init__()
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob
        self.p = p
        
    def _add_noise(self, img):
        h, w, c = img.shape
        noisy_img = img.copy()
        
        # 盐噪声(白色)
        salt_mask = np.random.random((h, w)) < self.salt_prob
        noisy_img[salt_mask] = 255
        
        # 椒噪声(黑色)
        pepper_mask = np.random.random((h, w)) < self.pepper_prob
        noisy_img[pepper_mask] = 0
        
        return noisy_img
    
    def apply_image(self, img):
        if np.random.random() < self.p:
            return self._add_noise(img)
        return img
    
    def apply_coords(self, coords):
        return coords  # 噪声不影响坐标