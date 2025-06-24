import numpy as np
from detectron2.data import transforms as T

# 定义高斯噪声Transform
class AddGaussianNoise(T.Transform):
    def __init__(self, mean=0, std_range=(10, 50), p=0.5):
        """
        添加高斯噪声
        
        参数:
            mean: 噪声均值
            std_range: 噪声标准差范围，随机采样
            p: 应用噪声的概率
        """
        super().__init__()
        self.mean = mean
        self.std_range = std_range
        self.p = p
        
    def _add_noise(self, img):
        std = np.random.uniform(*self.std_range)
        noise = np.random.normal(self.mean, std, img.shape).astype(np.uint8)
        noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)
        return noisy_img
    
    def apply_image(self, img):
        if np.random.random() < self.p:
            return self._add_noise(img)
        return img
    
    def apply_coords(self, coords):
        return coords  # 噪声不影响坐标