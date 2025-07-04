import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d

def align_images_lightglue(img1_path, img2_path, feature_extractor='superpoint'):
    """
    使用LightGlue对齐两张图像
    
    参数:
        img1_path: 参考图像路径
        img2_path: 待对齐图像路径
        feature_extractor: 特征提取器，可选'superpoint'或'disk'
    
    返回:
        aligned_img: 对齐后的图像
        H: 单应性矩阵
    """
    # 读取图像
    image0 = load_image(img1_path)
    image1 = load_image(img2_path)
    
    # 选择特征提取器
    if feature_extractor == 'superpoint':
        extractor = SuperPoint(max_num_keypoints=2048).eval()  # 加载SuperPoint特征提取器
    else:
        extractor = DISK(max_num_keypoints=2048).eval()  # 加载DISK特征提取器
    
    # 加载LightGlue匹配器
    matcher = LightGlue(features=feature_extractor).eval()
    
    # 使用GPU如果可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = extractor.to(device)
    matcher = matcher.to(device)
    
    # 提取特征
    with torch.no_grad():
        feats0 = extractor.extract(image0.to(device))
        feats1 = extractor.extract(image1.to(device))
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # 移除批次维度
        
        # 获取匹配点
        kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    
    # 转换为numpy数组
    points0 = m_kpts0.cpu().numpy()
    points1 = m_kpts1.cpu().numpy()
    
    # 使用RANSAC计算单应性矩阵
    H, mask = cv2.findHomography(points1, points0, cv2.RANSAC, 5.0)
    
    # 读取原始图像用于对齐
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    height, width = img1.shape[:2]
    
    # 应用单应性矩阵进行图像对齐
    aligned_img = cv2.warpPerspective(img2, H, (width, height))
    
    return aligned_img, H

def visualize_matches(img1_path, img2_path, feature_extractor='superpoint'):
    """
    可视化匹配的关键点
    
    参数:
        img1_path: 参考图像路径
        img2_path: 待对齐图像路径
        feature_extractor: 特征提取器，可选'superpoint'或'disk'
    """
    # 读取图像
    image0 = load_image(img1_path)
    image1 = load_image(img2_path)
    
    # 选择特征提取器
    if feature_extractor == 'superpoint':
        extractor = SuperPoint(max_num_keypoints=2048).eval()
    else:
        extractor = DISK(max_num_keypoints=2048).eval()
    
    # 加载LightGlue匹配器
    matcher = LightGlue(features=feature_extractor).eval()
    
    # 使用GPU如果可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = extractor.to(device)
    matcher = matcher.to(device)
    
    # 提取特征并匹配
    with torch.no_grad():
        feats0 = extractor.extract(image0.to(device))
        feats1 = extractor.extract(image1.to(device))
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    
    # 可视化匹配结果
    kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    
    axes = viz2d.plot_images([image0, image1])
    viz2d.plot_matches(m_kpts0, m_kpts1, color='lime', lw=0.2)
    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
    
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 图像路径
    ref_img_path = '/home/kky/detrex/detrex/testdata/after_frame_000348_angle_5.00.png'  # 参考图像
    target_img_path = '/home/kky/detrex/detrex/testdata/before_frame_000167_angle_25.05.png'  # 待对齐图像
    
    # 可视化匹配点
    print("Visualizing matches...")
    visualize_matches(ref_img_path, target_img_path)
    
    # 对齐图像
    print("Aligning images...")
    aligned_img, H = align_images_lightglue(ref_img_path, target_img_path)
    
    # 保存结果
    cv2.imwrite('aligned_image.jpg', aligned_img)
    print(f"Homography matrix:\n{H}")
    print("Aligned image saved as 'aligned_image.jpg'")
    
    # 显示结果
    ref_img = cv2.imread(ref_img_path)
    target_img = cv2.imread(target_img_path)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.title('Reference Image')
    plt.imshow(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(132)
    plt.title('Target Image')
    plt.imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(133)
    plt.title('Aligned Image')
    plt.imshow(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.show()