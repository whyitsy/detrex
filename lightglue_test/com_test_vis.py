import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d

def align_with_lightglue(ref_img_path, target_img_path, feature_extractor='superpoint'):
    """使用LightGlue对齐图像"""
    # 读取图像
    image0 = load_image(ref_img_path)
    image1 = load_image(target_img_path)
    
    # 选择特征提取器
    extractor = SuperPoint(max_num_keypoints=2048).eval() if feature_extractor == 'superpoint' else DISK(max_num_keypoints=2048).eval()
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
        
        kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    
    # 转换为numpy数组
    points0 = m_kpts0.cpu().numpy()
    points1 = m_kpts1.cpu().numpy()
    
    # 计算单应性矩阵
    H, mask = cv2.findHomography(points1, points0, cv2.RANSAC, 5.0)
    
    return H, points0, points1

def align_with_sift(ref_img_path, target_img_path):
    """使用SIFT对齐图像"""
    # 读取图像
    img1 = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
    
    # 初始化SIFT检测器
    sift = cv2.SIFT_create()
    
    # 检测关键点和描述符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # 使用FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # 筛选好的匹配点
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    
    # 获取匹配点坐标
    points1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    points2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
    # 计算单应性矩阵
    H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
    
    return H, points1.reshape(-1, 2), points2.reshape(-1, 2)

def visualize_alignment(ref_img_path, target_img_path, method='lightglue'):
    """可视化对齐效果"""
    # 读取图像
    ref_img = cv2.imread(ref_img_path)
    target_img = cv2.imread(target_img_path)
    
    # 根据选择的方法获取单应性矩阵和匹配点
    if method == 'lightglue':
        H, ref_points, target_points = align_with_lightglue(ref_img_path, target_img_path)
        method_name = 'LightGlue'
    else:
        H, ref_points, target_points = align_with_sift(ref_img_path, target_img_path)
        method_name = 'SIFT'
    
    # 应用单应性矩阵变换目标点
    transformed_points = cv2.perspectiveTransform(
        target_points.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
    
    # 绘制结果
    plt.figure(figsize=(15, 10))
    
    # 显示参考图像和点
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
    plt.scatter(ref_points[:, 0], ref_points[:, 1], c='red', s=10, label='Reference Points')
    plt.title(f'Reference Image with {method_name} Points')
    plt.legend()
    
    # 显示目标图像和点
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
    plt.scatter(target_points[:, 0], target_points[:, 1], c='blue', s=10, label='Target Points')
    plt.title(f'Target Image with {method_name} Points')
    plt.legend()
    
    # 显示对齐后的点与参考点的比较
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
    plt.scatter(ref_points[:, 0], ref_points[:, 1], c='red', s=10, label='Reference Points')
    plt.scatter(transformed_points[:, 0], transformed_points[:, 1], c='green', s=10, label='Aligned Points')
    plt.title(f'Aligned Points vs Reference Points ({method_name})')
    plt.legend()
    
    # 计算并显示误差向量
    plt.subplot(2, 2, 4)
    errors = transformed_points - ref_points
    plt.quiver(ref_points[:, 0], ref_points[:, 1], 
               errors[:, 0], errors[:, 1], 
               angles='xy', scale_units='xy', scale=1, color='purple', width=0.002)
    plt.imshow(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Alignment Error Vectors ({method_name})')
    
    plt.tight_layout()
    plt.show()
    
    # 计算并打印平均误差
    distances = np.linalg.norm(errors, axis=1)
    print(f"{method_name} Average Alignment Error: {np.mean(distances):.2f} pixels")
    print(f"{method_name} Median Alignment Error: {np.median(distances):.2f} pixels")

def compare_methods(ref_img_path, target_img_path):
    """比较两种方法的对齐效果"""
    # 读取图像
    ref_img = cv2.imread(ref_img_path)
    target_img = cv2.imread(target_img_path)
    
    # 获取两种方法的结果
    H_lightglue, ref_points_lg, target_points_lg = align_with_lightglue(ref_img_path, target_img_path)
    H_sift, ref_points_sift, target_points_sift = align_with_sift(ref_img_path, target_img_path)
    
    # 转换点
    transformed_points_lg = cv2.perspectiveTransform(
        target_points_lg.reshape(-1, 1, 2).astype(np.float32), H_lightglue).reshape(-1, 2)
    transformed_points_sift = cv2.perspectiveTransform(
        target_points_sift.reshape(-1, 1, 2).astype(np.float32), H_sift).reshape(-1, 2)
    
    # 计算误差
    errors_lg = transformed_points_lg - ref_points_lg
    errors_sift = transformed_points_sift - ref_points_sift
    
    distances_lg = np.linalg.norm(errors_lg, axis=1)
    distances_sift = np.linalg.norm(errors_sift, axis=1)
    
    # 绘制比较图
    plt.figure(figsize=(15, 6))
    
    # 误差分布直方图
    plt.subplot(1, 2, 1)
    plt.hist(distances_lg, bins=50, alpha=0.5, label='LightGlue')
    plt.hist(distances_sift, bins=50, alpha=0.5, label='SIFT')
    plt.xlabel('Alignment Error (pixels)')
    plt.ylabel('Frequency')
    plt.title('Alignment Error Distribution')
    plt.legend()
    
    # 箱线图比较
    plt.subplot(1, 2, 2)
    plt.boxplot([distances_lg, distances_sift], labels=['LightGlue', 'SIFT'])
    plt.ylabel('Alignment Error (pixels)')
    plt.title('Alignment Error Comparison')
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    print("\n=== LightGlue ===")
    print(f"Average Error: {np.mean(distances_lg):.2f} pixels")
    print(f"Median Error: {np.median(distances_lg):.2f} pixels")
    print(f"Max Error: {np.max(distances_lg):.2f} pixels")
    
    print("\n=== SIFT ===")
    print(f"Average Error: {np.mean(distances_sift):.2f} pixels")
    print(f"Median Error: {np.median(distances_sift):.2f} pixels")
    print(f"Max Error: {np.max(distances_sift):.2f} pixels")

# 使用示例
if __name__ == "__main__":
    # 图像路径
    ref_img_path = '/home/kky/detrex/lightglue_test/data/after_frame_000348_angle_5.00.png'  # 请替换为你的参考图像路径
    target_img_path = '/home/kky/detrex/lightglue_test/data/before_frame_000167_angle_25.05.png'  # 请替换为你的目标图像路径
    
    # 可视化LightGlue对齐效果
    print("Visualizing LightGlue alignment...")
    visualize_alignment(ref_img_path, target_img_path, method='lightglue')
    
    # 可视化SIFT对齐效果
    print("\nVisualizing SIFT alignment...")
    visualize_alignment(ref_img_path, target_img_path, method='sift')
    
    # 比较两种方法
    print("\nComparing methods...")
    compare_methods(ref_img_path, target_img_path)