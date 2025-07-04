import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd

def create_output_folder():
    """创建输出文件夹"""
    output_dir = '/home/kky/detrex/Lightglue_test/result'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def align_with_lightglue(ref_img_path, target_img_path, feature_extractor='superpoint'):
    """使用LightGlue对齐图像"""
    image0 = load_image(ref_img_path)
    image1 = load_image(target_img_path)
    
    extractor = SuperPoint(max_num_keypoints=2048).eval() if feature_extractor == 'superpoint' else DISK(max_num_keypoints=2048).eval()
    matcher = LightGlue(features=feature_extractor).eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = extractor.to(device)
    matcher = matcher.to(device)
    
    with torch.no_grad():
        feats0 = extractor.extract(image0.to(device))
        feats1 = extractor.extract(image1.to(device))
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
        
        kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    
    points0 = m_kpts0.cpu().numpy()
    points1 = m_kpts1.cpu().numpy()
    H, mask = cv2.findHomography(points1, points0, cv2.RANSAC, 5.0)#这里的H是单应性矩阵
    
    return H, points0, points1, matches01["matches"].shape[0]

def align_with_sift(ref_img_path, target_img_path):
    """使用SIFT对齐图像"""
    img1 = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    
    points1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    points2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
    
    return H, points1.reshape(-1, 2), points2.reshape(-1, 2), len(good)

def save_visualization(fig, output_dir, filename, dpi=300):
    """保存可视化图像"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved visualization to {path}")

def visualize_and_save_alignment(ref_img_path, target_img_path, output_dir, method='lightglue'):
    """可视化并保存对齐效果"""
    ref_img = cv2.imread(ref_img_path)
    target_img = cv2.imread(target_img_path)
    
    if method == 'lightglue':
        H, ref_points, target_points, num_matches = align_with_lightglue(ref_img_path, target_img_path)
        method_name = 'LightGlue'
        color = (0, 255, 0)  # 绿色
    else:
        H, ref_points, target_points, num_matches = align_with_sift(ref_img_path, target_img_path)
        method_name = 'SIFT'
        color = (0, 165, 255)  # 橙色
    
    # 应用单应性矩阵变换目标点
    transformed_points = cv2.perspectiveTransform(
        target_points.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
    
    # 计算误差
    errors = transformed_points - ref_points
    distances = np.linalg.norm(errors, axis=1)
    
    # 1. 绘制参考图像和点
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.imshow(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
    ax1.scatter(ref_points[:, 0], ref_points[:, 1], c='red', s=10, label='Reference Points')
    ax1.set_title(f'Reference Image with {method_name} Points\n({num_matches} matches)')
    ax1.legend()
    save_visualization(fig1, output_dir, f'{method_name.lower()}_reference_points.png')
    
    # 2. 绘制目标图像和点
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
    ax2.scatter(target_points[:, 0], target_points[:, 1], color='blue', s=10, label='Target Points')#这里颜色变为统一的color
    ax2.set_title(f'Target Image with {method_name} Points\n({num_matches} matches)')
    ax2.legend()
    save_visualization(fig2, output_dir, f'{method_name.lower()}_target_points.png')
    
    # 3. 绘制对齐后的点与参考点的比较
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    ax3.imshow(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
    ax3.scatter(ref_points[:, 0], ref_points[:, 1], c='red', s=10, label='Reference Points')
    ax3.scatter(transformed_points[:, 0], transformed_points[:, 1], color='blue', s=10, label='Aligned Points')
    ax3.set_title(f'Aligned Points vs Reference Points ({method_name})\nAvg Error: {np.mean(distances):.2f}px, Med Error: {np.median(distances):.2f}px')
    ax3.legend()
    save_visualization(fig3, output_dir, f'{method_name.lower()}_aligned_vs_reference.png')
    
    # 4. 绘制误差向量
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    ax4.quiver(ref_points[:, 0], ref_points[:, 1], 
              errors[:, 0], errors[:, 1], 
              angles='xy', scale_units='xy', scale=1, color='purple', width=0.002)
    ax4.imshow(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
    ax4.set_title(f'Alignment Error Vectors ({method_name})\nMax Error: {np.max(distances):.2f}px')
    save_visualization(fig4, output_dir, f'{method_name.lower()}_error_vectors.png')
    
    # 5. 绘制误差直方图
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    ax5.hist(distances, bins=50, color='blue')
    ax5.set_xlabel('Alignment Error (pixels)')
    ax5.set_ylabel('Frequency')
    ax5.set_title(f'{method_name} Alignment Error Distribution')
    save_visualization(fig5, output_dir, f'{method_name.lower()}_error_distribution.png')
    
    return np.mean(distances), np.median(distances), np.max(distances), num_matches

def compare_and_save_results(ref_img_path, target_img_path, output_dir):
    """比较并保存两种方法的结果"""
    # 获取两种方法的结果
    metrics = {}
    for method in ['lightglue', 'sift']:
        metrics[method] = visualize_and_save_alignment(ref_img_path, target_img_path, output_dir, method)
    
    # 读取结果用于比较
    ref_img = cv2.imread(ref_img_path)
    H_lg, ref_points_lg, target_points_lg, _ = align_with_lightglue(ref_img_path, target_img_path)
    H_sift, ref_points_sift, target_points_sift, _ = align_with_sift(ref_img_path, target_img_path)
    
    # 转换点
    transformed_points_lg = cv2.perspectiveTransform(
        target_points_lg.reshape(-1, 1, 2).astype(np.float32), H_lg).reshape(-1, 2)
    transformed_points_sift = cv2.perspectiveTransform(
        target_points_sift.reshape(-1, 1, 2).astype(np.float32), H_sift).reshape(-1, 2)
    
    # 计算误差
    distances_lg = np.linalg.norm(transformed_points_lg - ref_points_lg, axis=1)
    distances_sift = np.linalg.norm(transformed_points_sift - ref_points_sift, axis=1)
    
    # 6. 绘制误差分布比较图
    fig6, ax6 = plt.subplots(figsize=(12, 6))
    ax6.hist(distances_lg, bins=50, alpha=0.5, label='LightGlue', color='green')
    ax6.hist(distances_sift, bins=50, alpha=0.5, label='SIFT', color='orange')
    ax6.set_xlabel('Alignment Error (pixels)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Alignment Error Distribution Comparison')
    ax6.legend()
    save_visualization(fig6, output_dir, 'error_distribution_comparison.png')
    
    # 7. 绘制箱线图比较
    fig7, ax7 = plt.subplots(figsize=(8, 6))
    ax7.boxplot([distances_lg, distances_sift], 
               labels=[f'LightGlue\n({metrics["lightglue"][3]} matches)', 
                      f'SIFT\n({metrics["sift"][3]} matches)'])
    ax7.set_ylabel('Alignment Error (pixels)')
    ax7.set_title('Alignment Error Comparison')
    save_visualization(fig7, output_dir, 'error_boxplot_comparison.png')
    
    # 8. 创建文本摘要
    summary = f"""Alignment Results Summary:
    
=== LightGlue ===
Average Error: {metrics["lightglue"][0]:.2f} pixels
Median Error: {metrics["lightglue"][1]:.2f} pixels
Max Error: {metrics["lightglue"][2]:.2f} pixels
Number of Matches: {metrics["lightglue"][3]}

=== SIFT ===
Average Error: {metrics["sift"][0]:.2f} pixels
Median Error: {metrics["sift"][1]:.2f} pixels
Max Error: {metrics["sift"][2]:.2f} pixels
Number of Matches: {metrics["sift"][3]}

Conclusion:
LightGlue outperforms SIFT with:
- {metrics["lightglue"][0]/metrics["sift"][0]*100:.1f}% of the average error
- {metrics["lightglue"][1]/metrics["sift"][1]*100:.1f}% of the median error
"""
    
    # 保存摘要到文本文件
    summary_path = os.path.join(output_dir, 'results_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"Saved results summary to {summary_path}")
    
    # 打印摘要
    print("\n" + summary)

# 使用示例
if __name__ == "__main__":
    # 图像路径
    ref_img_path = '/home/kky/detrex/lightglue_test/data/before_frame_000197_angle_20.04.png'  # 请替换为你的参考图像路径
    target_img_path = '/home/kky/detrex/lightglue_test/data/before_frame_000223_angle_14.94.png'  # 请替换为你的目标图像路径
    
    # 创建输出文件夹
    output_dir = create_output_folder()
    print(f"All results will be saved to: {output_dir}")
    
    # 比较并保存结果
    compare_and_save_results(ref_img_path, target_img_path, output_dir)