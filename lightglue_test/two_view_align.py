import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
import kornia as K

def Two_View_Align_LightGlue(ref_view, view_to_align, bboxes_center_to_align: list[tuple[float, float]],
                           ransac_iterations=2000, ransac_threshold=2.0,
                           spatial_ratio_threshold=1.2, device='cuda'):
    """
    使用LightGlue替代SIFT的双视图对齐函数
    
    参数:
        ref_view: 参考图像 (numpy数组, HxWxC)
        view_to_align: 待对齐图像 (numpy数组, HxWxC) 
        bboxes_center_to_align: 需要对齐的bbox中心点坐标列表
        device: 使用的计算设备 ('cuda' 或 'cpu')
    """
    
    # 0. 初始化检查
    if (ref_view == view_to_align).all():
        return np.array(bboxes_center_to_align, dtype=np.float32)

    # 1. 转换为PyTorch张量并加载到设备
    def numpy_to_torch(img):
        img = K.image_to_tensor(img, False).float() / 255.
        return img.to(device)
    
    ref_tensor = numpy_to_torch(ref_view)
    align_tensor = numpy_to_torch(view_to_align)
    
    # 2. 初始化LightGlue和特征提取器
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features='superpoint').eval().to(device)
    
    # 3. 提取特征
    with torch.no_grad():
        feats0 = extractor.extract(ref_tensor)
        feats1 = extractor.extract(align_tensor)
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
        
        # 获取匹配点
        kpts0 = feats0['keypoints'].cpu().numpy()
        kpts1 = feats1['keypoints'].cpu().numpy()
        matches = matches01['matches'].cpu().numpy()
        m_kpts0 = kpts0[matches[..., 0]]
        m_kpts1 = kpts1[matches[..., 1]]
    
    print(f"LightGlue匹配点数量: {len(m_kpts0)}")
    
    # 4. 空间一致性过滤
    m_kpts0, m_kpts1 = filter_matches_by_spatial_consistency(
        m_kpts0, m_kpts1, 
        max_ratio=spatial_ratio_threshold
    )
    print(f"过滤后匹配点数量: {len(m_kpts0)}")
    
    if len(m_kpts0) < 4:
        print("警告: 匹配点不足")
        return []
    
    # 5. 计算单应性矩阵
    H, mask = cv2.findHomography(m_kpts1, m_kpts0, cv2.RANSAC, 
                                ransac_threshold, maxIters=ransac_iterations)
    
    if H is None:
        print("警告: 无法计算单应性矩阵")
        return []
    
    # 6. 变换目标点
    points_array = np.array(bboxes_center_to_align, dtype=np.float32)
    points_to_transform = points_array.reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(points_to_transform, H)
    
    return transformed_points.reshape(-1, 2)

def filter_matches_by_spatial_consistency(src_pts, dst_pts, max_ratio=1.2):
    """空间一致性过滤 (与之前相同)"""
    src_dists = np.linalg.norm(src_pts[:,None]-src_pts, axis=2)
    dst_dists = np.linalg.norm(dst_pts[:,None]-dst_pts, axis=2)
    ratios = dst_dists/(src_dists + 1e-6)
    mask = (ratios > 1/max_ratio) & (ratios < max_ratio)
    mask = mask.mean(axis=1) > 0.7
    return src_pts[mask], dst_pts[mask]

# 使用示例
if __name__ == "__main__":
    # 读取图像
    ref_img = cv2.imread("reference.jpg")
    align_img = cv2.imread("target.jpg")
    
    # 模拟一些需要对齐的bbox中心点
    bbox_centers = [(100, 200), (150, 300), (200, 400)]
    
    # 调用对齐函数
    aligned_points = Two_View_Align_LightGlue(
        ref_img, align_img, bbox_centers,
        ransac_iterations=3000,
        ransac_threshold=1.5
    )
    
    print("对齐后的坐标:", aligned_points)