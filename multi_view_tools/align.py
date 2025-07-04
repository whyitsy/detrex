import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform

def XYXY_To_Center(bboxes: list[list[float]]) -> list[tuple[float, float]]:
    """
    bboxes: list of bounding boxes, each represented as [x1, y1, x2, y2]
    Returns a list of center points for each bounding box.
    """
    centers = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        centers.append((center_x, center_y))
    return centers

def calculate_match_quality(ref_pts, align_pts):
    """计算匹配点的质量指标"""
    # 计算参考点和对齐点之间的距离标准差
    distances = np.sqrt(np.sum((ref_pts - align_pts) ** 2, axis=1))
    return np.std(distances)

def filter_matches_by_spatial_consistency(ref_pts, align_pts, max_ratio=1.5, min_valid_pairs=5):
    """基于空间一致性过滤匹配点，增加鲁棒性"""
    if len(ref_pts) < 4:
        return ref_pts, align_pts
    
    # 计算参考点之间的距离矩阵
    ref_distances = squareform(pdist(ref_pts))
    align_distances = squareform(pdist(align_pts))
    
    # 计算距离比率矩阵
    ratio_matrix = np.zeros_like(ref_distances)
    valid_mask = np.logical_and(ref_distances > 0, align_distances > 0)
    
    # 计算有效距离对的比率
    ratio_matrix[valid_mask] = ref_distances[valid_mask] / align_distances[valid_mask]
    
    # 统计每个点的有效比率分布
    point_ratios = []
    for i in range(len(ref_pts)):
        ratios = []
        for j in range(len(ref_pts)):
            if i != j and valid_mask[i, j]:
                ratios.append(ratio_matrix[i, j])
        if ratios:
            point_ratios.append(np.array(ratios))
        else:
            point_ratios.append(np.array([]))
    
    # 计算每个点的有效比率范围
    valid_points = []
    for i, ratios in enumerate(point_ratios):
        if len(ratios) >= min_valid_pairs:  # 至少需要有足够的有效对
            median_ratio = np.median(ratios)
            # 使用更宽松的阈值判断
            valid_ratios = np.logical_and(ratios > median_ratio/max_ratio, ratios < median_ratio*max_ratio)
            valid_ratio_count = np.sum(valid_ratios)
            
            # 如果大部分比率都是一致的，则认为该点有效
            if valid_ratio_count / len(ratios) > 0.5:
                valid_points.append(i)
    
    print(f"空间一致性过滤: 原始点={len(ref_pts)}, 过滤后={len(valid_points)}")
    
    if len(valid_points) < 4:
        # 如果过滤后点太少，尝试更宽松的策略
        print("警告: 严格过滤后匹配点不足，尝试宽松策略")
        # 寻找全局一致的比率范围
        all_valid_ratios = ratio_matrix[valid_mask]
        if len(all_valid_ratios) > 10:  # 至少需要有足够的样本
            global_median = np.median(all_valid_ratios)
            global_valid_mask = np.logical_and(
                ratio_matrix > global_median/max_ratio, 
                ratio_matrix < global_median*max_ratio
            )
            
            # 统计每个点的全局有效对数量
            point_valid_counts = np.sum(global_valid_mask, axis=1)
            # 选择有效对数量最多的前N个点
            sorted_indices = np.argsort(point_valid_counts)[::-1]
            top_indices = sorted_indices[:min(100, len(sorted_indices))]  # 最多取100个点
            
            # 确保至少有4个点
            if len(top_indices) >= 4:
                print(f"宽松策略保留点: {len(top_indices)}")
                return ref_pts[top_indices], align_pts[top_indices]
    
    # 返回过滤后的点
    if valid_points:
        return ref_pts[valid_points], align_pts[valid_points]
    else:
        # 如果没有找到有效点，返回原始点（不推荐，但避免程序崩溃）
        print("警告: 空间一致性过滤后没有找到有效点，返回原始点集")
        return ref_pts, align_pts

def Two_View_Align(ref_view, view_to_align, bboxes_center_to_align: list[tuple[float, float]], 
                   use_bf_matcher=False, ransac_iterations=2000, ransac_threshold=2.0,
                   spatial_ratio_threshold=1.2):
    """
    ref_view: 参考帧
    view_to_align: 需要对齐的帧
    bboxes_center_to_align: 需要对齐的bbox中心点坐标列表
    use_bf_matcher: 是否使用BFMatcher代替FLANN
    ransac_iterations: RANSAC迭代次数
    ransac_threshold: RANSAC距离阈值
    spatial_ratio_threshold: 空间一致性过滤的距离比率阈值
    对齐两个视图, 使用SIFT特征检测和特征匹配。
    """
    
    if (ref_view == view_to_align).all():
        # 如果两个视图相同，直接返回原始坐标
        return np.array(bboxes_center_to_align, dtype=np.float32)

    # 1. 初始化特征检测器 - 增加关键点数量和调整参数
    sift = cv2.SIFT_create(
        nfeatures=2000,           # 增加特征点数量
        contrastThreshold=0.01,   # 降低对比度阈值，检测更多特征点
        edgeThreshold=100         # 增加边缘阈值，减少边缘附近的特征点
    )
    
    # 转换为灰度图进行特征检测
    if len(ref_view.shape) == 3:
        ref_gray = cv2.cvtColor(ref_view, cv2.COLOR_BGR2GRAY)
        align_gray = cv2.cvtColor(view_to_align, cv2.COLOR_BGR2GRAY)
    else:
        ref_gray = ref_view
        align_gray = view_to_align
    
    # 检测并计算特征点
    kp1, des1 = sift.detectAndCompute(ref_gray, None)
    kp2, des2 = sift.detectAndCompute(align_gray, None)
    
    print(f"检测到的特征点数量: 参考图={len(kp1)}, 待对齐图={len(kp2)}")

    # 2. 特征匹配
    if use_bf_matcher:
        # 使用BFMatcher代替FLANN
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)
    else:
        # 使用FLANN匹配器
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        matches = matcher.knnMatch(des1, des2, k=2)

    # 筛选优质匹配点（Lowe's ratio test）
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # 稍微放宽比率阈值
            good_matches.append(m)
    
    print(f"初始匹配点数量: {len(good_matches)}")

    # 3. 提取匹配点坐标
    if len(good_matches) < 4:
        print(f"警告: 匹配点数量不足 ({len(good_matches)}/4)，无法计算单应性矩阵")
        return []

    ref_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    align_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
    
    # 4. 基于空间一致性过滤匹配点
    ref_pts, align_pts = filter_matches_by_spatial_consistency(
        ref_pts, align_pts, 
        max_ratio=spatial_ratio_threshold
    )
    
    print(f"过滤后匹配点数量: {len(ref_pts)}")
    
    if len(ref_pts) < 4:
        print(f"警告: 过滤后匹配点数量不足 ({len(ref_pts)}/4)，尝试使用未过滤的点")
        return []

    # 5. 计算单应性矩阵
    # 尝试多种方法并选择最佳结果
    methods = [
        (cv2.RANSAC, ransac_threshold, ransac_iterations),
        (cv2.LMEDS, 0, 0),
        (cv2.RHO, 0, 0)
    ]
    
    best_H = None
    best_mask = None
    best_quality = float('inf')
    
    for method, threshold, iterations in methods:
        try:
            if method == cv2.RANSAC:
                H, mask = cv2.findHomography(align_pts, ref_pts, method, threshold, maxIters=iterations)
            else:
                H, mask = cv2.findHomography(align_pts, ref_pts, method)
            
            if H is not None:
                # 计算内点
                inliers = mask.ravel().nonzero()[0]
                if len(inliers) >= 4:
                    inlier_ref_pts = ref_pts[inliers]
                    inlier_align_pts = align_pts[inliers]
                    
                    # 计算内点变换后的坐标
                    inlier_align_pts_homogeneous = np.column_stack((inlier_align_pts, np.ones(len(inlier_align_pts))))
                    transformed_pts = np.dot(H, inlier_align_pts_homogeneous.T).T
                    transformed_pts = transformed_pts[:, :2] / transformed_pts[:, 2:3]
                    
                    # 计算变换质量
                    quality = calculate_match_quality(inlier_ref_pts, transformed_pts)
                    
                    if quality < best_quality:
                        best_quality = quality
                        best_H = H
                        best_mask = mask
        except Exception as e:
            print(f"方法 {method} 计算单应性矩阵失败: {e}")
    
    if best_H is None:
        print("警告: 无法计算有效的单应性矩阵")
        return []
    
    print(f"最佳单应性矩阵质量: {best_quality}")
    if best_quality < 0.5:
        print("警告: 单应性矩阵质量较差，可能导致对齐不准确, 略过")
        return []
    
    # 6. 变换待对齐图像上的点坐标
    points_array = np.array(bboxes_center_to_align, dtype=np.float32)
    
    # 确保点的形状正确
    if points_array.ndim == 1:
        points_to_transform = points_array.reshape(1, 1, 2)
    else:
        points_to_transform = points_array.reshape(-1, 1, 2)
    
    # 应用变换
    src_points_transformed = cv2.perspectiveTransform(points_to_transform, best_H)
    
    return src_points_transformed.reshape(-1, 2)