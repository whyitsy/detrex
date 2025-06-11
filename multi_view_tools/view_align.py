import cv2
import numpy as np


def XYXY_To_Center(bboxes:list[list[float]])-> list[tuple[float, float]]:
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

def Two_View_Align(ref_view, view_to_align, bboxes_center_to_align:list[tuple[float, float]]):
    """
    ref_view: 参考帧
    view_to_align: 需要对齐的帧
    bboxes_center_to_align: 需要对齐的bbox中心点坐标列表
    对齐两个视图吗, 使用SIFT特征检测和FLANN匹配器。
    """
    if ref_view == view_to_align:
        # 如果两个视图相同，直接返回原始坐标
        return np.array(bboxes_center_to_align, dtype=np.float32)

    # 1. 初始化特征检测器
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(ref_view, None)
    kp2, des2 = sift.detectAndCompute(view_to_align, None)

    # 2. 特征匹配（FLANN匹配器）
    matcher = cv2.FlannBasedMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)

    # 筛选优质匹配点（Lowe's ratio test）
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 4. 计算单应性矩阵（至少需4个匹配点）
    ref_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    align_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)


    H, _ = cv2.findHomography(align_pts, ref_pts, cv2.RANSAC, 5.0)

    # 变换待转换图像上的点坐标
    src_points_transformed = cv2.perspectiveTransform(np.array([bboxes_center_to_align], dtype=np.float32), H)[0]

    return src_points_transformed


