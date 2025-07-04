import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

def visualize_points_on_images(
    image1: np.ndarray,
    image2: np.ndarray,
    points1: List[Tuple[float, float]],
    points2: List[Tuple[float, float]],
    output_path: str = "points_visualization.jpg",
    draw_lines: bool = True,
    point_colors: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = ((255, 0, 0), (0, 0, 255)),
    point_size: int = 10,
    line_color: Tuple[int, int, int] = (0, 255, 0),
    line_width: int = 2,
    dpi: int = 300
) -> None:
    """
    在两张图片上可视化点坐标并保存结果
    
    参数:
        image1: 第一张图片 (numpy数组, BGR格式)
        image2: 第二张图片 (numpy数组, BGR格式)
        points1: 第一张图片上的点坐标列表 [(x1,y1), (x2,y2), ...]
        points2: 第二张图片上的点坐标列表 [(x1,y1), (x2,y2), ...]
        output_path: 输出图片路径
        draw_lines: 是否绘制点之间的连线
        point_colors: 两张图片点的颜色 (BGR格式)
        point_size: 点的大小
        line_color: 连线颜色 (BGR格式)
        line_width: 连线宽度
        dpi: 输出图片分辨率
    """
    # 1. 参数校验
    assert len(image1.shape) == 3 and image1.shape[2] == 3, "image1应为3通道BGR图像"
    assert len(image2.shape) == 3 and image2.shape[2] == 3, "image2应为3通道BGR图像"
    assert len(points1) == len(points2), "两张图片的点数量必须相同"
    
    # 2. 创建副本避免修改原图
    img1 = image1.copy()
    img2 = image2.copy()
    
    # 3. 绘制点
    for i, (x, y) in enumerate(points1):
        cv2.circle(img1, (int(x), int(y)), point_size, point_colors[0], -1)
        cv2.putText(img1, str(i), (int(x)+5, int(y)+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    for i, (x, y) in enumerate(points2):
        cv2.circle(img2, (int(x), int(y)), point_size, point_colors[1], -1)
        cv2.putText(img2, str(i), (int(x)+5, int(y)+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    # 4. 水平拼接图像
    concatenated = np.concatenate([img1, img2], axis=1)
    
    # 5. 绘制连线（如果需要）
    if draw_lines:
        width = image1.shape[1]
        for (x1, y1), (x2, y2) in zip(points1, points2):
            cv2.line(concatenated, 
                    (int(x1), int(y1)), 
                    (int(x2)+width, int(y2)), 
                    line_color, line_width)
    
    # 6. 转换颜色空间用于显示
    concatenated_rgb = cv2.cvtColor(concatenated, cv2.COLOR_BGR2RGB)
    
    # 7. 创建画布并保存
    plt.figure(figsize=(16, 8))
    plt.imshow(concatenated_rgb)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"可视化结果已保存至: {output_path}")

# 使用示例
if __name__ == "__main__":
    # 1. 读取图片
    img1 = cv2.imread("/home/kky/detrex/lightglue_test/data/before_frame_000197_angle_20.04.png")  # 替换为你的图片路径
    img2 = cv2.imread("/home/kky/detrex/lightglue_test/data/before_frame_000223_angle_14.94.png")  # 替换为你的图片路径
    
    # 2. 定义点坐标 (示例数据)
    points_img1 = [(100, 200), (150, 300), (200, 400)]  # 第一张图片上的点
    points_img2 = [(244.62, 212.19), (287.26, 307.05), (329.97, 402.08)]  # 第二张图片上的点
    
    # 3. 调用可视化函数
    visualize_points_on_images(
        image1=img1,
        image2=img2,
        points1=points_img1,
        points2=points_img2,
        output_path="matched_points_result.jpg",
        draw_lines=True,
        point_colors=((255, 0, 0), (0, 255, 0)),  # 第一张图红色，第二张图绿色
        point_size=15,
        line_color=(0, 0, 255),  # 蓝色连线
        line_width=2,
        dpi=300
    )