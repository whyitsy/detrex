import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random

def visualize_multiple_grids(grids, titles=None, colors=None, figsize=(12, 10), 
                             grid_size=None, show_grid_lines=True):
    """
    将多个网格化的目标检测结果可视化在同一张图片上
    
    参数:
    grids: 包含多个网格数据的列表，每个网格是一个二维数组
    titles: 每个网格数据的标题列表
    colors: 每个网格数据的边框颜色列表
    figsize: 图像大小
    grid_size: 网格大小(rows, cols)，如果为None则自动计算
    show_grid_lines: 是否显示网格线
    """
    # 设置默认值
    if titles is None:
        titles = [f"Grid {i+1}" for i in range(len(grids))]
    
    if colors is None:
        # 使用预定义的颜色，不够则随机生成
        predefined_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
        colors = predefined_colors[:len(grids)]
        if len(grids) > len(predefined_colors):
            for _ in range(len(grids) - len(predefined_colors)):
                colors.append((random.random(), random.random(), random.random()))
    
    # 确定图像尺寸
    if grid_size is None:
        max_rows = max(len(grid) for grid in grids)
        max_cols = max(len(grid[0]) for grid in grids)
    else:
        max_rows, max_cols = grid_size
    
    # 创建图像
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制每个网格的数据
    for i, grid in enumerate(grids):
        color = colors[i]
        title = titles[i]
        
        for row_idx, row in enumerate(grid):
            for col_idx, cell in enumerate(row):
                if cell:  # 如果当前网格有检测结果
                    score, cls, bbox = cell
                    x1, y1, x2, y2 = bbox
                    
                    # 绘制边界框
                    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                     linewidth=2, edgecolor=color, facecolor='none',
                                     label=title if row_idx == 0 and col_idx == 0 else "")
                    ax.add_patch(rect)
                    
                    # 添加标签
                    label = f"{title}: cls={cls}, score={score:.2f}"
                    ax.text(x1, y1 - 5, label, color=color, fontsize=9, 
                            bbox=dict(facecolor='white', alpha=0.5, pad=1))
    
    # 设置图像属性
    if show_grid_lines:
        # 绘制网格线
        for i in range(max_rows + 1):
            ax.axhline(y=i * (100 / max_rows), color='gray', linestyle='-', alpha=0.3)
        for j in range(max_cols + 1):
            ax.axvline(x=j * (100 / max_cols), color='gray', linestyle='-', alpha=0.3)
    
    ax.set_title('Multiple Grid Detections Visualization')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    
    # 设置坐标轴范围
    ax.set_xlim(0, 100)  # 假设图像宽度为100个单位
    ax.set_ylim(100, 0)  # 反转y轴，使原点在左上角
    
    # 添加图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # 去重
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.tight_layout()
    plt.show()
