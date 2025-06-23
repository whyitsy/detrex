import cv2
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
import hashlib
import os

from multi_view_tools.logger_setup import setup_multi_view_logger
multi_view_logger = setup_multi_view_logger()

def get_fixed_color(label):
    """根据标签生成固定颜色，确保相同标签始终使用同一颜色"""
    # 使用哈希算法将标签转换为固定颜色
    hash_obj = hashlib.md5(label.encode())
    hex_dig = hash_obj.hexdigest()
    # 取前6位作为RGB颜色值
    r = int(hex_dig[0:2], 16)
    g = int(hex_dig[2:4], 16)
    b = int(hex_dig[4:6], 16)
    return (r, g, b)

def visual_single_view(data, img, output_path):
    """
    可视化单个视角的识别结果
    :param data: 识别结果数据
    :param img: cv2.imread读取的图片
    """
    # 获取参考帧的识别结果，是dict
    bboxes, classes, scores = data['pred_boxes'], data['pred_classes'], data['pred_scores']

    # 将classes转为labels
    labels_path = "/home/kky/detrex/datasets/shampoo/filtered_label.json"
    with open(labels_path, 'r') as f:
        labels_id = json.load(f)
        labels = [labels_id[cls] for cls in classes]
    
    # 将cv2图像转换为PIL图像以支持中文
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 选择一个支持中文的字体路径
    font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"  # 请根据实际环境修改字体路径
    font = ImageFont.truetype(font_path, 20)

    for bbox, label, score in zip(bboxes, labels, scores):
        bbox = np.array(bbox, dtype=np.int32)
        # 根据标签获取固定颜色
        color = get_fixed_color(label)
        
        # 使用PIL绘制矩形框
        draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline=color, width=5)
        
        # 用PIL绘制中文（文本颜色与框颜色一致）
        draw.text((bbox[0], bbox[1] - 25), f"{label} {score:.2f}", font=font, fill=color)
    
    # 保存图片
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    success = cv2.imwrite(output_path, img)
    if success:
        print(f"图片成功保存至: {output_path}")
    else:
        print(f"图片保存失败: {output_path}")

def visual_multi_view_result(data, img, output_path):
    """
    可视化多视角的识别结果
    """
    # 将cv2图像转换为PIL图像以支持中文
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 选择一个支持中文的字体路径
    font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"  # 请根据实际环境修改字体路径
    font = ImageFont.truetype(font_path, 20)

    # 将classes转为labels
    labels_path = "/home/kky/detrex/datasets/shampoo/filtered_label.json"
    with open(labels_path, 'r') as f:
        labels_id = json.load(f)

    data_grid, data_list = filter_by_top_left(data, iou_threshold=0.2)  # 过滤重叠检测结果
    # 记录有数据的坐标
    exist_data = [(x, y, cell) for x, row in enumerate(data_grid) for y, cell in enumerate(row) if cell]  # 仅当 cell 非空时保留
    multi_view_logger.info(f"处理后有{len(exist_data)}个网格有内容")    
    for i, view_data in enumerate(data_grid):
        for j, item in enumerate(view_data):
            if item:
                cls, score, bbox = item[0], item[2], item[1]
                label = labels_id[cls]
                bbox = np.array(bbox, dtype=np.int32)
                # 根据标签获取固定颜色
                color = get_fixed_color(label)
                
                # 使用PIL绘制矩形框
                draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline=color, width=5)
                
                # 用PIL绘制中文（文本颜色与框颜色一致）
                draw.text((bbox[0], bbox[1] - 25), f"{label} {score:.2f}", font=font, fill=color)
    
    # 保存图片
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, img)



def bbox_iou(box1, box2):
    """计算两个边界框之间的IOU（交并比）"""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    x_inter1 = max(x1, x3)
    y_inter1 = max(y1, y3)
    x_inter2 = min(x2, x4)
    y_inter2 = min(y2, y4)
    
    inter_area = max(0, x_inter2 - x_inter1) * max(0, y_inter2 - y_inter1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / (union_area + 1e-10)

def filter_by_top_left(detection_grid, iou_threshold=0.5):
    """
    按左上角位置优先过滤重叠检测框
    
    参数:
    detection_grid: 二维数组，每个元素为[socre,cls,bbox(xyxy)]或空列表
    iou_threshold: IOU重叠阈值，超过此值的检测框将被过滤
    
    返回:
    filtered_grid: 过滤后的网格检测结果
    kept_detections: 保留的检测框列表（含位置信息）
    """
    # 收集所有检测框（带左上角坐标和网格位置）
    detections = []
    for i in range(len(detection_grid)):
        for j in range(len(detection_grid[0])):
            if detection_grid[i][j]:
                cls, bbox, score   = detection_grid[i][j]
                # print(f"Processing grid ({i}, {j}): score={score}, cls={cls}, bbox={bbox}")
                x1, y1, _, _ = bbox
                detections.append((x1, y1, score, cls, bbox, i, j))
    
    # 按左上角坐标排序（x1优先，y1其次）
    detections.sort(key=lambda x: (x[0], x[1]))
    
    # 按左上角顺序过滤重叠框
    keep = []
    while detections:
        # 取出左上角最靠左上的检测框
        x1, y1, score, cls, bbox, i, j = detections.pop(0)
        keep.append((score, cls, bbox, i, j))
        
        # 过滤与当前框重叠的后续框
        filtered = []
        for det in detections:
            det_x1, det_y1, det_score, det_cls, det_bbox, det_i, det_j = det
            if bbox_iou(bbox, det_bbox) <= iou_threshold:
                filtered.append(det)
        detections = filtered
    
    # 还原为网格结构
    filtered_grid = [[[] for _ in row] for row in detection_grid]
    for score, cls, bbox, i, j in keep:
        filtered_grid[i][j] = [cls, bbox, score]
    
    return filtered_grid, keep