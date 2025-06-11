import math

def multi_view_grid_process(multi_view_predictions, grid_size=30):
    """
    将处理后的结果做网格对齐算法
    """

    #这里还应该有一个数据结构来装网格的结果   
    grid_data_list = []#这里使用list来存储一个批量的所有网格数据
    #遍历模型的预测
    for single_predictions in multi_view_predictions:
        height = single_predictions["image_height"]
        width = single_predictions["image_width"] #这里得到原来图像的高和宽
        #计算网格数量
        grid_width = width // grid_size
        grid_height = height // grid_size
        #遍历这张图上模型预测的描框
        boxes = single_predictions["pred"].boxes #这里的box是(x1, y1, x2, y2)形式
        scores = single_predictions["pred"].scores #这里的score是每个box的置信度
        classes = single_predictions["pred"].pred_classes #这里的class是每个box的类别
        aligned_center_points = single_predictions["aligned_center_points"] #这里的center_points是每个box的中心点坐标

        grid_data = [[[] for _ in range(grid_width)] for _ in range(grid_height)]#这里使用二维列表结构来存储网格
        for box, score, class_id, aligned_center_point in zip(boxes, scores, classes, aligned_center_points):#这里是遍历每一张图中每一个检测到的实例
            #现在就是要得到预测点的中心坐标
           
            cx, cy = aligned_center_point
            
            #计算中心点在网格的位置
            grid_x = math.floor(cx / grid_size)
            grid_y = math.floor(cy / grid_size)

            #确保网格索引在合理范围内
            grid_x = max(0, min(grid_x, grid_width - 1))
            grid_y = max(0, min(grid_y, grid_height - 1))
            
            grid_data[grid_y][grid_x].append(score)
            grid_data[grid_y][grid_x].append(class_id)
            grid_data[grid_y][grid_x].append(box)

        grid_data_list.append(grid_data)

    return grid_data_list  # 返回所有视图的网格数据


def process_grid_data(grid_data_list, ref_frame_index=0):
    """
    处理多维网格数据，实现投票决策功能
    投票数 > 置信度 > 参考帧顺序
    
    Args:
        grid_data_list: 四维列表 [视角数][网格高度][网格宽度][预测结果]
                预测结果格式: [标签, 边界框(bbox), 置信度]
    
    Returns:
        处理后的单个视角结果 [网格高度][网格宽度][预测结果]
    """
    if not grid_data_list:
        return []
    
    # 获取网格维度信息
    view_count = len(grid_data_list)
    grid_height = len(grid_data_list[0])
    grid_width = len(grid_data_list[0][0])
    
    # 初始化输出结果网格
    result_grid = [[[] for _ in range(grid_width)] for _ in range(grid_height)]
    
    # 遍历每个网格位置
    for h in range(grid_height):
        for w in range(grid_width):
            # 收集所有视角在该位置的预测结果
            position_results = []
            for v in range(view_count):
                # 跳过空结果
                if not grid_data_list[v][h][w]:
                    continue
                position_results.append({
                    'view': v,
                    'label': grid_data_list[v][h][w][0],
                    'bbox': grid_data_list[v][h][w][1],
                    'confidence': grid_data_list[v][h][w][2]
                })
            
            # 如果没有任何视角有结果，不处理
            if not position_results:
                continue
            
            # 1. 补充处理：多数投票原则
            label_votes = {}
            for result in position_results:
                label = result['label']
                if label in label_votes:
                    label_votes[label] += 1
                else:
                    label_votes[label] = 1
            
            # 找到得票最多的标签
            most_voted_label = max(label_votes, key=label_votes.get)
            most_voted_count = label_votes[most_voted_label]
            
            # 2. 校准处理：处理冲突情况
            # 筛选出得票最多的标签的所有结果
            candidates = [r for r in position_results if r['label'] == most_voted_label]
            
            # 如果只有一个候选，直接使用
            if len(candidates) == 1:
                final_result = candidates[0]
            else:
                # 多个候选，按置信度排序
                candidates.sort(key=lambda x: x['confidence'], reverse=True)
                
                # 高置信度覆盖低置信度
                highest_confidence_candidates = [c for c in candidates 
                                               if c['confidence'] == candidates[0]['confidence']]
                
                # 如果仍有多个候选（置信度相同），使用第3帧(参考帧)
                if len(highest_confidence_candidates) > 1:
                    ref_candidates = [c for c in highest_confidence_candidates if c['view'] == ref_frame_index]
                    final_result = ref_candidates[0] if ref_candidates else highest_confidence_candidates[0]
                else:
                    final_result = highest_confidence_candidates[0]
            
            # 3. 存储最终结果
            result_grid[h][w] = [final_result['label'], final_result['bbox'], final_result['confidence']]
    
    return result_grid