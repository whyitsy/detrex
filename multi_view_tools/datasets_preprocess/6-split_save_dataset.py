import os
import json
import random

from detectron2.structures import BoxMode
from multi_view_tools.logger_setup import setup_dataset_logger

dataset_logger = setup_dataset_logger()


def process_dataset(root_dir_path:str, label2id_path: str, label_path: str) -> list:
    """
    加载所有数据集文件，提取框架所需的图像信息和标注信息作为列表，并返回。
    """
    # 加载标签映射
    with open(label2id_path, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    
    with open(label_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    
    output_list = []
    # 遍历根目录
    for level1_path, level1_dirs, _ in os.walk(root_dir_path):
        for level1_dir in level1_dirs:
            for level2_path, _, level2_files in os.walk(os.path.join(level1_path, level1_dir)):  
                    # 获取内层文件夹所有文件并排序
                    sorted_files = sorted(
                        [f for f in level2_files if f.endswith(".json")] ,
                        key=lambda x: int(x.split("_")[2])
                    )
                               
                    
                    # 处理每个文件
                    for filename in sorted_files:
                        # 初始化当前item
                        item = {
                            "file_name": None,
                            "image_id": len(output_list) + 1,
                            "height": None,
                            "width": None,
                            "annotations": []
                        }

                        file_path = os.path.join(level2_path, filename)
                        item["file_name"] = file_path.replace(".json", ".png")  

                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            item["height"] = data["imageHeight"]
                            item["width"] = data["imageWidth"]
                            shapes = data["shapes"]
                        
                        for shape in shapes:
                            if shape["label"] not in labels:
                                continue
                            
                            
                            bbox = calculate_bbox(shape["points"])
                            try:
                                item["annotations"].append({
                                    "category_id": label2id[shape["label"]],
                                    "bbox_mode": BoxMode.XYXY_ABS,
                                    "bbox": bbox
                                })
                            except KeyError:
                                dataset_logger.warning(f"标签 '{shape['label']}' 未在 label2id 中找到，文件为: {file_path}")
                                
                        
                        if item["annotations"]:
                            output_list.append(item)
                        
    return output_list

def calculate_bbox(points: list) -> list:
    """
    转换点坐标为边界框格式。process_dataset函数使用。
    """
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x1 = min(x_coords)
    y1 = min(y_coords)
    x2 = max(x_coords)
    y2 = max(y_coords)

    return [x1, y1, x2, y2]

def train_test_split(input_list: list, 
                     output_train_dir: str,
                     output_test_dir: str, 
                     ratio:int =0.8) -> None:
    """
    将处理好的列表分割为训练集和测试集，并保存为JSON文件。
    :param input_list: 处理好的数据列表
    :param ratio: 训练集占总数据集的比例，默认为0.8
    """
    random.shuffle(input_list)
    # 计算分割位置
    split_index = int(len(input_list) * ratio)
    
    # 分割列表
    train_list = input_list[:split_index]
    test_list = input_list[split_index:]
    
    with open(output_train_dir, "w", encoding="utf-8") as f:
        json.dump(train_list, f, ensure_ascii=False, indent=4)
        dataset_logger.info(f"训练集已保存到{output_train_dir}")

    with open(output_test_dir, "w", encoding="utf-8") as f:
        json.dump(test_list, f, ensure_ascii=False, indent=4)
        dataset_logger.info(f"测试集已保存到{output_test_dir}")

if __name__ == "__main__":
    root_path = "/mnt/data/datasets/shampoo_dataset_new/"
    label2id_path = "/home/kky/detrex/datasets/shampoo_new/5-filtered_label2id.json"
    label_path = "/home/kky/detrex/datasets/shampoo_new/4-filtered_label.json"

    output_train_path = "/home/kky/detrex/datasets/shampoo_new/shampoo_train.json"
    output_test_path = "/home/kky/detrex/datasets/shampoo_new/shampoo_test.json"

    dataset_list = process_dataset(root_path, label2id_path, label_path)
    train_test_split(dataset_list, output_train_path, output_test_path)
    
    dataset_logger.info("数据集处理和分割完成。")