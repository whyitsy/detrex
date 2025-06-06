from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator

from detrex.data import DetrDatasetMapper

from detectron2.structures import BoxMode
import os
import json
import logging
import random


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_dataset(root_dir:str, label2id_path: str, label_path: str) -> None:
    # 加载标签映射
    with open(label2id_path, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    
    with open(label_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    
    output_list = []
    # 遍历根目录
    for level1_path, level1_dirs, _ in os.walk(root_dir):
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
                                logger.warning(f"标签 '{shape['label']}' 未在 label2id 中找到，文件为: {file_path}")
                                
                        
                        if item["annotations"]:
                            output_list.append(item)
                        
    return output_list

def calculate_bbox(points: list) -> list:
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x1 = min(x_coords)
    y1 = min(y_coords)
    x2 = max(x_coords)
    y2 = max(y_coords)

    return [x1, y1, x2, y2]

def train_test_split(input_list: list, ratio:int =0.8) -> tuple[list, list]:

    random.shuffle(input_list)
    # 计算分割位置
    split_index = int(len(input_list) * ratio)
    
    # 分割列表
    train_list = input_list[:split_index]
    test_list = input_list[split_index:]
    
    with open("/mnt/data/kky/datasets/shampoo_train.json", "w", encoding="utf-8") as f:
        json.dump(train_list, f, ensure_ascii=False, indent=4)

    with open("/mnt/data/kky/datasets/shampoo_test.json", "w", encoding="utf-8") as f:
        json.dump(test_list, f, ensure_ascii=False, indent=4)



def train_load(train_dataset_dir:str) -> list:
    with open(train_dataset_dir, "r", encoding="utf-8") as f:
        train_list = json.load(f)
    return train_list

def test_load(test_dataset_dir:str) -> list:
    with open(test_dataset_dir, "r", encoding="utf-8") as f:
        test_list = json.load(f)
    return test_list
    

# 初次加载数据集划分训练集和测试集保存下来。
# train_list, test_list = train_test_split(
#     process_dataset(root_dir="/mnt/data/kky/datasets/shampoo_datasets",
#                     label2id_path="/home/kky/detrex/datasets/shampoo/filtered_label2id.json",
#                     label_path="/home/kky/detrex/datasets/shampoo/filtered_label.json"),
#     ratio=0.8
# )




def shampoo_train_datasets():
    return train_load("/mnt/data/kky/datasets/shampoo_train.json")

def shampoo_test_datasets():
    return test_load("/mnt/data/kky/datasets/shampoo_test.json")
    
from detectron2.data import DatasetCatalog
DatasetCatalog.register("shampoo_train_datasets", shampoo_train_datasets)
DatasetCatalog.register("shampoo_test_datasets", shampoo_test_datasets)

## metaData
from detectron2.data import MetadataCatalog
import json
# 添加labels
with open("/home/kky/detrex/datasets/shampoo/filtered_label.json", "r", encoding='utf-8') as f:
    label2id = json.load(f)
    MetadataCatalog.get("shampoo_train_datasets").thing_classes = list(label2id)
    MetadataCatalog.get("shampoo_test_datasets").thing_classes = list(label2id)



dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="shampoo_train_datasets"),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        augmentation_with_crop=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(400, 500, 600),
                sample_style="choice",
            ),
            L(T.RandomCrop)(
                crop_type="absolute_range",
                crop_size=(384, 600),
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        is_train=True,
        mask_on=False,
        img_format="RGB",
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="shampoo_test_datasets", filter_empty=False),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
