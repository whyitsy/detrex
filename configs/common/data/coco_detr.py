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

import json


from multi_view_tools.AddNoise.addGaussianNoise import AddGaussianNoise
# 定义噪声增强配置 高斯噪声
noise_transform = L(AddGaussianNoise)(
    mean=0,
    std_range=(10, 50),
    p=0.7  # 70%的概率应用噪声
)


train_data_path = "/home/kky/detrex/datasets/shampoo_new/shampoo_train.json"
test_data_path = "/home/kky/detrex/datasets/shampoo_new/shampoo_test.json"
label_path = "/home/kky/detrex/datasets/shampoo_new/4-filtered_label.json"

def shampoo_train_datasets():
    with open(train_data_path, "r", encoding="utf-8") as f:
        train_list = json.load(f)
    return train_list

def shampoo_test_datasets():
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_list = json.load(f)
    return test_list


from detectron2.data import DatasetCatalog
DatasetCatalog.register("shampoo_train_datasets", shampoo_train_datasets)
DatasetCatalog.register("shampoo_test_datasets", shampoo_test_datasets)

## metaData
from detectron2.data import MetadataCatalog
# 添加labels
with open(label_path, "r", encoding='utf-8') as f:
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
            noise_transform,  # 添加噪声增强
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
            noise_transform,  # 添加噪声增强
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
