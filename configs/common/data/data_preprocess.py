import json
import os
import random


# 第一次运行时需要将数据集划分为训练集和测试集
def train_test_split(root_dir:str, ratio:float=0.8) -> tuple[list[str], list[str]]:
    """
    将数据集划分为训练集和测试集, 然后保存
    """
    for _, input_dir_list, _ in os.walk(root_dir):

        random.shuffle(input_dir_list)
    
        # 计算分割位置
        split_index = int(len(input_dir_list) * ratio)
        
        # 分割列表
        train_list = input_dir_list[:split_index]
        test_list = input_dir_list[split_index:]
        break
    
    with open("/mnt/data/kky/datasets/multi_view_train_list.json", "w", encoding="utf-8") as f:
        json.dump(train_list, f, ensure_ascii=False, indent=4)
    
    with open("/mnt/data/kky/datasets/multi_view_test_list.json", "w", encoding="utf-8") as f:
        json.dump(test_list, f, ensure_ascii=False, indent=4)

    return train_list, test_list


if __name__ == "__main__":
    # 划分训练集和测试集
    train_list, test_list = train_test_split(
        root_dir="/mnt/data/datasets/shampoo_datasets",
        ratio=0.8
    )