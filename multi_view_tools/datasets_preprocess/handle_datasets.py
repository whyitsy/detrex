import os
import re
'''
删除复制过程中出现的重复文件
'''

# 数据集根目录
dataset_root = '/mnt/data/datasets/shampoo_dataset_new/'

for dir_path, dir_names, file_names in os.walk(dataset_root):
    for file_name in file_names:
        if re.search(r'\(\d\)\.',file_name):
            # 输出当前文件的路径
            old_file_path = os.path.join(dir_path, file_name)
            # 删除该文件
            os.remove(old_file_path)
            print(f"Deleted file: {old_file_path}")