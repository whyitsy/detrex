import os
import json

"""
过滤所有标签的空格和U+FEFF（字节顺序标记，BOM）字符 
"""
root_dir = "/mnt/data/datasets/shampoo_dataset_new/"

for dirpath, dirnames, _ in os.walk(root_dir):
    for subdir in dirnames:
        for sub_dir_path, _, files in os.walk(os.path.join(dirpath, subdir)):
                for file in files:
                    if file.endswith(".json"):
                        with open(os.path.join(dirpath, subdir, file), "r", encoding='utf-8') as f:
                            data = json.load(f)
                            
                        for annotation in data["shapes"]: 
                            annotation["label"] = annotation["label"].strip().replace("\uFEFF", "")
                        
                        with open(os.path.join(dirpath, subdir, file), "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=4)

 
                                