import os
import json

"""统计所有标签的实例数量"""
root_dir = "/mnt/data/datasets/shampoo_dataset_new/"

output_dict = {}

for dirpath, dirnames, _ in os.walk(root_dir):
    for subdir in dirnames:
        for sub_dir_path, _, files in os.walk(os.path.join(dirpath, subdir)):
                for file in files:
                    if file.endswith(".json"):
                        with open(os.path.join(dirpath, subdir, file), "r", encoding='utf-8') as f:
                            data = json.load(f)
                            for annotation in data["shapes"]:
                                if annotation["label"] == '': 
                                    continue
                                if annotation["label"] not in output_dict:
                                    output_dict[annotation["label"]] = 1
                                else:
                                    output_dict[annotation["label"]] += 1


# save the output_dict to a JSON file
output_path = "/home/kky/detrex/datasets/shampoo_new/3-label_count.json"
with open(output_path, "w", encoding='utf-8') as f:
    json.dump(output_dict, f, ensure_ascii=False, indent=4)
                                    
                              