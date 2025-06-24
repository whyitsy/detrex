import os
import json

'''
过滤掉标签数量小于指定阈值的标签，并生成一个新的标签到ID的映射文件
'''
root_dir = "/mnt/data/datasets/shampoo_dataset_new/"
filtered_label_path = "/home/kky/detrex/datasets/shampoo_new/4-filtered_label.json"
output_path = "/home/kky/detrex/datasets/shampoo_new/5-filtered_label2id.json"
output_dict = {}

with open(filtered_label_path, "r", encoding='utf-8') as f:
    filtered_label = json.load(f)

image_count = 0
annotation_count = 0
video_count = 0
for dirpath, dirnames, _ in os.walk(root_dir):
    for subdir in dirnames:
        for sub_dir_path, _, files in os.walk(os.path.join(dirpath, subdir)):
                video_count += 1
                for file in files:
                    image_count += 1
                    if file.endswith(".json"):
                        with open(os.path.join(dirpath, subdir, file), "r", encoding='utf-8') as f:
                            data = json.load(f)
                            for annotation in data["shapes"]:
                                if annotation["label"] == '': 
                                    continue
                                if annotation["label"] not in filtered_label:
                                    continue
                                annotation_count += 1
                                if annotation["label"] not in output_dict:
                                    output_dict[annotation["label"]] = len(output_dict) 


print(f"Total images: {image_count}")
print(f"Total annotations: {annotation_count}")
print(f"Total videos: {video_count}")
# save the output_dict to a JSON file
with open(output_path, "w", encoding='utf-8') as f:
    json.dump(output_dict, f, ensure_ascii=False, indent=4)
                                    
                              