import json

"""过滤掉标签数量小于指定阈值的标签"""
filter_threshold = 30  # 设置过滤阈值
label_count_path = "/home/kky/detrex/datasets/shampoo_new/3-label_count.json"
output_path = "/home/kky/detrex/datasets/shampoo_new/4-filtered_label.json"

with open(label_count_path, "r", encoding='utf-8') as f:
    label_with_count = json.load(f)
    filtered_labels = [k for k, v in label_with_count.items() if v >= filter_threshold]

    # 保存过滤后的标签到新文件
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(filtered_labels, f, ensure_ascii=False, indent=4)