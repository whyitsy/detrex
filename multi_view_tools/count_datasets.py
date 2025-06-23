import os
import argparse
from collections import defaultdict

def count_files_recursively(directory, ignore_hidden=True, file_extensions=None, exclude_dirs=None):
    """
    递归统计指定目录下的文件数量
    
    参数:
        directory (str): 要统计的目录路径
        ignore_hidden (bool): 是否忽略隐藏文件和目录
        file_extensions (list): 要统计的文件扩展名列表，None表示统计所有文件
        exclude_dirs (list): 要排除的目录名称列表
    
    返回:
        tuple: (文件总数, 按扩展名统计的字典)
    """
    if not os.path.isdir(directory):
        raise ValueError(f"{directory} 不是一个有效的目录")
    
    # 初始化计数器
    total_count = 0
    extension_count = defaultdict(int)
    
    # 确保排除目录列表是小写的
    if exclude_dirs:
        exclude_dirs = [d.lower() for d in exclude_dirs]
    
    # 递归遍历目录
    for root, dirs, files in os.walk(directory):
        # 处理排除目录
        if exclude_dirs:
            dirs[:] = [d for d in dirs if d.lower() not in exclude_dirs]
        
        # 处理隐藏目录
        if ignore_hidden:
            dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            # 处理隐藏文件
            if ignore_hidden and file.startswith('.'):
                continue
            
            # 检查文件扩展名
            if file_extensions:
                ext = os.path.splitext(file)[1].lower()
                if not ext or ext[1:] not in file_extensions:
                    continue
            
            # 更新计数器
            total_count += 1
            ext = os.path.splitext(file)[1].lower()
            if ext:
                extension_count[ext[1:]] += 1
            else:
                extension_count['no_extension'] += 1
    
    return total_count, dict(extension_count)

def print_file_stats(directory, total_count, extension_count):
    """打印文件统计信息"""
    print(f"目录: {directory}")
    print(f"总文件数: {total_count}")
    
    if extension_count:
        print("\n按扩展名统计:")
        # 按数量降序排列
        sorted_extensions = sorted(extension_count.items(), key=lambda x: x[1], reverse=True)
        for ext, count in sorted_extensions:
            percentage = (count / total_count) * 100
            print(f"  {ext}: {count} ({percentage:.2f}%)")

def main():
    """主函数，处理命令行参数"""


    try:
        total, extensions = count_files_recursively(
            "/mnt/data/kky/datasets/shampoo_datasets",
            ignore_hidden= False
        )
        print_file_stats("/mnt/data/kky/datasets/shampoo_datasets", total, extensions)
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()
