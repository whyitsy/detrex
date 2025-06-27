#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：lowkeyway time:11/13/2019

import sys
import os
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time

def draw_matches(img1, kp1, img2, kp2, matches, color=(0, 255, 0), thickness=1):
    """绘制特征点匹配结果"""
    return cv.drawMatches(img1, kp1, img2, kp2, matches, None,
                          matchColor=color,
                          singlePointColor=None,
                          flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

def adjust_detector_params(detector_name):
    """使用检测器的默认参数"""
    if detector_name == "ORB":
        return cv.ORB_create(nfeatures=10000)
    elif detector_name == "BRISK":
        return cv.BRISK_create()
    elif detector_name == "AKAZE":
        return cv.AKAZE_create()
    elif detector_name == "SIFT":
        return cv.SIFT_create(nfeatures=10000)
    else:
        return None

def detect_and_match(img1, img2, detector_name, geom_method=None, threshold=5.0):
    """使用指定检测器进行特征检测和匹配"""
    results = {
        "detector": detector_name,
        "geom_method": geom_method,
        "keypoints1": 0,
        "keypoints2": 0,
        "matches": 0,
        "inliers": 0,
        "time": 0
    }

    start_time = time()

    # 创建检测器（使用默认参数）
    detector = adjust_detector_params(detector_name)
    if detector is None:
        print(f"Unsupported detector: {detector_name}")
        return results, None, None, None

    # 检测关键点和计算描述符
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    results["keypoints1"] = len(kp1) if kp1 is not None else 0
    results["keypoints2"] = len(kp2) if kp2 is not None else 0

    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        print(f"{detector_name}: No descriptors found!")
        return results, None, None, None

    # 创建匹配器
    if detector_name == "SIFT":
        bf = cv.BFMatcher_create(cv.NORM_L2, crossCheck=True)
    else:
        bf = cv.BFMatcher_create(cv.NORM_HAMMING, crossCheck=True)

    # 进行特征匹配
    matches = bf.match(des1, des2)
    results["matches"] = len(matches)

    if len(matches) < 4:
        print(f"{detector_name}: Not enough matches found!")
        return results, kp1, kp2, matches

    # 按距离排序匹配点
    matches = sorted(matches, key=lambda x: x.distance)

    # 应用几何验证（如果指定了方法）
    inlier_matches = matches
    if geom_method is not None:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        try:
            H, mask = cv.findHomography(src_pts, dst_pts, geom_method, threshold)
            if mask is not None:
                inliers = mask.ravel().tolist()
                results["inliers"] = sum(inliers)
                inlier_matches = [m for i, m in enumerate(matches) if inliers[i]]
            else:
                results["inliers"] = len(matches)
        except:
            print(f"Error with geometric method {geom_method}")
            results["inliers"] = len(matches)
    else:
        results["inliers"] = len(matches)

    results["time"] = time() - start_time

    return results, kp1, kp2, inlier_matches

def main_func(argv):
    # 读取图像
    img1 = cv.imread("after_frame_000139.png", cv.IMREAD_GRAYSCALE)
    img2 = cv.imread("before_frame_000085.png", cv.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("Error: Could not load images!")
        return

    # 特征检测器列表
    detectors = ["ORB", "BRISK", "AKAZE", "SIFT"]
    # 几何验证方法列表 (None表示原始匹配)
    geom_methods = [None, cv.RANSAC, cv.LMEDS, cv.RHO]
    method_names = ["Raw", "RANSAC", "LMEDS", "RHO"]

    # 存储所有结果
    all_results = []

    # 为每种检测器创建结果文件夹
    for detector in detectors:
        detector_dir = f"{detector}_results"
        if not os.path.exists(detector_dir):
            os.makedirs(detector_dir)

    # 对每种检测器和几何方法组合进行处理
    for detector in detectors:
        print(f"\n===== Processing {detector} =====")

        for method, method_name in zip(geom_methods, method_names):
            # 处理并获取结果
            try:
                results, kp1, kp2, matches = detect_and_match(img1, img2, detector, method, threshold=8.0)

                if kp1 is None or kp2 is None:
                    print(f"Skipping {detector} {method_name} due to errors")
                    continue

                # 保存结果到列表
                results["geom_method"] = method_name
                all_results.append(results)

                # 确定显示的匹配点数量（覆盖整个图像）
                display_count = min(500, len(matches))  # 最多显示500个匹配点
                display_matches = matches[:display_count]

                # 绘制匹配结果
                match_img = draw_matches(img1, kp1, img2, kp2, display_matches)

                # 获取图像尺寸用于标题布局
                h, w = img1.shape[:2]
                title_y = 30 if h > 500 else 20
                title_size = 0.8 if h > 500 else 0.6

                # 添加标题
                title = f"{detector} {method_name}"
                cv.putText(match_img, title, (10, title_y), cv.FONT_HERSHEY_SIMPLEX,
                           title_size, (0, 0, 255), 2, cv.LINE_AA)

                # 添加统计信息
                info_line1 = f"KP: {results['keypoints1']}/{results['keypoints2']}"
                info_line2 = f"Matches: {results['matches']}"
                if method_name != "Raw":
                    info_line2 += f", Inliers: {results['inliers']} ({results['inliers'] / results['matches'] * 100:.1f}%)"

                cv.putText(match_img, info_line1, (10, title_y + 40), cv.FONT_HERSHEY_SIMPLEX,
                           title_size, (0, 0, 255), 1, cv.LINE_AA)
                cv.putText(match_img, info_line2, (10, title_y + 80), cv.FONT_HERSHEY_SIMPLEX,
                           title_size, (0, 0, 255), 1, cv.LINE_AA)

                # 保存图像
                filename = f"{detector}_{method_name}.png"
                save_path = os.path.join(f"{detector}_results", filename)
                cv.imwrite(save_path, match_img)
                print(f"Saved: {save_path}")

            except Exception as e:
                print(f"Error processing {detector} with {method_name}: {e}")

    if not all_results:
        print("No valid results to display!")
        return

    # 创建数据框
    df = pd.DataFrame(all_results)

    # 保存统计数据到CSV
    df.to_csv("feature_matching_stats.csv", index=False)
    print("\nSaved statistics to feature_matching_stats.csv")

    # ===================== 创建特征点统计图 =====================
    plt.figure(figsize=(15, 12), dpi=100)

    # 特征点数量对比
    plt.subplot(2, 2, 1)
    for detector in detectors:
        detector_data = df[df["detector"] == detector]
        plt.plot(method_names, detector_data["keypoints1"], 'o-', linewidth=2, markersize=8, label=f"{detector} KP1")
        plt.plot(method_names, detector_data["keypoints2"], 'x--', linewidth=2, markersize=8, label=f"{detector} KP2")
    plt.title("Keypoints Count Comparison", fontsize=14)
    plt.ylabel("Count", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    # 匹配数量对比
    plt.subplot(2, 2, 2)
    for detector in detectors:
        detector_data = df[df["detector"] == detector]
        plt.plot(method_names, detector_data["matches"], 's-', linewidth=2, markersize=8, label=detector)
    plt.title("Total Matches Comparison", fontsize=14)
    plt.ylabel("Count", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    # 内点数量对比
    plt.subplot(2, 2, 3)
    for detector in detectors:
        detector_data = df[df["detector"] == detector]
        # 原始匹配没有内点概念，跳过
        filtered_data = detector_data[detector_data["geom_method"] != "Raw"]
        plt.plot(filtered_data["geom_method"], filtered_data["inliers"], 'D-', linewidth=2, markersize=8,
                 label=detector)
    plt.title("Inliers after Geometric Verification", fontsize=14)
    plt.ylabel("Count", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    # 处理时间对比
    plt.subplot(2, 2, 4)
    for detector in detectors:
        detector_data = df[df["detector"] == detector]
        plt.plot(method_names, detector_data["time"], '*-', linewidth=2, markersize=10, label=detector)
    plt.title("Processing Time Comparison", fontsize=14)
    plt.ylabel("Time (seconds)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    plt.tight_layout(pad=3.0)
    plt.savefig("Feature_Matching_Comparison.png", dpi=150)
    print("Saved comparison chart to Feature_Matching_Comparison.png")

    # ===================== 创建详细统计表格 =====================
    plt.figure(figsize=(14, 8), dpi=120)
    ax = plt.gca()
    ax.axis('off')

    # 准备表格数据
    table_data = []
    for res in all_results:
        if res["geom_method"] == "Raw":
            inliers = "N/A"
            inlier_ratio = "N/A"
        else:
            inliers = res["inliers"]
            inlier_ratio = f"{res['inliers'] / res['matches'] * 100:.1f}%" if res["matches"] > 0 else "N/A"

        table_data.append([
            res["detector"],
            res["geom_method"],
            f"{res['keypoints1']}",
            f"{res['keypoints2']}",
            f"{res['matches']}",
            f"{inliers}",
            inlier_ratio,
            f"{res['time']:.3f}s"
        ])

    # 创建表格
    columns = ["Detector", "Method", "KP1", "KP2", "Matches", "Inliers", "Inlier Ratio", "Time"]
    table = plt.table(cellText=table_data,
                      colLabels=columns,
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.12] * len(columns))

    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # 设置表头样式
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # 表头行
            cell.set_text_props(fontweight='bold', color='white')
            cell.set_facecolor('#4f81bd')
        elif i % 2 == 1:  # 奇数行
            cell.set_facecolor('#dbe5f0')
        else:  # 偶数行
            cell.set_facecolor('#eaf1dd')

    # 添加标题
    plt.title("Feature Matching Algorithm Comparison", fontsize=16, pad=20)

    # 保存表格
    plt.savefig("Feature_Matching_Statistics.png", bbox_inches='tight', dpi=150)
    print("Saved statistics table to Feature_Matching_Statistics.png")

    # 显示图表
    plt.show()

    print("\nProcessing completed successfully!")


if __name__ == '__main__':
    main_func(sys.argv)