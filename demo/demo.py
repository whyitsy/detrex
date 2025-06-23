# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import sys
import tempfile
import time
import cv2
import tqdm
import json

sys.path.insert(0, "./")  # noqa
from demo.predictors import VisualizationDemo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.structures import Boxes, Instances

from multi_view_tools.view_align import (
    XYXY_To_Center,
    # Two_View_Align
    )
from multi_view_tools.align import Two_View_Align

from multi_view_tools.grid_process import (
    multi_view_grid_process,
    process_grid_data
    )
from multi_view_tools.logger_setup import setup_multi_view_logger

from multi_view_tools.visual.visual_outputs import visual_single_view, visual_multi_view_result
from multi_view_tools.visual.visual_multiple_grids import visualize_multiple_grids

# constants
WINDOW_NAME = "COCO detections"


def setup(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="detrex demo for visualizing customized inputs")
    parser.add_argument(
        "--config-file",
        default="projects/dino/configs/dino_r50_4scale_12ep.py",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--multi_view_input", type=str, default=None,help="Path to multi-view input directory.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--min_size_test",
        type=int,
        default=800,
        help="Size of the smallest side of the image during testing. Set to zero to disable resize in testing.",
    )
    parser.add_argument(
        "--max_size_test",
        type=float,
        default=1333,
        help="Maximum size of the side of the image during testing.",
    )
    parser.add_argument(
        "--img_format",
        type=str,
        default="RGB",
        help="The format of the loading images.",
    )
    parser.add_argument(
        "--metadata_dataset",
        type=str,
        default="coco_2017_val",
        help="The metadata infomation to be used. Default to COCO val metadata.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup(args)

    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.train.init_checkpoint)

    model.eval()

    demo = VisualizationDemo(
        model=model,
        min_size_test=args.min_size_test,
        max_size_test=args.max_size_test,
        img_format=args.img_format,
        metadata_dataset=args.metadata_dataset,
    )

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img, args.confidence_threshold)
            print("predictions", predictions)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit

    elif args.multi_view_input:
        """
        处理多视角图片, 该参数为一个目录，目录下的每个子目录为一组视角
        """
        top_start_time = time.time()
        multi_view_logger = setup_multi_view_logger()
        dirnames = os.listdir(args.multi_view_input)
        for dirname in dirnames:
            # 这里不应该使用os.walk遍历，应该直接使用os.listdir遍历子目录下的图片
            subdirpath = os.path.join(args.multi_view_input, dirname)
            img_files_path = [f for f in os.listdir(subdirpath) if f.endswith((".jpg", ".png", ".jpeg"))]
            
            multi_view_logger.info(f"Processing directory: {subdirpath}")

            # img_files_path = sorted(img_files_path, key=lambda x: int(x.split("_")[0])) # 确保多视角图片的命名格式：index_xxxx.png
            ref_index = [i for i, s in enumerate(img_files_path) if s.startswith("ref_")][0]
            multi_view_logger.info(f"参考帧下标: {ref_index}")
            
            start_time = time.time()
            multi_view_predictions = []
            imgs = []

            for i, path in tqdm.tqdm(enumerate(img_files_path), desc="模型运行中..."):
                sigle_view_result = {}
                sigle_view_result["aligned_center_points"] = []
                # use PIL, to be consistent with evaluation
                img = read_image(os.path.join(subdirpath, path), format="BGR")
                img_cv2 = cv2.imread(os.path.join(subdirpath, path))
                imgs.append(img_cv2)
                predictions, visualized_output = demo.run_on_image(img, args.confidence_threshold)
                # print("predictions['instances']_type: ", type(predictions["instances"]))
                # print("predictions['instances']: ", predictions["instances"])
                sigle_view_result["num_instances"] = len(predictions['instances'])
                sigle_view_result["image_height"], sigle_view_result["image_width"] = predictions['instances'].image_size

                pred = {}
                pred["pred_boxes"] = predictions['instances'].pred_boxes.tensor.cpu().numpy().tolist()
                pred["pred_classes"] = predictions['instances'].pred_classes.cpu().numpy().tolist()
                pred["pred_scores"] = predictions['instances'].scores.cpu().numpy().tolist()
                sigle_view_result["predictions"] = pred
                
                multi_view_predictions.append(sigle_view_result)

            multi_view_logger.info(f"处理视角数: {len(multi_view_predictions)}")
            multi_view_logger.info(f"每个视角识别到的实例数: {[result['num_instances'] for result in multi_view_predictions]}")


            # 可视化单个视角的识别结果
            visual_single_view(
                multi_view_predictions[ref_index]["predictions"],
                np.copy(imgs[ref_index]),
                os.path.join(subdirpath, "output", "single_view_result.png")
            )

            # 视角对齐, 返回对齐的物体中心点   
            for img, sigle_view_result in zip(imgs, multi_view_predictions):
                if sigle_view_result["num_instances"] == 0:
                    sigle_view_result["aligned_center_points"] = []
                    continue
                bboxes_center_to_align = XYXY_To_Center(sigle_view_result["predictions"]["pred_boxes"])

                aligned_bboxes_center = Two_View_Align(
                    imgs[ref_index], img, bboxes_center_to_align
                )
                if isinstance(aligned_bboxes_center, list) and len(aligned_bboxes_center) == 0:
                    continue  # 如果对齐失败，则跳过该视角

                sigle_view_result["aligned_center_points"] = aligned_bboxes_center.tolist()
                
            useful_views = [index for index, result in enumerate(multi_view_predictions) if len(result["aligned_center_points"]) > 0]
            if len(useful_views) == 1:
                multi_view_logger.warning(f"{subdirpath} 对齐过滤后只剩下参考帧一帧")
            else:
                multi_view_logger.info(f"{subdirpath} 对齐过滤后剩余视角: f{[useful_views]}")
                
            file_path = os.path.join(subdirpath, "output", "single_result.json")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding='utf-8') as f:
                json.dump(multi_view_predictions, f, indent=4)
                
            # 网格化处理+数据综合处理
            grid_datas = multi_view_grid_process(multi_view_predictions, grid_size= 100)
            
            final_result = process_grid_data(grid_datas, ref_frame_index=ref_index) # 单个元素[cls, box, score]
            
            if not isinstance(subdirpath, str):
                print(f"警告: subdirpath类型不正确: {type(subdirpath)}")
            file_path = os.path.join(subdirpath, "output", "final_result.json")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding='utf-8') as f:
                json.dump(final_result, f, indent=4)

            # 可视化多视角对齐后的结果
            visual_multi_view_result(
                final_result,
                np.copy(imgs[ref_index]),
                os.path.join(subdirpath, "output", "multi_view_result.png")
            )
            multi_view_logger.info(f"处理时间: {time.time() - start_time:.2f}秒")          
                
        multi_view_logger.info(f"总处理时间: {time.time() - top_start_time:.2f}秒")