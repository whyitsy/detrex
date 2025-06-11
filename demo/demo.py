# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import sys
import tempfile
import time
import warnings
import cv2
import tqdm

sys.path.insert(0, "./")  # noqa
from demo.predictors import VisualizationDemo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.structures import Boxes, Instances

from multi_view_tools.view_align import (
    XYXY_To_Center,
    Two_View_Align
    )

from multi_view_tools.grid_process import (
    multi_view_grid_process,
    process_grid_data
    )

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
        TODO: 将参考帧放在最前面的位置, 确保一组视角中能够确定参考帧的位置。需要修改现在的逻辑
        TODO: 对final_result进行可视化
        TODO: 将多视角投票结果与单独视角的结果可视化进行对比

        处理多视角图片, 该参数为一个目录，目录下的每个子目录为一组视角
        每个子目录下的图片命名格式为 index_xxxx.png, 其中 index 为视角编号
        """
        for dirpath, dirnames, _ in os.walk(args.multi_view_input):
            for dirname in dirnames:
                for subdirpath, _, img_files_path in os.walk(os.path.join(dirpath, dirname)):
                    img_files_path = sorted(img_files_path, key=lambda x: int(x.split("_")[0])) # 确保多视角图片的命名格式：index_xxxx.png
                    start_time = time.time()
                    multi_view_predictions = []
                    imgs = []

                    for path in tqdm.tqdm(img_files_path):
                        sigle_view_result = {}
                        sigle_view_result["aligned_center_box_points"] = []
                        # use PIL, to be consistent with evaluation
                        img = read_image(os.path.join(subdirpath, path), format="BGR")
                        img_cv2 = cv2.imread(os.path.join(subdirpath, path))
                        imgs.append(img_cv2)
                        predictions, visualized_output = demo.run_on_image(img, args.confidence_threshold)
                        sigle_view_result["num_instances"] = predictions.num_instances
                        sigle_view_result["image_height"] = predictions.image_height
                        sigle_view_result["image_width"] = predictions.image_width
                        pred = {}
                        pred["pred_boxes"] = predictions.get("pred_boxes").cpu().tolist()
                        pred["pred_classes"] = predictions.get("pred_classes").tolist()
                        pred["scores"] = predictions.get("scores").tolist()
                        sigle_view_result["predictions"] = pred
                        
                        multi_view_predictions.append(sigle_view_result)
                    
                    # 视角对齐, 返回对齐的物体中心点   
                    for img, sigle_view_result in zip(imgs, multi_view_predictions):
                        bboxes_center_to_align = XYXY_To_Center(sigle_view_result["predictions"]["pred_boxes"])
                        aligned_bboxes_center = Two_View_Align(
                            imgs[2], img, bboxes_center_to_align
                        )

                        sigle_view_result["aligned_center_box_points"].append(aligned_bboxes_center)
                    # 网格化处理+数据综合处理
                    grid_datas = multi_view_grid_process(multi_view_predictions, grid_size=30)
                    final_result = process_grid_data(grid_datas)


                    logger.info(
                        "{}: {} in {:.2f}s".format(
                            path,
                            "detected {} instances".format(multi_view_predictions["num_instances"])
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

    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
