# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/demo/demo.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------

import argparse
import multiprocessing as mp
import os
import torch
import random
# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import time
import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)
from predictor import VisualizationDemo

# constants
WINDOW_NAME = "OneFormer Demo"

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="oneformer demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="../configs/ade20k/swin/oneformer_swin_large_IN21k_384_bs16_160k.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--task", help="Task type")
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
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    def find_files(root_dir, par_dir = ""):
        """
        递归查找指定目录下的所有文件路径
        """
        file_paths = []
        file_abspaths = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                file_paths.append(os.path.join(root, file))
                file_abspaths.append(os.path.join(par_dir, file))
            for dir in dirs:
                paths, abspaths = find_files(os.path.join(root, dir), os.path.join(par_dir, dir))
                file_paths.extend(paths)
                file_abspaths.extend(abspaths)
            break
        return file_paths, file_abspaths
    
    if args.input:
        if os.path.isdir(args.input[0]):
            args.input, abspaths = find_files(args.input[0])

        out_json = {}
        for path, abspath in tqdm.tqdm(zip(args.input, abspaths), total=len(args.input)):
            # use PIL, to be consistent with evaluation
                
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img, args.task)

            panoptic_seg, segments_info = predictions["panoptic_seg"]
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
                # if len(args.input) == 1:
                #     for k in visualized_output.keys():
                #         os.makedirs(k, exist_ok=True)
                #         out_filename = os.path.join(k, args.output)
                #         visualized_output[k].save(out_filename)
                # else:
                from PIL import Image
                opath = os.path.join(args.output, "panoptic")
                out_filename = os.path.join(opath, abspath.replace(".jpg", ".png"))
                os.makedirs(os.path.dirname(out_filename), exist_ok=True)

                panoptic_img = Image.fromarray(panoptic_seg.cpu().numpy().astype(np.uint8))
                panoptic_img.save(out_filename)

                out_json[abspath] = segments_info

                    # for k in visualized_output.keys():
                    #     opath = os.path.join(args.output, k)    
                    #     os.makedirs(opath, exist_ok=True)
                    #     out_filename = os.path.join(opath, os.path.basename(path))
                    #     visualized_output[k].save(out_filename)    
            else:
                raise ValueError("Please specify an output path!")
        if args.output:
            import json
            with open(os.path.join(args.output, "panoptic.json"), "w") as f:
                json.dump(out_json, f)
    else:
        raise ValueError("No Input Given")
