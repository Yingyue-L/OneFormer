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
import glob

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)
from predictor import VisualizationDemo
from colormap import random_color
from detectron2.data import MetadataCatalog
import inflect
import math
import json

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]



import warnings
warnings.filterwarnings("ignore")
 
 
# Declare method of inflect module
p = inflect.engine()

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
        type=str,
        help="folder",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)

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
    parser.add_argument(
        "--depth",
        type=str,
        help="folder",
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

    os.makedirs(args.output, exist_ok=True)
    demo = VisualizationDemo(cfg, parallel=True)

    def find_files(root_dir, par_dir = ""):
        """
        递归查找指定目录下的所有文件路径
        """
        file_paths = []
        file_abspaths = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                # if os.path.join(par_dir, file) not in merge_json:
                # if not os.path.exists(f"{args.output}/{os.path.join(par_dir, file).split('.')[0]}.jpg"):
                if (file.endswith('.jpg') or file.endswith('.png')):
                    file_paths.append(os.path.join(root, file))
                    file_abspaths.append(os.path.join(par_dir, file))
            for dir in dirs:
                paths, abspaths = find_files(os.path.join(root, dir), os.path.join(par_dir, dir))
                file_paths.extend(paths)
                file_abspaths.extend(abspaths)
            break
        return file_paths, file_abspaths

    if args.input:
        # with open(os.path.join(args.output, f"error_{args.task}.txt"), "w") as f:
        #     f.write("")

        # if args.output:
            # if os.path.exists(f'{args.output}/{args.task}.json'):
            #     with open(f'{args.output}/{args.task}.json', 'r') as infile:
            #         merge_json = json.load(infile)
            # else:
            #     merge_json = {}
            # with open(f'{args.output}/{args.task}_v1.txt', 'w') as f:
            #     f.write("")
            
            # with open(f'{args.output}/{args.task}_{args.num_chunks}_{args.chunk_idx}.jsonl', 'w') as f:
            #     f.write("")
        paths, abspaths = find_files(args.input)
        # paths, abspaths = paths[:10], abspaths[:10]
        paths = get_chunk(paths, args.num_chunks, args.chunk_idx)
        abspaths = get_chunk(abspaths, args.num_chunks, args.chunk_idx)
        
        segment_json = {}
        for path, abspath in tqdm.tqdm(zip(paths, abspaths), disable=not args.output, total=len(paths)):
            # if abspath in merge_json:
            #     ori_json_output = merge_json[abspath]
            # else: 
            # use PIL, to be consistent with evaluation
            # try:
            # img = read_image(path, format="BGR")
            if os.path.exists(f"{args.output}/{abspath.split('.')[0]}.jpg"):
                continue
            img = Image.open(path).convert("RGB")
            from detectron2.data.detection_utils import _apply_exif_orientation
            img = _apply_exif_orientation(img)
            max_length = 2000
            scale_factor = 1 if img.width < max_length else max_length / img.width
            img  = img.resize((int(scale_factor * img.width), int(scale_factor * img.height)))

            img = np.asarray(img)
            img = img[:, :, ::-1]
            # except:
            #     with open(os.path.join(args.output, f"error_{args.task}.txt"), "a") as f:
            #         f.write(abspath + " Image open error\n")
            #     continue
            start_time = time.time()
            depth = None

            # if args.task == "panoptic":
            #     # try:
            # depth_path = os.path.join(args.depth, abspath.split(".")[0] + ".npy")
            # depth = np.load(depth_path)
            #     # except:
            #     #     with open(os.path.join(args.output, f"error_{args.task}.txt"), "a") as f:
            #     #         f.write(abspath + " Depth open error\n")
            #     #     continue
            #     # try:
            #     predictions, visualized_output, txt_output, json_output = demo.run_on_image(img, args.task)
            #     # except:
            #     #     with open(os.path.join(args.output, f"error_{args.task}.txt"), "a") as f:
            #     #         f.write(abspath + " Panoptic error\n")
            #         # continue
            #     # save
            #     # if args.output:
            #     #     panoptic_inference = visualized_output["panoptic_inference"]
            #     #     panoptic_inference.save(f"{args.output}/{abspath.split('.')[0]}.jpg")


            # else:
            predictions, visualized_output, txt_output, json_output = demo.run_on_image(img, args.task, depth)
            if args.output:
                panoptic_inference = visualized_output["panoptic_inference"]
                par_dir = os.path.dirname(f"{args.output}/{abspath.split('.')[0]}.jpg")
                if not os.path.exists(par_dir):
                    os.makedirs(par_dir)
                panoptic_inference.save(f"{args.output}/{abspath.split('.')[0]}.jpg")

            # segment_json[abspath] = json_output
            # assert len(json_output) == len(ori_json_output)
            # assert all([new_obj["category"] == obj["category"] for new_obj, obj in zip(json_output, ori_json_output)])
            # assert all([new_obj["bbox"] == obj["bbox"] for new_obj, obj in zip(json_output, ori_json_output)])
            # segment_json[abspath] = [
            #     {
            #         "depth": obj["depth"] if "depth" in obj else None,
            #         "bbox": obj["bbox"],
            #         "category": obj["category"],
            #         "is_thing": obj["is_thing"],
            #         "mask": new_obj["mask"],
            #     }
            #     for new_obj, obj in zip(json_output, ori_json_output)
            # ]
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )
            # if len(json_output) == 0:
            #     txt_output = "There are no detectable objects in the image"
            # else:
            #     categories = {}
            #     for obj in json_output:
            #         if obj["category"] not in categories:
            #             categories[obj["category"]] = 1
            #         elif obj["is_thing"]:
            #             categories[obj["category"]] += 1
            #     if len(categories) == 1 and categories[list(categories.keys())[0]] == 1:
            #         txt_output = "The objects present in the image is: "
            #     else:
            #         txt_output = "The objects present in the image are: "
            #     SPECIAL_WORDS = ['baseball bat',
            #                     'baseball glove',
            #                     'cell phone',
            #                     'dining table',
            #                     'fire hydrant',
            #                     'french fries',
            #                     'hair drier',
            #                     'hot dog',
            #                     'parking meter',
            #                     'potted plant',
            #                     'soccer ball',
            #                     'soccer player',
            #                     'sports ball',
            #                     'stop sign',
            #                     'teddy bear',
            #                     'tennis racket',
            #                     'toy figure',
            #                     'traffic light',
            #                     'wine glass',
            #                     "bulletin board",
            #                     "crt screen",
            #                     "trash can",
            #                     "trade name",
            #                     "conveyer belt",
            #                     "dirt track",
            #                     "street lamp",
            #                     "arcade machine",
            #                     "swivel chair",
            #                     "kitchen island",
            #                     "coffee table",
            #                     "screen door"
            #                     ]
            #     for cat in categories:
            #         if categories[cat] == 1:
            #             cat = cat.split(",")[0].split("-")[0]
            #             txt_output += f"{p.singular_noun(cat) if cat != 'grass' and cat not in SPECIAL_WORDS and p.singular_noun(cat) else cat}, "
            #         else:
            #             txt_output += f"{p.number_to_words(categories[cat])} {p.plural(cat) if cat not in SPECIAL_WORDS and p.plural(cat) else cat}, "
            #     txt_output = txt_output[:-2] + "."
            # with open(f'{args.output}/{args.task}_v1.txt', 'a') as f:
            #     f.write("<IMG>" + abspath + "<IMG>" + txt_output + "\n")
        # if args.output:
        #     with open(f'{args.output}/{args.task}_{args.num_chunks}_{args.chunk_idx}.json', 'w') as output_file:
        #         json.dump(segment_json, output_file)
    else:
        raise ValueError("No Input Given")