
import json
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

import inflect
p = inflect.engine()
import argparse

parser = argparse.ArgumentParser(description='Depth order generation')
parser.add_argument('--segment-info', type=str, help='Path to the segment info', default="playground/data/eval/gqa/data/seg_images/instance.json")
parser.add_argument('--output', type=str, help='Path to the output json', default="playground/data/eval/gqa/data/seg_images")
parser.add_argument('--split', default=None, type=str)

args = parser.parse_args()

segment_info = json.load(open(args.segment_info))
images = segment_info["images"]
segment_info = segment_info["annotations"]

categories = [{"supercategory": "person", "isthing": 1, "id": 1, "name": "person"}, 
              {"supercategory": "vehicle", "isthing": 1, "id": 2, "name": "bicycle"}, 
              {"supercategory": "vehicle", "isthing": 1, "id": 3, "name": "car"}, 
              {"supercategory": "vehicle", "isthing": 1, "id": 4, "name": "motorcycle"}, 
              {"supercategory": "vehicle", "isthing": 1, "id": 5, "name": "airplane"}, 
              {"supercategory": "vehicle", "isthing": 1, "id": 6, "name": "bus"}, 
              {"supercategory": "vehicle", "isthing": 1, "id": 7, "name": "train"}, 
              {"supercategory": "vehicle", "isthing": 1, "id": 8, "name": "truck"}, 
              {"supercategory": "vehicle", "isthing": 1, "id": 9, "name": "boat"}, 
              {"supercategory": "outdoor", "isthing": 1, "id": 10, "name": "traffic light"}, 
              {"supercategory": "outdoor", "isthing": 1, "id": 11, "name": "fire hydrant"}, 
              {"supercategory": "outdoor", "isthing": 1, "id": 13, "name": "stop sign"}, 
              {"supercategory": "outdoor", "isthing": 1, "id": 14, "name": "parking meter"}, 
              {"supercategory": "outdoor", "isthing": 1, "id": 15, "name": "bench"}, 
              {"supercategory": "animal", "isthing": 1, "id": 16, "name": "bird"}, 
              {"supercategory": "animal", "isthing": 1, "id": 17, "name": "cat"}, 
              {"supercategory": "animal", "isthing": 1, "id": 18, "name": "dog"}, 
              {"supercategory": "animal", "isthing": 1, "id": 19, "name": "horse"}, 
              {"supercategory": "animal", "isthing": 1, "id": 20, "name": "sheep"}, 
              {"supercategory": "animal", "isthing": 1, "id": 21, "name": "cow"}, 
              {"supercategory": "animal", "isthing": 1, "id": 22, "name": "elephant"}, {"supercategory": "animal", "isthing": 1, "id": 23, "name": "bear"}, {"supercategory": "animal", "isthing": 1, "id": 24, "name": "zebra"}, {"supercategory": "animal", "isthing": 1, "id": 25, "name": "giraffe"}, {"supercategory": "accessory", "isthing": 1, "id": 27, "name": "backpack"}, {"supercategory": "accessory", "isthing": 1, "id": 28, "name": "umbrella"}, {"supercategory": "accessory", "isthing": 1, "id": 31, "name": "handbag"}, {"supercategory": "accessory", "isthing": 1, "id": 32, "name": "tie"}, {"supercategory": "accessory", "isthing": 1, "id": 33, "name": "suitcase"}, {"supercategory": "sports", "isthing": 1, "id": 34, "name": "frisbee"}, {"supercategory": "sports", "isthing": 1, "id": 35, "name": "skis"}, {"supercategory": "sports", "isthing": 1, "id": 36, "name": "snowboard"}, {"supercategory": "sports", "isthing": 1, "id": 37, "name": "sports ball"}, {"supercategory": "sports", "isthing": 1, "id": 38, "name": "kite"}, {"supercategory": "sports", "isthing": 1, "id": 39, "name": "baseball bat"}, {"supercategory": "sports", "isthing": 1, "id": 40, "name": "baseball glove"}, {"supercategory": "sports", "isthing": 1, "id": 41, "name": "skateboard"}, {"supercategory": "sports", "isthing": 1, "id": 42, "name": "surfboard"}, {"supercategory": "sports", "isthing": 1, "id": 43, "name": "tennis racket"}, {"supercategory": "kitchen", "isthing": 1, "id": 44, "name": "bottle"}, {"supercategory": "kitchen", "isthing": 1, "id": 46, "name": "wine glass"}, {"supercategory": "kitchen", "isthing": 1, "id": 47, "name": "cup"}, {"supercategory": "kitchen", "isthing": 1, "id": 48, "name": "fork"}, {"supercategory": "kitchen", "isthing": 1, "id": 49, "name": "knife"}, {"supercategory": "kitchen", "isthing": 1, "id": 50, "name": "spoon"}, {"supercategory": "kitchen", "isthing": 1, "id": 51, "name": "bowl"}, {"supercategory": "food", "isthing": 1, "id": 52, "name": "banana"}, {"supercategory": "food", "isthing": 1, "id": 53, "name": "apple"}, {"supercategory": "food", "isthing": 1, "id": 54, "name": "sandwich"}, {"supercategory": "food", "isthing": 1, "id": 55, "name": "orange"}, {"supercategory": "food", "isthing": 1, "id": 56, "name": "broccoli"}, {"supercategory": "food", "isthing": 1, "id": 57, "name": "carrot"}, {"supercategory": "food", "isthing": 1, "id": 58, "name": "hot dog"}, {"supercategory": "food", "isthing": 1, "id": 59, "name": "pizza"}, {"supercategory": "food", "isthing": 1, "id": 60, "name": "donut"}, {"supercategory": "food", "isthing": 1, "id": 61, "name": "cake"}, {"supercategory": "furniture", "isthing": 1, "id": 62, "name": "chair"}, {"supercategory": "furniture", "isthing": 1, "id": 63, "name": "couch"}, {"supercategory": "furniture", "isthing": 1, "id": 64, "name": "potted plant"}, {"supercategory": "furniture", "isthing": 1, "id": 65, "name": "bed"}, {"supercategory": "furniture", "isthing": 1, "id": 67, "name": "dining table"}, {"supercategory": "furniture", "isthing": 1, "id": 70, "name": "toilet"}, 
              {"supercategory": "electronic", "isthing": 1, "id": 72, "name": "tv"}, {"supercategory": "electronic", "isthing": 1, "id": 73, "name": "laptop"}, {"supercategory": "electronic", "isthing": 1, "id": 74, "name": "mouse"}, {"supercategory": "electronic", "isthing": 1, "id": 75, "name": "remote"}, {"supercategory": "electronic", "isthing": 1, "id": 76, "name": "keyboard"}, {"supercategory": "electronic", "isthing": 1, "id": 77, "name": "cell phone"}, {"supercategory": "appliance", "isthing": 1, "id": 78, "name": "microwave"}, {"supercategory": "appliance", "isthing": 1, "id": 79, "name": "oven"}, {"supercategory": "appliance", "isthing": 1, "id": 80, "name": "toaster"}, {"supercategory": "appliance", "isthing": 1, "id": 81, "name": "sink"}, {"supercategory": "appliance", "isthing": 1, "id": 82, "name": "refrigerator"}, {"supercategory": "indoor", "isthing": 1, "id": 84, "name": "book"}, {"supercategory": "indoor", "isthing": 1, "id": 85, "name": "clock"}, {"supercategory": "indoor", "isthing": 1, "id": 86, "name": "vase"}, {"supercategory": "indoor", "isthing": 1, "id": 87, "name": "scissors"}, {"supercategory": "indoor", "isthing": 1, "id": 88, "name": "teddy bear"}, {"supercategory": "indoor", "isthing": 1, "id": 89, "name": "hair drier"}, {"supercategory": "indoor", "isthing": 1, "id": 90, "name": "toothbrush"}, {"supercategory": "textile", "isthing": 0, "id": 92, "name": "banner"}, {"supercategory": "textile", "isthing": 0, "id": 93, "name": "blanket"}, {"supercategory": "building", "isthing": 0, "id": 95, "name": "bridge"}, {"supercategory": "raw-material", "isthing": 0, "id": 100, "name": "cardboard"}, {"supercategory": "furniture-stuff", "isthing": 0, "id": 107, "name": "counter"}, {"supercategory": "textile", "isthing": 0, "id": 109, "name": "curtain"}, {"supercategory": "furniture-stuff", "isthing": 0, "id": 112, "name": "door-stuff"}, {"supercategory": "floor", "isthing": 0, "id": 118, "name": "floor-wood"}, {"supercategory": "plant", "isthing": 0, "id": 119, "name": "flower"}, {"supercategory": "food-stuff", "isthing": 0, "id": 122, "name": "fruit"}, {"supercategory": "ground", "isthing": 0, "id": 125, "name": "gravel"}, {"supercategory": "building", "isthing": 0, "id": 128, "name": "house"}, {"supercategory": "furniture-stuff", "isthing": 0, "id": 130, "name": "light"}, {"supercategory": "furniture-stuff", "isthing": 0, "id": 133, "name": "mirror-stuff"}, {"supercategory": "structural", "isthing": 0, "id": 138, "name": "net"}, {"supercategory": "textile", "isthing": 0, "id": 141, "name": "pillow"}, {"supercategory": "ground", "isthing": 0, "id": 144, "name": "platform"}, {"supercategory": "ground", "isthing": 0, "id": 145, "name": "playingfield"}, {"supercategory": "ground", "isthing": 0, "id": 147, "name": "railroad"}, {"supercategory": "water", "isthing": 0, "id": 148, "name": "river"}, {"supercategory": "ground", "isthing": 0, "id": 149, "name": "road"}, {"supercategory": "building", "isthing": 0, "id": 151, "name": "roof"}, {"supercategory": "ground", "isthing": 0, "id": 154, "name": "sand"}, {"supercategory": "water", "isthing": 0, "id": 155, "name": "sea"}, {"supercategory": "furniture-stuff", "isthing": 0, "id": 156, "name": "shelf"}, {"supercategory": "ground", "isthing": 0, "id": 159, "name": "snow"}, {"supercategory": "furniture-stuff", "isthing": 0, "id": 161, "name": "stairs"}, {"supercategory": "building", "isthing": 0, "id": 166, "name": "tent"}, {"supercategory": "textile", "isthing": 0, "id": 168, "name": "towel"}, {"supercategory": "wall", "isthing": 0, "id": 171, "name": "wall-brick"}, {"supercategory": "wall", "isthing": 0, "id": 175, "name": "wall-stone"}, {"supercategory": "wall", "isthing": 0, "id": 176, "name": "wall-tile"}, {"supercategory": "wall", "isthing": 0, "id": 177, "name": "wall-wood"}, {"supercategory": "water", "isthing": 0, "id": 178, "name": "water-other"}, {"supercategory": "window", "isthing": 0, "id": 180, "name": "window-blind"}, {"supercategory": "window", "isthing": 0, "id": 181, "name": "window-other"}, {"supercategory": "plant", "isthing": 0, "id": 184, "name": "tree-merged"}, {"supercategory": "structural", "isthing": 0, "id": 185, "name": "fence-merged"}, {"supercategory": "ceiling", "isthing": 0, "id": 186, "name": "ceiling-merged"}, {"supercategory": "sky", "isthing": 0, "id": 187, "name": "sky-other-merged"}, {"supercategory": "furniture-stuff", "isthing": 0, "id": 188, "name": "cabinet-merged"}, {"supercategory": "furniture-stuff", "isthing": 0, "id": 189, "name": "table-merged"}, {"supercategory": "floor", "isthing": 0, "id": 190, "name": "floor-other-merged"}, {"supercategory": "ground", "isthing": 0, "id": 191, "name": "pavement-merged"}, {"supercategory": "solid", "isthing": 0, "id": 192, "name": "mountain-merged"}, {"supercategory": "plant", "isthing": 0, "id": 193, "name": "grass-merged"}, {"supercategory": "ground", "isthing": 0, "id": 194, "name": "dirt-merged"}, {"supercategory": "raw-material", "isthing": 0, "id": 195, "name": "paper-merged"}, {"supercategory": "food-stuff", "isthing": 0, "id": 196, "name": "food-other-merged"}, {"supercategory": "building", "isthing": 0, "id": 197, "name": "building-other-merged"}, {"supercategory": "solid", "isthing": 0, "id": 198, "name": "rock-merged"}, {"supercategory": "wall", "isthing": 0, "id": 199, "name": "wall-other-merged"}, {"supercategory": "textile", "isthing": 0, "id": 200, "name": "rug-merged"}]
categories_id = {cat["id"]: i for i, cat in enumerate(categories)}

segment_json = {}
for image in images:
    segment_json[image["file_name"]] = []

args.output = os.path.join(args.output, args.split)
if not os.path.exists(args.output):
    os.makedirs(args.output)

with open(os.path.join(args.output, f"instance.txt"), "w") as f:
    f.write("")

for seg in tqdm(segment_info):
    # 补前导0
    image_path = str(seg["image_id"]).zfill(12) + ".jpg"
    category = categories[categories_id[seg["category_id"]]]
    bbox = seg["bbox"]
    assert image_path in segment_json, f"{image_path} not in json"
    segment_json[image_path].append({"category": category["name"].split("-")[0],
                                "is_thing": category["isthing"],
                                "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]})

for image_path, seg in segment_json.items():
    if len(seg) > 0:
        class_name = {}
        for seg in segment_json[image_path]:
            assert seg["is_thing"] == 1, "Non-thing category found"
            if seg["category"] not in class_name:
                class_name[seg["category"]] = 1
            else:
                class_name[seg["category"]] += 1

        seg_answer = "The objects present in the image are: "
        for cls, v in class_name.items():
            if v == 1:
                seg_answer += cls + ", "
            else:
                seg_answer += p.number_to_words(v) + " " + p.plural_noun(cls) + ", "
        seg_answer = seg_answer[:-2] + "."
    else:
        seg_answer = "There are no detectable objects in the image"

    with open(os.path.join(args.output, f"instance.txt"), "a") as f:
        f.write("<IMG>" + image_path + "<IMG>" + seg_answer + "\n")