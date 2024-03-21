# OneFormer Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SHI-Labs/OneFormer/blob/main/colab/oneformer_colab.ipynb) [![Huggingface space](https://img.shields.io/badge/🤗-Huggingface%20Space-cyan.svg)](https://huggingface.co/spaces/shi-labs/OneFormer)

- Pick a model and its config file from. For example, `configs/ade20k/swin/oneformer_swin_large_IN21k_384_bs16_160k.yaml`.
- We provide `demo.py` that is able to demo builtin configs.
- You need to specify the `task` token value during inference, The outputs will be saved accordingly in the specified `OUTPUT_DIR`:
  - `panoptic`: Panoptic, Semantic and Instance Predictions when the value of `task` token is `panoptic`.
  - `instance`: Instance Predictions when the value of `task` token is `instance`.
  - `semantic`: Semantic Predictions when the value of `task` token is `semantic`.
  - >Note: You can change the outputs to be saved on line 60 in [predictor.py](predictor.py).

```bash
export task=panoptic

cd demo
CUDA_VISIBLE_DEVICES=0,1,2,3 bash process.sh

cd ..
CUDA_VISIBLE_DEVICES=0 python demo/generate_json.py --depth playground/data/ocr_vqa/depth \
--panoptic-seg playground/data/ocr_vqa/seg_images/panoptic \
--segment-info playground/data/ocr_vqa/seg_images/panoptic.json \
--output playground/data/ocr_vqa/seg_images

CUDA_VISIBLE_DEVICES=1 python demo/generate_gt_panoptic.py --depth playground/data/coco_segm_text/depth/val/depth \
--panoptic-seg playground/data/coco/annotations/panoptic_annotations/panoptic_val2017 \
--segment-info playground/data/coco/annotations/panoptic_annotations/panoptic_val2017.json \
--output playground/data/coco/annotations/val2017

CUDA_VISIBLE_DEVICES=2 python demo/generate_gt_instance.py \
--segment-info playground/data/coco/annotations/instances_val2017.json \
--output playground/data/coco/annotations/val2017

CUDA_VISIBLE_DEVICES=2 python demo/generate_gt_semantic.py \
--semantic-seg playground/data/stuffthingmaps_trainval2017/val2017 \
--output playground/data/coco/annotations/val2017
```

For details of the command line arguments, see `demo.py -h` or look at its source code
to understand its behavior. 