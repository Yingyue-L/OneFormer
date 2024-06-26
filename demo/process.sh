#!/bin/bash

CONFIG="../configs/coco/dinat/oneformer_dinat_large_bs16_100ep.yaml"
CHECKPOINT="../150_16_dinat_l_oneformer_coco_100ep.pth"
OUTPUT_DIR="../playground/data/vg/seg_VG_100K"
TASK="panoptic"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python demo.py \
     --config-file ${CONFIG} \
     --data-path ../playground/data/llava_v1_5_mix665k.json \
     --image-path ../playground/data \
     --input ../playground/data/vg/VG_100K \
     --output ${OUTPUT_DIR} \
     --task ${TASK} \
     --num-chunks $CHUNKS \
     --chunk-idx $IDX \
     --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS ${CHECKPOINT} &
    done

wait

python merge_json.py --output ${OUTPUT_DIR} --num-chunks $CHUNKS --task ${TASK}