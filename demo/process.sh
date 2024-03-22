#!/bin/bash

CONFIG="../configs/coco/dinat/oneformer_dinat_large_bs16_100ep.yaml"
CHECKPOINT="../150_16_dinat_l_oneformer_coco_100ep.pth"

DATA_DIR="../playground/data/coco/train2017"
INPUT_DIR=${DATA_DIR}
OUTPUT_DIR="../playground/data/coco_segm_text/train"
DEPTH_DIR=${DATA_DIR}/depth
TASK="instance"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python demo.py \
     --config-file ${CONFIG} \
     --input ${INPUT_DIR} \
     --output ${OUTPUT_DIR} \
     --task ${TASK} \
     --num-chunks $CHUNKS \
     --chunk-idx $IDX \
     --depth ${DEPTH_DIR} \
     --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS ${CHECKPOINT} &
    done

wait

python merge_json.py --output ${OUTPUT_DIR} --num-chunks $CHUNKS --task ${TASK}
