#!/bin/bash

CONFIG="configs/coco/dinat/oneformer_dinat_large_bs16_100ep.yaml"
CHECKPOINT="150_16_dinat_l_oneformer_coco_100ep.pth"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}



INPUT_DIR="data/COCO17/train2017"
OUTPUT_DIR="playground/data/coco_segm_text/train"
DEPTH_DIR="playground/data/coco_segm_text/depth/train/depth"


for TASK in "semantic" "instance" "panoptic"; do

    python demo/merge_json.py --output ${OUTPUT_DIR} --num-chunks $CHUNKS --task ${TASK}
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python demo/demo.py \
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
    python demo/merge_json.py --output ${OUTPUT_DIR} --num-chunks $CHUNKS --task ${TASK}
done

# INPUT_DIR="data/MME_Benchmark_release_version/existence"
# OUTPUT_DIR="data/MME_Benchmark_release_version_panoptic/existence"
# # DEPTH_DIR="LMUData/images/MMStar-depth-npy"


# for TASK in "panoptic"; do

#     # python demo/merge_json.py --output ${OUTPUT_DIR} --num-chunks $CHUNKS --task ${TASK}
#     for IDX in $(seq 0 $((CHUNKS-1))); do
#         CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python demo/demo.py \
#         --config-file ${CONFIG} \
#         --input ${INPUT_DIR} \
#         --output ${OUTPUT_DIR} \
#         --task ${TASK} \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS ${CHECKPOINT} &
#         done

#     wait
#     # python demo/merge_json.py --output ${OUTPUT_DIR} --num-chunks $CHUNKS --task ${TASK}
# done



# INPUT_DIR="data/MME_Benchmark_release_version/count"
# OUTPUT_DIR="data/MME_Benchmark_release_version_panoptic/count"


# for TASK in "panoptic"; do

#     # python demo/merge_json.py --output ${OUTPUT_DIR} --num-chunks $CHUNKS --task ${TASK}
#     for IDX in $(seq 0 $((CHUNKS-1))); do
#         CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python demo/demo.py \
#         --config-file ${CONFIG} \
#         --input ${INPUT_DIR} \
#         --output ${OUTPUT_DIR} \
#         --task ${TASK} \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS ${CHECKPOINT} &
#         done

#     wait
#     # python demo/merge_json.py --output ${OUTPUT_DIR} --num-chunks $CHUNKS --task ${TASK}
# done



# INPUT_DIR="data/MME_Benchmark_release_version/position"
# OUTPUT_DIR="data/MME_Benchmark_release_version_panoptic/position"


# for TASK in "panoptic"; do

#     # python demo/merge_json.py --output ${OUTPUT_DIR} --num-chunks $CHUNKS --task ${TASK}
#     for IDX in $(seq 0 $((CHUNKS-1))); do
#         CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python demo/demo.py \
#         --config-file ${CONFIG} \
#         --input ${INPUT_DIR} \
#         --output ${OUTPUT_DIR} \
#         --task ${TASK} \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS ${CHECKPOINT} &
#         done

#     wait
#     # python demo/merge_json.py --output ${OUTPUT_DIR} --num-chunks $CHUNKS --task ${TASK}
# done



# INPUT_DIR="data/mmstar/MMStar"
# OUTPUT_DIR="data/mmstar/MMStar_panoptic"



# for TASK in "panoptic"; do

#     # python demo/merge_json.py --output ${OUTPUT_DIR} --num-chunks $CHUNKS --task ${TASK}
#     for IDX in $(seq 0 $((CHUNKS-1))); do
#         CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python demo/demo.py \
#         --config-file ${CONFIG} \
#         --input ${INPUT_DIR} \
#         --output ${OUTPUT_DIR} \
#         --task ${TASK} \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS ${CHECKPOINT} &
#         done

#     wait
#     # python demo/merge_json.py --output ${OUTPUT_DIR} --num-chunks $CHUNKS --task ${TASK}
# done

# CUDA_VISIBLE_DEVICES=0 python demo/demo.py \
# --config-file configs/coco/dinat/oneformer_dinat_large_bs16_100ep.yaml \
# --input data/mmstar/MMStar \
# --output data/mmstar/MMStar_panoptic \
# --task panoptic \
# --num-chunks 1 \
# --chunk-idx 0 \
# --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS 150_16_dinat_l_oneformer_coco_100ep.pth


# INPUT_DIR="data/mmbench/images"
# OUTPUT_DIR="data/mmbench/images_panoptic"



# for TASK in "panoptic"; do

#     # python demo/merge_json.py --output ${OUTPUT_DIR} --num-chunks $CHUNKS --task ${TASK}
#     for IDX in $(seq 0 $((CHUNKS-1))); do
#         CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python demo/demo.py \
#         --config-file ${CONFIG} \
#         --input ${INPUT_DIR} \
#         --output ${OUTPUT_DIR} \
#         --task ${TASK} \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS ${CHECKPOINT} &
#         done

#     wait
#     # python demo/merge_json.py --output ${OUTPUT_DIR} --num-chunks $CHUNKS --task ${TASK}
# done

# INPUT_DIR="data/SEED-Bench/SEED-Bench-image"
# OUTPUT_DIR="data/SEED-Bench/SEED-Bench-image_panoptic"



# for TASK in "panoptic"; do

#     # python demo/merge_json.py --output ${OUTPUT_DIR} --num-chunks $CHUNKS --task ${TASK}
#     for IDX in $(seq 0 $((CHUNKS-1))); do
#         CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python demo/demo.py \
#         --config-file ${CONFIG} \
#         --input ${INPUT_DIR} \
#         --output ${OUTPUT_DIR} \
#         --task ${TASK} \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS ${CHECKPOINT} &
#         done

#     wait
#     # python demo/merge_json.py --output ${OUTPUT_DIR} --num-chunks $CHUNKS --task ${TASK}
# done