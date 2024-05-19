#!/bin/bash

CONFIG="configs/coco/dinat/oneformer_dinat_large_bs16_100ep.yaml"
CHECKPOINT="150_16_dinat_l_oneformer_coco_100ep.pth"

# CONFIG="configs/ade20k/dinat/coco_pretrain_oneformer_dinat_large_bs16_160k_1280x1280.yaml"
# CHECKPOINT="coco_pretrain_1280x1280_150_16_dinat_l_oneformer_ade20k_160k.pth"

# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
# IFS=',' read -ra GPULIST <<< "$gpu_list"

GPULIST=(0 1 2 3)

GPUS=${#GPULIST[@]}
CHUNKS=`expr ${#GPULIST[@]} \* 2`



INPUT_DIR="data/MME_Benchmark_release_version"
OUTPUT_DIR="data/MME_Benchmark_release_version_panoptic"


for TASK in "panoptic"; do
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX % $GPUS]} python demo/demo.py \
        --config-file ${CONFIG} \
        --input ${INPUT_DIR} \
        --output ${OUTPUT_DIR} \
        --task ${TASK} \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS ${CHECKPOINT} &
        done

    wait
    # python demo/merge_json.py --output ${OUTPUT_DIR} --num-chunks $CHUNKS --task ${TASK}
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