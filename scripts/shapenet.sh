#!/bin/bash

REPO_DIR=/media/chamin8TB/OG-INR_clean/OG-INR
SHAPENET_DATA_DIR=$REPO_DIR/data/NSP_dataset

LOG_DIR=$REPO_DIR/log/ShapeNet_run1
DEPTH=7
for SHAPE_CLASS in 'airplane' 'bench' 'cabinet' 'car' 'chair' 'display' 'lamp' 'loudspeaker' 'rifle' 'sofa' 'table' 'telephone' 'watercraft'
do
    DATASET_FOLDER=$SHAPENET_DATA_DIR/$SHAPE_CLASS
    for FILE_PATH in $DATASET_FOLDER/*
    do
        FILENAME="$(basename "$FILE_PATH")"
        SHAPE_NAME="${FILENAME%.*}"
        SHAPE_PATH=$SHAPENET_DATA_DIR/$SHAPE_CLASS/${SHAPE_NAME}.ply

        python makeOctree.py --input_pc_path=$SHAPE_PATH --dataset_name=ShapeNet --log_dir=$LOG_DIR/$SHAPE_CLASS --shape_name=$SHAPE_NAME --final_depth=$DEPTH --show_vis=False --save_pkl=True --use_wandb=False

        python train_INR.py --shape_name=$SHAPE_NAME --input_pc_path=$SHAPE_PATH --log_dir=$LOG_DIR/$SHAPE_CLASS --octree_path=$LOG_DIR/$SHAPE_CLASS/$SHAPE_NAME/${SHAPE_NAME}_depth_$DEPTH.pkl --scaling_path=$LOG_DIR/$SHAPE_CLASS/$SHAPE_NAME/scaling.npz --octree_depth=$DEPTH --inr_type=siren
    done
done