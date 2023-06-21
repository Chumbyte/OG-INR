#!/bin/bash

REPO_DIR=/media/chamin8TB/OG-INR_clean/OG-INR
SRB_DATA_DIR=$REPO_DIR/data/deep_geometric_prior_data

LOG_DIR=$REPO_DIR/log/SRB_run2
DEPTH=7
for SHAPE_NAME in 'anchor' 'daratech' 'dc' 'gargoyle' 'lord_quas'
do
    echo $SHAPE_NAME
    SHAPE_PATH=$SRB_DATA_DIR/scans/${SHAPE_NAME}.ply

    python makeOctree.py --input_pc_path=$SHAPE_PATH --dataset_name=SRB --log_dir=$LOG_DIR --shape_name=$SHAPE_NAME --final_depth=$DEPTH --show_vis=False --save_pkl=True --use_wandb=False

    python train_INR.py --shape_name=$SHAPE_NAME --input_pc_path=$SHAPE_PATH --log_dir=$LOG_DIR --octree_path=$LOG_DIR/$SHAPE_NAME/${SHAPE_NAME}_depth_$DEPTH.pkl --scaling_path=$LOG_DIR/$SHAPE_NAME/scaling.npz --octree_depth=$DEPTH --inr_type=siren

done
