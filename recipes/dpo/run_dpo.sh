#!/bin/bash

# source /opt/conda/etc/profile.d/conda.sh
# conda activate /nas-wulanchabu/miniconda3/envs/tianyuan_llama_factory/

### DATA PREPARE
# DATA_PREPARE_BASE=/nas-wulanchabu/shitianyuan.sty/alignment-handbook
# cd $DATA_PREPARE_BASE

# bash completion_inference/run_completion_infer_scoring_pipeline.sh

# source ~/.bashrc
# conda activate llama_factory_sty

### DPO TRAINING
PROJ_DIR=/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion
cd $PROJ_DIR

LOG_DIR=$PROJ_DIR/log/0409
mkdir -p $LOG_DIR
exec > $LOG_DIR/run_openmathinstruct2_completion_dpo_iter_forward_2.log 2>&1

CUDA_VISIBLE_DEVICES=0,1,2,3 FORCE_TORCHRUN=1 llamafactory-cli train $PROJ_DIR/recipes/dpo/llama3_full_dpo_ds3.yaml