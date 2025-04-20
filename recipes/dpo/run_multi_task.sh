#!/bin/bash

set -e  # 出现错误时退出

PROJ_DIR=/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion
cd $PROJ_DIR

bash recipes/dpo/iterative_misalign_dpo_v2_chosen_fixed/run_iterative_misalign_dpo_v2_chosen_fixed.sh
bash recipes/dpo/iterative_updated_coarse_adv_based_misalign_dpo/run_iterative_updated_coarse_adv_based_misalign_dpo.sh

nohup bash $PROJ_DIR/benchmark_inference/iterative_misalign_dpo_v2_chosen_fixed.sh &
nohup bash $PROJ_DIR/benchmark_inference/iterative_updated_coarse_adv_based_misalign_dpo.sh &