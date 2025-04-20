#!/bin/bash
### RUN RESPONSE INFER
# source ~/.bashrc
# module unload CUDA/11.8
# module load CUDA/12.2
# conda activate vllm_zlg

PROJ_DIR=/GLOBALFS/gznwp_3/qxj/shitianyuan/Model-Fusion
cd $PROJ_DIR

INPUT_FILE=/GLOBALFS/gznwp_3/qxj/shitianyuan/data/FuseChat-3.0-DPO-Data/openmathinstruct2_validation.json
# /GLOBALFS/gznwp_3/qxj/shitianyuan/data/model_fusion_0226/openr1_math/openr1_solution_max_10_step_min_8_step_datasize_10000/solution.json
# /GLOBALFS/gznwp_3/qxj/shitianyuan/data/model_fusion_0226/openr1_math/openr1_solution_max_10_step_min_8_step_datasize_10000/openr1_solution_step_rollout/Qwen2.5-3B-Instruct/completion_with_rm_score/data_to_check.json
# /GLOBALFS/gznwp_3/qxj/shitianyuan/data/model_fusion_0226/openr1_math/openr1_solution_max_10_step_min_8_step_datasize_10000/solution.json
# /data/sty/model_fusion/zlg_1118/source_responses_top_4/source_model_response_filtered_28806_over_8_steps.json
# /GLOBALFS/gznwp_3/qxj/shitianyuan/data/model_fusion_0226/source_responses_top_4/source_model_response_filtered_28806_over_8_steps.json
MODEL=/GLOBALFS/gznwp_3/qxj/models/FuseChat-Llama-3.2-3B-SFT
# /data/wanfq/fuse3/models/Llama-3.2-3B-Instruct
# /GLOBALFS/gznwp_3/qxj/yangzy/models/Llama-3.2-3B-Instruct

DATASET_ID=FuseChat-Llama-3.2-3B-SFT-sampling
OUTPUT_DIR=/GLOBALFS/gznwp_3/qxj/shitianyuan/data/FuseChat-3.0-DPO-Data/$DATASET_ID
# /GLOBALFS/gznwp_3/qxj/shitianyuan/data/model_fusion_0226/openr1_math/openr1_solution_max_10_step_min_8_step_datasize_10000/$DATASET_ID
# /data/sty/model_fusion/zlg_1118/source_responses_top_4/$DATASET_ID

# LOG_DIR=$PROJ_DIR/log/0406
# mkdir -p $LOG_DIR
# exec > $LOG_DIR/run_math_infer_verification_pipeline.log 2>&1

SEEDS=(100 13 21 42 79)
GPUS=(0 1 2 3 4 5 6 7) # 1 2 3 4 5 6 7
NUM_SEEDS=${#SEEDS[@]}  # seed 的个数
NUM_GPUS=${#GPUS[@]}    # 可用的 GPU 数量
CHUNK_NUM=$NUM_GPUS
JOB_INDEX=0

# while [ $JOB_INDEX -lt $NUM_SEEDS ]; do
#     SEED=${SEEDS[$JOB_INDEX]}
#     # 分配 CHUNK_NUM 块数据到 GPU
#     for (( CHUNK_ID=0; CHUNK_ID<CHUNK_NUM; CHUNK_ID++ )); do
#         GPU_INDEX=$((CHUNK_ID % NUM_GPUS))
#         GPU=${GPUS[$GPU_INDEX]}  # 获取当前 GPU
#         CUDA_VISIBLE_DEVICES=$GPU python $PROJ_DIR/completion_inference/math_infer.py \
#                 --input_file $INPUT_FILE \
#                 --model_name_or_path $MODEL \
#                 --output_dir $OUTPUT_DIR \
#                 --temperature 0.9 \
#                 --seed $SEED \
#                 --chunk_num $CHUNK_NUM \
#                 --chunk_id $CHUNK_ID &
#     done
    
#     # 等待所有 GPU 的任务完成
#     wait

#     JOB_INDEX=$((JOB_INDEX + 1))
# done


### RESPONSE POST-PROCESS
POST_PROCESS_TYPE=merge_multi_seeds_response_chunks #merge_completion_chunks 
RESPONSE_DIR=$OUTPUT_DIR

# python $PROJ_DIR/completion_inference/build_dataset_v2.py \
#     --response_file_dir $RESPONSE_DIR \
#     --post_process_type $POST_PROCESS_TYPE


### COMPLETION RM SCORING
# conda activate llama_factory_sty

RM_SCORING_INPUT=$RESPONSE_DIR/all_generated_responses.json
REWARD_MODEL=/GLOBALFS/gznwp_3/qxj/models/Qwen2.5-Math-PRM-7B
# /data/chenrj/7b_model/ArmoRM-Llama3-8B-v0.1
# /GLOBALFS/gznwp_3/qxj/models/ArmoRM-Llama3-8B-v0.1
RM_SCORED_OUTPUT_DIR=$RESPONSE_DIR/completion_with_rm_score

VERIFY_MODE=Rule # PRM or Rule or LLM-as-a-judge

if [ "$VERIFY_MODE" = "Rule" ]; then
    CHUNK_NUM=1
elif [ "$VERIFY_MODE" = "PRM" ]; then
    CHUNK_NUM=$NUM_GPUS
else
    echo "Invalid VERIFY_MODE: $VERIFY_MODE"
    exit 1
fi

for ((i=0; i<CHUNK_NUM; i++))
do
    # 计算当前任务分配的 GPU 索引
    GPU_INDEX=$((i % NUM_GPUS))
    GPU=${GPUS[$GPU_INDEX]}

    # 使用 & 符号将任务放到后台运行
    CUDA_VISIBLE_DEVICES=$GPU python $PROJ_DIR/completion_inference/math_infer_verification.py \
        --input_file "$RM_SCORING_INPUT" \
        --model_name_or_path "$REWARD_MODEL" \
        --output_dir $RM_SCORED_OUTPUT_DIR \
        --verfiy_mode $VERIFY_MODE \
        --chunk_num $CHUNK_NUM \
        --chunk_id $i &
done
wait