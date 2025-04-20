import json
import os
from collections import defaultdict, Counter
from pathlib import Path
from tqdm import tqdm


def write_json_file(input_data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(input_data, f, ensure_ascii=False, indent=4)


def create_step_weighted_dataset(input_file, target_model_response_file, save_dir):
    with open(target_model_response_file, 'r', encoding='utf-8') as file:
        target_model_response_verified = json.load(file)

    target_model_response_score_dict = defaultdict(float)
    target_model_zero_score_count = 0
    target_model_one_score_count = 0
    for data in target_model_response_verified:
        prompt = data['prompt']
        rule_scores = data['rule_scores']
        all_generated_responses = data['all_generated_responses']
        target_model_response_score_dict[prompt] = sum(rule_scores) / len(rule_scores)
        if sum(rule_scores) / len(rule_scores) == 0:
            target_model_zero_score_count += 1
        elif sum(rule_scores) / len(rule_scores) == 1:
            target_model_one_score_count += 1
    print(f"target_model_zero_score_count: {target_model_zero_score_count}, target_model_one_score_count: {target_model_one_score_count}")

    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print(f"raw data size: {len(data)}")

    first_rollout_allright_index_statis = []
    conversations = []
    for item in data:
        prompt = item['prompt']
        response = item['answer']
        rule_scores = item['rule_scores']

        assert prompt in target_model_response_score_dict
        baseline_score = target_model_response_score_dict[prompt]
        if baseline_score == 1:
            continue

        step_scores = [1-baseline_score]
        for rule_scores_i in rule_scores:
            step_score = 1 - (sum(rule_scores_i) / len(rule_scores_i))
            step_scores.append(step_score)

        if 0 in step_scores:
            first_rollout_allright_index_statis.append(step_scores.index(0))

        conversation = [{'from':'human', 'value':prompt}, {'from': 'gpt', 'value': response}]
        conversations.append({'conversations': conversation, 'step_scores': step_scores})

    first_rollout_allright_index_statis = Counter(first_rollout_allright_index_statis)
    print(f"first_rollout_allright_index_statis: {dict(first_rollout_allright_index_statis)}, total: {sum(first_rollout_allright_index_statis.values())}")

    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'openmathinstruct2_validation_with_step_score_v2.json'
    write_json_file(conversations, save_path)


if __name__ == "__main__":
    input_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/FuseChat-3.0-DPO-Data/all_step_rollout_rule_verified.json'
    target_model_response_file = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/FuseChat-3.0-DPO-Data/FuseChat-Llama-3.2-3B-SFT-sampling/completion_with_rm_score/all_generated_response_rule_verified.json'
    save_dir = '/GLOBALFS/gznwp_3/qxj/shitianyuan/data/FuseChat-3.0-DPO-Data'

    # create_step_weighted_dataset(input_file, target_model_response_file, save_dir)

    # from transformers import AutoTokenizer

    # tokenizer_path = '/GLOBALFS/gznwp_3/qxj/models/FuseChat-Llama-3.2-3B-SFT'
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # prompt = 'my name is'
    # messages = [{"role": "user", "content": prompt}]
    # prompt = tokenizer.apply_chat_template(
    #                 messages,
    #                 tokenize=False,
    #                 add_generation_prompt=True
    #             )

    # print(repr(prompt))
    
    # print('you are you\n\n'.split('\n\n'))

    def get_split_end_indices(steps, n):
        length = len(steps)
        # 每份的基本长度
        base_size = length // n
        # 前 remainder 份会多一个元素
        remainder = length % n

        ends = []
        end = 0
        for i in range(n):
            # 当前份的大小（前 remainder 份多一个）
            current_size = base_size + (1 if i < remainder else 0)
            end += current_size
            ends.append(end)  # 切分点是当前份的“结尾索引”（非包含）

        return ends

    completion_steps = ['step1', 'step2', 'step3', 'step4', 'step5', 'step6', 'step7']
    n = 3

    part_size = len(completion_steps) // n
    remainder = len(completion_steps) % n
    parts = []
    start_index = 0
    for i in range(n):
        # 如果有多余的元素，当前部分需要多一个元素
        end_index = start_index + part_size + (1 if i < remainder else 0)
        parts.append('\n\n'.join(completion_steps[start_index:end_index]))
        start_index = end_index  # 更新下一个部分的起始索引

    print('\n\n'.join(completion_steps[:1]))