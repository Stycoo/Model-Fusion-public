import json
import os
import argparse
import tqdm
import numpy as np
import datasets
from pathlib import Path
from math_verify import parse, verify
import re

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default='', help="Path to the output generation file")
parser.add_argument("--model_name_or_path", type=str, default='', help="Path to reward model")
parser.add_argument("--output_dir", type=str, default='', help="Path to output directory")
parser.add_argument("--chunk_num", type=int, default=8)
parser.add_argument("--chunk_id", type=int, default=0)
parser.add_argument("--verfiy_mode", type=str, default='', help="PRM or Rule or LLM-as-a-judge")
parser.add_argument("--check_file", type=str, default='')

args = parser.parse_args()

def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i] # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res

def has_repeated_chars(s, threshold):
    pattern = rf"(.)\1{{{threshold},}}"  # `\1`表示匹配的第一个分组，`{threshold,}`表示重复 threshold 次及以上
    return bool(re.search(pattern, s))

with open(args.input_file, 'r') as f:
    input_data = json.load(f)
    # input_data = input_data[:1000]
    input_data_size = len(input_data)

if args.verfiy_mode == 'PRM':
    chunk_size = input_data_size // args.chunk_num
    if args.chunk_id < args.chunk_num - 1:
        input_data = input_data[args.chunk_id * chunk_size: (args.chunk_id + 1) * chunk_size]
    else:
        input_data = input_data[args.chunk_id * chunk_size: ]

    device = "auto"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_name_or_path, 
        device_map=device, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()

    for data in tqdm.tqdm(input_data):
        prompt = data["prompt"]
        completion_inputs = data["completion_inputs"]
        completion_outputs = data["completion_outputs"]

        step_scores = []
        for i, completion_input in enumerate(completion_inputs):
            _step_scores = []
            for j, completion_output in enumerate(completion_outputs[i]):
                response = completion_input.split("<|im_start|>assistant\n")[1] + completion_output
            
                data = {
                    "system": "Please reason step by step, and put your final answer within \\boxed{}.",
                    "query": prompt,
                    "response": response.split('\n\n')
                    }
                messages = [
                    {"role": "system", "content": data['system']},
                    {"role": "user", "content": data['query']},
                    {"role": "assistant", "content": "<extra_0>".join(data['response']) + "<extra_0>"},
                ]
                conversation_str = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )

                input_ids = tokenizer.encode(
                    conversation_str, 
                    return_tensors="pt", 
                    ).to(model.device)
                outputs = model(input_ids=input_ids)

                step_sep_id = tokenizer.encode("<extra_0>")[0]
                token_masks = (input_ids == step_sep_id)
                step_reward = make_step_rewards(outputs[0], token_masks)
                print(step_reward)  # eg: [[1.0, 0.1904296875, 0.9765625, 1.0]]
                _step_scores.append(step_reward[0])

            step_scores.append(_step_scores)

        data["prm_step_scores"] = step_scores

    file_name = f"chunk_{args.chunk_id}.json"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(os.path.join(output_dir, file_name), 'w') as f:
        json.dump(input_data, f, indent=4)

if args.verfiy_mode == 'Rule':
    step_rollout_acc_statis = []
    step_rollout_acc_zero_ratio = []
    step_rollout_acc_zero_step_num = []
    data_need_check = []

    for data in tqdm.tqdm(input_data):
        prompt = data["prompt"]
        answer = data['answer']
        completion_inputs = data["completion_inputs"]
        completion_outputs = data["completion_outputs"]

        answer_verified = []
        max_step_rollout_acc = 0
        step_rollout_acc_zero_count = 0
        for i, completion_input in enumerate(completion_inputs):
            _answer_verified = []
            for j, completion_output in enumerate(completion_outputs[i]):
                # response = completion_input.split("<|im_start|>assistant\n")[1] + completion_output
                
                last_step = completion_output.split('\n\n')[-1]
                if has_repeated_chars(last_step, 20):
                    _answer_verified.append(0)
                    print(f"last_step: {last_step}\nanswer: {answer}")
                    continue
                
                extracted_answer = parse(last_step)

                # gold = parse(f"${answer}$")
                gold_last_step = answer.split('\n\n')[-1]
                gold = parse(f"${gold_last_step}$")
                
                true_or_false = verify(gold, extracted_answer)
                if true_or_false:
                    _answer_verified.append(1)
                else:
                    _answer_verified.append(0)

            answer_verified.append(_answer_verified)
            step_rollout_acc = sum(_answer_verified) / len(_answer_verified)
            if step_rollout_acc > max_step_rollout_acc:
                max_step_rollout_acc = step_rollout_acc

            if sum(_answer_verified) == 0:
                step_rollout_acc_zero_count += 1

        data["rule_scores"] = answer_verified
        step_rollout_acc_statis.append(max_step_rollout_acc)
        step_rollout_acc_zero_ratio.append(step_rollout_acc_zero_count / len(completion_inputs))
        if step_rollout_acc_zero_count / len(completion_inputs) == 1:
            step_rollout_acc_zero_step_num.append(len(completion_inputs))
            data_need_check.append({'prompt': prompt, 'answer': answer})

    file_name = 'all_step_rollout_rule_verified.json'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(os.path.join(output_dir, file_name), 'w') as f:
        json.dump(input_data, f, indent=4)
    
    # with open(os.path.join(output_dir, 'data_to_check.json'), 'w') as f:
    #     json.dump(data_need_check, f, indent=4)

    # step rollout acc 统计
    min_acc = min(step_rollout_acc_statis)
    max_acc = max(step_rollout_acc_statis)
    avg_acc = sum(step_rollout_acc_statis) / len(step_rollout_acc_statis) if step_rollout_acc_statis else 0

    print(f"""min step rollout acc: {min_acc:.2f}
    max step rollout acc: {max_acc:.2f}
    avg step rollout acc: {avg_acc:.2f}""")
    
    # step rollout acc 等于0的step在response中的占比
    min_ratio = min(step_rollout_acc_zero_ratio)
    max_ratio = max(step_rollout_acc_zero_ratio)
    avg_ratio = sum(step_rollout_acc_zero_ratio) / len(step_rollout_acc_zero_ratio) if step_rollout_acc_zero_ratio else 0  # 避免除零错误

    count_zeros = step_rollout_acc_zero_ratio.count(0)  
    count_ones = step_rollout_acc_zero_ratio.count(1)

    print(f"""step_rollout_acc_zero_ratio:
    min-{min_ratio:.2f}, max-{max_ratio:.2f}, avg-{avg_ratio:.2f}
    count(0)-{count_zeros}, count(1)-{count_ones}""")

    # step rollout acc 都为零时，这些样本的step num
    min_step_num = min(step_rollout_acc_zero_step_num)
    max_step_num = max(step_rollout_acc_zero_step_num)
    avg_step_num = sum(step_rollout_acc_zero_step_num) / len(step_rollout_acc_zero_step_num)

    print(f"""step_rollout_acc_zero_step_num min: {min_step_num}
    max: {max_step_num}
    avg: {avg_step_num}""")