import datetime
import importlib
import json
import os
import re
import traceback
from argparse import ArgumentParser
from copy import deepcopy

import numpy as np

from utils.azure_llm import query_llm_azure
from utils.caption import get_caption_for_code
from utils.texts import file_to_str, import_module_from_string
from utils.token_usage import num_tokens_from_messages


def get_reward(code, obs, past_obs, past_actions):
    code_header = file_to_str("prompts/coding/code_header.py")
    feature_module = import_module_from_string("background", f"{code_header}{code}")
    reward, info = feature_module.compute_reward(obs, past_obs, past_actions)
    return reward, info


def parse_reply(text):
    code = re.search(r"```python\n(.*)```", text, re.DOTALL).group(1)
    return code


def request_llm_with_retries(azure_config, messages, model_name, retry_times):
    code = None
    for i in range(retry_times):
        answer = query_llm_azure(azure_config, messages, model_name=model_name)
        try:
            code = parse_reply(answer)
            break
        except:
            error_msg = traceback.format_exc()
            print(f"{error_msg} Error in calling LLM, retrying... \n")
    return code, answer


def prune_messages(messages, model_name, max_tokens, history_index=2):
    token_nums = num_tokens_from_messages(messages, model_name)
    if token_nums < max_tokens:
        return messages

    pruned_messages = deepcopy(messages)
    while token_nums > max_tokens:
        # delete a conversation pair <assistant, user>
        del pruned_messages[history_index]
        del pruned_messages[history_index]

        token_nums = num_tokens_from_messages(pruned_messages, model_name)
    return pruned_messages


def sample_data(obs_data, description_data, action_data, sample_nums, sequence_length=5):
    # sample a set of obs and actions
    obs_list, desc_list, act_list = [], [], []
    for _ in range(sample_nums):
        while True:
            ep_idx = np.random.choice(len(obs_data))
            obs, descriptions, actions = obs_data[ep_idx], description_data[ep_idx], action_data[ep_idx]
            if len(actions) >= sequence_length:
                break
        step_idx = np.random.choice(actions.shape[0] - sequence_length + 1)
        obs, descriptions, actions = obs[step_idx : step_idx + sequence_length + 1], descriptions[step_idx : step_idx + sequence_length + 1], actions[step_idx : step_idx + sequence_length]
        obs_list.append(obs)
        desc_list.append(descriptions)
        act_list.append(actions)
    return obs_list, desc_list, act_list


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--env", type=str, choices=["minigrid"], default="minigrid")
    parser.add_argument("--model_name", type=str, default="gpt-4")
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--num_iterations", type=int, default=20)
    parser.add_argument("--num_eval_samples", type=int, default=5)
    parser.add_argument("--retry_times", type=int, default=15)
    parser.add_argument("--sequence_length", type=int, default=5)
    args = parser.parse_args()

    azure_config = json.load(open(f"cache/{args.model_name}.json", "r"))
    if args.max_tokens is None:
        if "gpt-3.5-turbo" in args.model_name:
            if "16k" in args.model_name:
                args.max_tokens = 14000
            else:
                args.max_tokens = 3000
        elif "gpt-4" in args.model_name:
            if "32k" in args.model_name:
                args.max_tokens = 30000
            else:
                args.max_tokens = 6000
        else:
            raise ValueError(f"Unspecified max_tokens for the model {args.model_name}")

    # initailize prompts
    llm_task_prompt = file_to_str("prompts/coding/system_task_description.txt")
    environment_prompt = file_to_str("prompts/coding/system_environment_description.txt")
    initial_prompt = file_to_str("prompts/coding/user_initial.txt")
    error_fix_prompt = file_to_str("prompts/coding/user_error_feedback.txt")
    improve_prompt = file_to_str("prompts/coding/user_improve.txt")

    # load data
    data = np.load(os.path.join(args.dataset_path, "dataset.npy"), allow_pickle=True)
    obs_data = [d["image"] for d in data]
    action_data = [d["action"] for d in data]
    description_data = [d["description"] for d in data]
    del data  # free memory

    log_path = os.path.join("logs", "bk-code", f"{args.model_name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(log_path, exist_ok=True)

    messages = [
        {"role": "system", "content": llm_task_prompt},
        {"role": "system", "content": environment_prompt},
        {"role": "user", "content": initial_prompt},
    ]

    for sample_step in range(args.num_iterations):
        # Try to call LLM
        messages = prune_messages(messages, args.model_name, args.max_tokens)
        current_code, answer = request_llm_with_retries(azure_config, messages, model_name=args.model_name, retry_times=args.retry_times)
        if current_code is None:
            print(f"Exceed retry times, cannot parse code from the answer:\n{answer}")
            break
        messages.append({"role": "assistant", "content": answer})

        # sample a set of obs and actions
        obs_list, desc_list, act_list = sample_data(obs_data, description_data, action_data, args.num_eval_samples, sequence_length=args.sequence_length)

        # Try to compute rewards using data
        rewards, infos = [], []
        print(f"Evaluation #{sample_step}, current code: \n{current_code}")
        for obs, actions in zip(obs_list, act_list):
            reward, info = None, None
            for i in range(args.retry_times):
                try:
                    reward, info = get_reward(current_code, obs[-1], obs[:-1], actions)
                    break
                except:
                    error_msg = traceback.format_exc()
                    print(f"{error_msg}Error in computing rewards, retrying... \n")
                    messages.append({"role": "user", "content": error_fix_prompt.format(traceback_msg=error_msg)})
                    messages = prune_messages(messages, args.model_name, args.max_tokens)
                    current_code, answer = request_llm_with_retries(azure_config, messages, model_name=args.model_name, retry_times=args.retry_times)
                    if current_code is None:
                        print(f"Exceed retry times, cannot parse code from the answer:\n{answer}")
                        break
                    messages.append({"role": "assistant", "content": answer})
            rewards.append(reward)
            infos.append(info)

        if any([r is None for r in rewards]):
            print(f"Exceed retry times, cannot compute reward from code:\n{current_code}")
            break
        successful_code = current_code

        # Ask llm to improve
        captions = []
        for obs, actions, descriptions in zip(obs_list, act_list, desc_list):
            caption = get_caption_for_code(args.env, descriptions, actions)
            captions.append(caption)
        output_caption = ""
        for i, caption in enumerate(captions):
            info_str = ", ".join([f"{k}: {v:.4f}" for k, v in infos[i].items()])
            output_caption += f"Sample#{i+1}\n{caption}\nThe computed reward is {rewards[i]:.4f}. Reward details: {info_str}.\n\n"

        improve_str = improve_prompt.format(examples_description=output_caption)
        messages.append({"role": "user", "content": improve_str})

        # Save code
        open(os.path.join(log_path, f"code-step{sample_step}.py"), "w").write(successful_code)

    # Save final code
    open(os.path.join(log_path, f"code-final.py"), "w").write(successful_code)
