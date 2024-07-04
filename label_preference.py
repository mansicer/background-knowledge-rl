import asyncio
import datetime
import json
import os
import re
import time
import traceback
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

from utils.azure_llm import async_query_llm_azure
from utils.caption import get_caption
from utils.texts import file_to_str


async def llm_label(azure_config, env_name, d1, d2, model_name: str = "gpt-3.5-turbo"):
    d1_desc = get_caption(env_name, d1["description"], d1["action"])
    d2_desc = get_caption(env_name, d2["description"], d2["action"])

    task_prompt = file_to_str("prompts/preference_labeling/system.txt")
    environment_description = file_to_str(f"prompts/{env_name}/descriptions.txt")
    task_prompt = task_prompt.format(environment_description=environment_description)
    user_prompt_template = file_to_str("prompts/preference_labeling/user.txt")
    user_prompt = user_prompt_template.format(desc_a=d1_desc, desc_b=d2_desc)
    messages = [{"role": "system", "content": task_prompt}, {"role": "user", "content": user_prompt}]

    try:
        answer = await async_query_llm_azure(azure_config, messages, model_name=model_name)
        rank = re.compile(r"\[Rank\](.+)", re.DOTALL).search(answer)
    except:
        answer = traceback.format_exc()
        label = -1
        print(f"Query LLMs error with input \n{user_prompt}\nerror message: \n{answer}\n")
        return user_prompt, answer, label

    if rank is None:
        print(f"Cannot find rank in result: \n{answer}\n")
        label = -1
    else:
        if "B > A" in rank.group(1):
            label = 1
        elif "A > B" in rank.group(1):
            label = 0
        elif "A = B" in rank.group(1):
            label = 0.5
        else:
            print(f"Cannot find rank in result: \n{answer}\n")
            label = -1
    return user_prompt, answer, label


async def batch_llm_label(azure_config, env_name, dataset, model_name: str = "gpt-3.5-turbo"):
    ret = await asyncio.gather(*[llm_label(azure_config, env_name, d[0], d[1], model_name) for d in dataset])
    inputs, answers, labels = zip(*ret)
    return inputs, answers, labels


def sample_data(obs_data, description_data, action_data, sequence_length=5):
    # sample a set of obs and actions
    while True:
        ep_idx = np.random.choice(len(obs_data))
        obs, descriptions, actions = obs_data[ep_idx], description_data[ep_idx], action_data[ep_idx]
        if len(actions) >= sequence_length:
            break
    step_idx = np.random.choice(actions.shape[0] - sequence_length + 1)
    obs, descriptions, actions = obs[step_idx : step_idx + sequence_length + 1], descriptions[step_idx : step_idx + sequence_length + 1], actions[step_idx : step_idx + sequence_length]
    sample = dict(image=obs, description=descriptions, action=actions)
    sample["timestep"] = np.arange(step_idx, step_idx + sequence_length + 1)
    return sample


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--env", type=str, choices=["minigrid", "crafter"], default="crafter")
    parser.add_argument("--output_path", type=str, default="data")
    parser.add_argument("--dataset_size", type=int, default=5000)
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--inference_batch_size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequence_length", type=int, default=5)
    parser.add_argument("--sleep_between_queries", type=float, default=2.0)
    args = parser.parse_args()

    azure_config = json.load(open(f"cache/{args.model_name}.json", "r"))
    np.random.seed(args.seed)

    # load data
    data = np.load(os.path.join(args.dataset_path, "dataset.npy"), allow_pickle=True)
    obs_data = [d["image"] for d in data]
    action_data = [d["action"] for d in data]
    description_data = [d["description"] for d in data]
    del data  # free memory
    print(f"load {len(description_data)} data")

    print("start generate data")
    dataset = []
    for _ in range(args.dataset_size):
        d1 = sample_data(obs_data, description_data, action_data, sequence_length=args.sequence_length)
        d2 = sample_data(obs_data, description_data, action_data, sequence_length=args.sequence_length)
        dataset.append([d1, d2])

    print("start label data")
    conversations = []
    for i in tqdm(range(0, args.dataset_size, args.inference_batch_size)):
        d1, d2 = zip(*dataset[i : i + args.inference_batch_size])
        inputs, answers, llm_labels = asyncio.run(batch_llm_label(azure_config, args.env, dataset[i : i + args.inference_batch_size], model_name=args.model_name))
        for j in range(len(llm_labels)):
            dataset[i + j].append(llm_labels[j])
        conversations.extend([[d, ans] for d, ans in zip(inputs, answers)])
        if args.sleep_between_queries > 0.0:
            time.sleep(args.sleep_between_queries)

    output_path = os.path.join(args.output_path, args.env, f"{args.model_name}-{os.path.basename(args.dataset_path)}")
    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, "dataset.npy"), dataset, allow_pickle=True)
