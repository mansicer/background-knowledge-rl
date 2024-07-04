import datetime
import importlib
import json
import os
import traceback
from argparse import ArgumentParser
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from utils.azure_llm import query_llm_azure
from utils.caption import get_caption
from utils.texts import file_to_str
from utils.token_usage import num_tokens_from_messages
from utils.texts import SentenceEmbedder


def parse_reply(text):
    goals = [s[2:] for s in text.split("\n") if s.startswith("- ")]
    if len(goals) == 0:
        raise RuntimeError(f"Cannot parse reply from LLM: {text}")
    return goals


def request_llm_with_retries(azure_config, messages, model_name, retry_times):
    goals = None
    for i in range(retry_times):
        try:
            answer = query_llm_azure(azure_config, messages, model_name=model_name)
            goals = parse_reply(answer)
            break
        except KeyboardInterrupt:
            exit(1)
        except:
            error_msg = traceback.format_exc()
            print(f"{error_msg} Error in calling LLM, retrying... \n")
    return goals, answer


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


def prune_messages(messages, model_name, max_tokens, history_index=2):
    token_nums = num_tokens_from_messages(messages, model_name)
    if token_nums < max_tokens:
        return messages

    pruned_messages = deepcopy(messages)
    while token_nums > max_tokens:
        # delete a conversation pair
        del pruned_messages[history_index]
        del pruned_messages[history_index]

        token_nums = num_tokens_from_messages(pruned_messages, model_name)
    return pruned_messages


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/episode-data/Crafter-dataset-0110")
    parser.add_argument("--env", type=str, choices=["minigrid", "crafter"], default="crafter")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--num_iterations", type=int, default=500)
    parser.add_argument("--retry_times", type=int, default=15)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--sequence_length", type=int, default=3)
    parser.add_argument("--embedding_model", type=str, default="prajjwal1/bert-small")
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

    # load data
    data = np.load(os.path.join(args.dataset_path, "dataset.npy"), allow_pickle=True)
    obs_data = [d["image"] for d in data]
    action_data = [d["action"] for d in data]
    description_data = [d["description"] for d in data]
    del data  # free memory
    print(f"load {len(description_data)} data")

    # initailize prompts
    system_prompt = file_to_str("prompts/goal_suggestion/system.txt")
    env_prompt = file_to_str(f"prompts/{args.env}/descriptions.txt")
    user_prompt_template = file_to_str("prompts/goal_suggestion/user.txt")

    log_path = os.path.join("logs", "bk-goal", f"{args.model_name}-{args.embedding_model.replace('/', '_')}-{args.env}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(log_path, exist_ok=True)

    messages = [{"role": "system", "content": system_prompt}, {"role": "system", "content": env_prompt}]

    data = []
    goals_dict = {}
    goals_idx = 0
    for sample_step in tqdm(range(args.num_iterations)):
        # sample some observations
        sample = sample_data(obs_data, description_data, action_data, sequence_length=args.sequence_length)
        caption = get_caption(args.env, sample["description"], sample["action"])

        user_prompt = user_prompt_template.format(desc=caption)
        messages.append({"role": "user", "content": user_prompt})
        messages = prune_messages(messages, args.model_name, args.max_tokens)

        # Try to call LLM
        new_goals, answer = request_llm_with_retries(azure_config, messages, model_name=args.model_name, retry_times=args.retry_times)
        messages.append({"role": "assistant", "content": answer})

        idx_list = []
        for goal in new_goals:
            goal = goal.strip()
            if goal not in goals_dict:
                goals_dict[goal] = goals_idx
                goals_idx += 1
            idx_list.append(goals_dict[goal])
        data.append([caption, idx_list])

    # Save final goals
    text_batch_size = 16
    model = SentenceEmbedder(device="cuda", model_name=args.embedding_model)

    print("Calculate keys embeddings ...")
    key_embeddings = []
    for i in tqdm(range(0, len(data), text_batch_size)):
        captions = [d[0] for d in data[i : i + text_batch_size]]
        embeddings = model.encode(captions)
        key_embeddings.append(embeddings.cpu().numpy())
    key_embeddings = np.concatenate(key_embeddings, axis=0)

    print("Calculate goals embeddings ...")
    goals_embeddings = []
    for j in tqdm(range(0, len(goals_dict), text_batch_size)):
        goals = list(goals_dict.keys())[j : j + text_batch_size]
        embeddings = model.encode(goals)
        goals_embeddings.append(embeddings.cpu().numpy())
    goals_embeddings = np.concatenate(goals_embeddings, axis=0)

    json.dump([d[0] for d in data], open(os.path.join(log_path, f"trajectory_text.json"), "w"), indent=4)
    json.dump([d[1] for d in data], open(os.path.join(log_path, f"goal_library.json"), "w"), indent=4)
    json.dump(goals_dict, open(os.path.join(log_path, f"goal_index.json"), "w"), indent=4)
    np.save(os.path.join(log_path, "key_embeddings.npy"), key_embeddings)
    np.save(os.path.join(log_path, "goals_embeddings.npy"), goals_embeddings)
