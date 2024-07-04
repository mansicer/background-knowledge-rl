import datetime
import json
import os
from argparse import ArgumentParser
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.preference_transformer import PreferenceTransformer
from models.networks import ActionProcessor, StateProcessor


class PreferenceDataset(Dataset):
    def __init__(self, data, obs_key: str = "image"):
        self.data = data
        self.obs_key = obs_key

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        d1, d2, label = self.data[index]
        s1 = self._process_data(d1)
        s2 = self._process_data(d2)
        return s1, s2, label

    def _process_data(self, sample):
        result = {}
        for key in sample.keys():
            if key not in ["mission", "description", "infos"]:
                np_sample = np.array(sample[key])
                if key in ["action", "timestep", "episode_idx"]:
                    dtype = torch.long
                else:
                    dtype = torch.float
                value = torch.tensor(np_sample, dtype=dtype)
                result[key] = value
        return result


config = dict(
    global_seed=42,
    batch_size=32,
    n_embed=128,
    n_layers=1,
    n_headers=1,
)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/llm-data/minigrid/gpt-3.5-turbo-Minigrid-dataset-0109")
    parser.add_argument("--env", type=str, choices=["minigrid", "crafter"], default="minigrid")
    parser.add_argument("--data_size", type=int, default=10000)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--log_interval", type=int, default=50)
    args = parser.parse_args()

    dataset_path = os.path.join(args.dataset_path, "dataset.npy")
    if args.env == "minigrid":
        config.update(dict(obs_shape=(3, 7, 7), num_actions=7))
    elif args.env == "crafter":
        config.update(dict(obs_shape=(3, 64, 64), num_actions=20))

    config.update(vars(args))
    args = SimpleNamespace(**config)
    np.random.seed(args.global_seed)
    torch.manual_seed(args.global_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = np.load(dataset_path, allow_pickle=True)
    data = data[: args.data_size]
    data = data[data[:, -1] != -1]
    print(data.shape)
    print(list(data[0][0].keys()))

    train_data, eval_data = train_test_split(data, test_size=args.test_size, shuffle=True, random_state=args.global_seed)

    train_dataset = PreferenceDataset(train_data)
    eval_dataset = PreferenceDataset(eval_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    state_processor = StateProcessor(input_shape=args.obs_shape, n_embed=args.n_embed)
    action_processor = ActionProcessor(num_actions=args.num_actions, n_embed=args.n_embed)
    model = PreferenceTransformer(state_processor, action_processor, max_seq_len=1000, n_layers=args.n_layers, n_embed=args.n_embed, n_headers=args.n_headers, dropout=0.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    datatime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_path = os.path.join("logs", "bk-pref", f"{'-'.join(os.path.basename(args.dataset_path).split('-')[:-2])}-{datatime_str}")
    os.makedirs(ckpt_path, exist_ok=False)
    json.dump(config, open(os.path.join(ckpt_path, "config.json"), "w"), indent=4, ensure_ascii=False)
    best_acc = 0.0
    num_iter = 0
    for epoch in range(args.epoch):
        for sample1, sample2, label in tqdm(train_loader):
            num_iter += 1

            state1 = sample1["image"].to(device)
            state2 = sample2["image"].to(device)
            action1 = sample1["action"].to(device)
            action2 = sample2["action"].to(device)
            timestep1 = sample1["timestep"].to(device)
            timestep2 = sample2["timestep"].to(device)
            label = label.to(device)

            logits, rewards = model.forward(state1, action1, timestep1, state2, action2, timestep2)
            labeled_idx = label != 0.5
            ce_loss = F.cross_entropy(logits[labeled_idx], label[labeled_idx].long())
            log_probs = torch.log_softmax(logits, dim=-1)
            smoothed_loss = (-log_probs).mean()
            loss = ce_loss + smoothed_loss * 0.3
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if num_iter % args.log_interval == 0:
                train_acc = (logits.argmax(dim=-1) == label).float().mean()
                with torch.no_grad():
                    eval_loss = 0
                    eval_pred = []
                    eval_data = []

                    for eval_sample1, eval_sample2, eval_label in eval_loader:
                        s1 = eval_sample1["image"].to(device)
                        s2 = eval_sample2["image"].to(device)
                        a1 = eval_sample1["action"].to(device)
                        a2 = eval_sample2["action"].to(device)
                        t1 = eval_sample1["timestep"].to(device)
                        t2 = eval_sample2["timestep"].to(device)
                        eval_label = eval_label.to(device)

                        eval_logits, _ = model.forward(s1, a1, t1, s2, a2, t2)
                        eval_logits = eval_logits[eval_label != 0.5]
                        eval_label = eval_label[eval_label != 0.5].long()
                        eval_loss += F.cross_entropy(eval_logits, eval_label) * eval_logits.shape[0]
                        eval_pred.append(eval_logits.argmax(dim=-1))
                        eval_data.append(eval_label)

                    eval_loss /= len(eval_dataset)
                    eval_pred = torch.cat(eval_pred, dim=0)
                    eval_data = torch.cat(eval_data, dim=0)
                    eval_acc = (eval_pred == eval_data).float().mean()

                print(f"Epoch {epoch}, Iter {num_iter % len(train_loader)}/{len(train_loader)}, Train loss {loss.item():.4f}, accuracy {train_acc.item():.4f}, Eval loss {eval_loss.item():.4f}, accuracy {eval_acc.item():.4f}")

                if best_acc < eval_acc:
                    save_path = os.path.join(ckpt_path, "best_model.pt")
                    print(f"New best eval acc {eval_acc} > {best_acc}. Save current model to {save_path}")
                    torch.save(model.state_dict(), save_path)
                    best_acc = eval_acc
