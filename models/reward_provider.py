import os
import json
import numpy as np
import torch
from models.networks import ActionProcessor, StateProcessor
from models.preference_transformer import PreferenceTransformer
from utils.texts import file_to_str, import_module_from_string, SentenceEmbedder
from utils.caption import get_caption


class RewardProvider:
    def __init__(self, args, device):
        self.args = args
        self.device = device

    def compute_reward(self, obs, actions, timesteps, descriptions):
        raise NotImplementedError()


class PreferenceModel(RewardProvider):
    def __init__(self, args, device):
        super().__init__(args, device)
        rm_config = json.load(open(os.path.join(args.pretrain_path, "config.json"), "r"))
        state_processor = StateProcessor(input_shape=rm_config["obs_shape"], n_embed=rm_config["n_embed"])
        action_processor = ActionProcessor(num_actions=rm_config["num_actions"], n_embed=rm_config["n_embed"])
        reward_model = PreferenceTransformer(state_processor, action_processor, max_seq_len=1000, n_layers=rm_config["n_layers"], n_embed=rm_config["n_embed"], n_headers=rm_config["n_headers"], dropout=0.0).to(device).eval()
        reward_model.load_state_dict(torch.load(os.path.join(args.pretrain_path, "best_model.pt")))
        self.reward_model = reward_model
        self.forward_batch_size = 256

    def compute_reward(self, obs, actions, timesteps, descriptions):
        output_rewards = []
        actions = actions[:, :-1]
        for i in range(0, obs.shape[0], self.forward_batch_size):
            output_rewards.append(self.reward_model.compute_reward(obs[i : i + self.forward_batch_size], actions[i : i + self.forward_batch_size], timesteps[i : i + self.forward_batch_size])[0])
        output_rewards = torch.cat(output_rewards, dim=0)
        return output_rewards


class CodingReward(RewardProvider):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.pretrain_path = args.pretrain_path
        self.reward_model_horizon = args.reward_model_horizon
        self.code = file_to_str(os.path.join(self.pretrain_path, "code-final.py"))
        self.code_header = file_to_str("prompts/coding/code_header.py")
        self.module = import_module_from_string("background", f"{self.code_header}{self.code}")

    def compute_reward(self, obs, actions, timesteps, descriptions):
        obs_np, actions_np = obs.cpu().numpy(), actions.cpu().numpy()
        cur_obs, past_obs, past_actions = obs_np[:, -1], obs_np[:, :-1], actions_np[:, :-1]
        rewards = []
        for p1, p2, p3 in zip(cur_obs, past_obs, past_actions):
            rewards.append(self.module.compute_reward(p1, p2, p3)[0])
        rewards = torch.from_numpy(np.array(rewards)).to(obs.device)
        return rewards


class GoalReward(RewardProvider):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.pretrain_path = args.pretrain_path
        self.lm_model_name = args.lm_model_name
        self.subgoal_reward_threshold = args.subgoal_reward_threshold
        self.st_forward_batch = 64
        self.st_model = SentenceEmbedder(device, model_name=self.lm_model_name).to(device)
        self.st_model.requires_grad_(False)
        self.goals_idx = json.load(open(os.path.join(self.pretrain_path, "goal_library.json"), "r"))
        self.key_embeddings = np.load(os.path.join(self.pretrain_path, "key_embeddings.npy"), allow_pickle=True)
        self.goals_embeddings = np.load(os.path.join(self.pretrain_path, "goals_embeddings.npy"), allow_pickle=True)

    def compute_reward(self, obs, actions, timesteps, descriptions):
        query_desc = [get_caption(self.args.env_type, d[:-1], a[:-2]).strip() for d, a in zip(descriptions, actions)]
        query_embeddings = []
        for i in range(0, len(query_desc), self.st_forward_batch):
            query_embeddings.append(self.st_model.encode(query_desc[i : i + self.st_forward_batch]))
        query_embeddings = torch.cat(query_embeddings, dim=0).cpu().numpy()

        similarities = self.cosine_similarity(query_embeddings, self.key_embeddings)
        topk_idx = np.argsort(similarities, axis=1)[:, -3:]
        selected_goals = [np.concatenate([self.goals_idx[i] for i in idx]) for idx in topk_idx]
        selected_goals_embeddings = [self.goals_embeddings[idx] for idx in selected_goals]

        combined_desc = [get_caption(self.args.env_type, d[-2:], a[-2:-1]).strip() for d, a in zip(descriptions, actions)]
        desc_embedding = []
        for i in range(0, len(combined_desc), self.st_forward_batch):
            desc_embedding.append(self.st_model.encode(combined_desc[i : i + self.st_forward_batch]))
        desc_embedding = torch.cat(desc_embedding, dim=0).cpu().numpy()

        rewards = []
        for i in range(len(combined_desc)):
            desired_goals = selected_goals_embeddings[i]
            cos_sim = self.cosine_similarity(desc_embedding[i].reshape(1, -1), desired_goals)[0]
            max_sim = np.max(cos_sim)
            reward = max_sim if max_sim > self.subgoal_reward_threshold else 0.0
            rewards.append(reward)
        rewards = torch.tensor(rewards).to(self.device)
        return rewards

    def cosine_similarity(self, query, key):
        values = np.matmul(query, key.T) / (np.linalg.norm(query, axis=1, keepdims=True) * np.linalg.norm(key, axis=1, keepdims=True).T)
        return values
