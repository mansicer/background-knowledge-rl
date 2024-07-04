import json
import numpy as np
import torch
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics


def evaluate(agent, env_fn, num_episodes, device, log_metrics=list()):
    env = env_fn()
    env = RecordEpisodeStatistics(env)
    metrics = {}
    for _ in range(num_episodes):
        done = False
        obs, _ = env.reset()
        while not done:
            obs = torch.Tensor(obs).to(device).unsqueeze(0)
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action[0])
            done = terminated or truncated
        metrics["eval/episodic_return"] = np.concatenate([metrics.get("eval/episodic_return", []), info["episode"]["r"]], axis=0)
        metrics["eval/episodic_length"] = np.concatenate([metrics.get("eval/episodic_length", []), info["episode"]["l"]], axis=0)
        for metric_name in set(log_metrics) & set(info.keys()):
            metrics[f"eval/{metric_name}"] = np.concatenate([metrics.get(f"eval/{metric_name}", []), [info[metric_name]]], axis=0)
    env.close()
    return metrics


def evaluate_and_collect(agent, env_fn, num_episodes, device, log_metrics=list(), save_path=None):
    env = env_fn()
    env = RecordEpisodeStatistics(env)
    metrics = {}
    should_collect = save_path is not None
    data = []
    for _ in range(num_episodes):
        done = False
        step = 0

        obs_dict, _ = env.reset()

        if should_collect:
            ep_data = dict(timestep=list(), action=list(), reward=list(), terminated=list(), truncated=list(), infos=list())
            for obs_k, obs_v in obs_dict.items():
                append_list = ep_data.get(obs_k, list())
                append_list.append(obs_v)
                ep_data[obs_k] = append_list

        while not done:
            obs = torch.Tensor(obs_dict["image"]).to(device).unsqueeze(0)
            action = agent.get_action_and_value(obs)[0]  # choose stochastic action
            obs_dict, reward, terminated, truncated, info = env.step(action[0])
            done = terminated or truncated

            if should_collect:
                for obs_k, obs_v in obs_dict.items():
                    append_list = ep_data.get(obs_k, list())
                    append_list.append(obs_v)
                    ep_data[obs_k] = append_list
                ep_data.get("action").append(int(action))
                ep_data.get("reward").append(reward)
                ep_data.get("terminated").append(terminated)
                ep_data.get("truncated").append(truncated)
                ep_data.get("infos").append(info)
                ep_data.get("timestep").append(step)
            step += 1

        if should_collect:
            ep_data = {k: np.array(v) for k, v in ep_data.items()}
            data.append(ep_data)

        metrics["eval/episodic_return"] = np.concatenate([metrics.get("eval/episodic_return", []), info["episode"]["r"]], axis=0)
        metrics["eval/episodic_length"] = np.concatenate([metrics.get("eval/episodic_length", []), info["episode"]["l"]], axis=0)
        for metric_name in set(log_metrics) & set(info.keys()):
            metrics[f"eval/{metric_name}"] = np.concatenate([metrics.get(f"eval/{metric_name}", []), [info[metric_name]]], axis=0)

    if should_collect:
        np.save(save_path, data)
        json.dump({k: v.tolist() for k, v in metrics.items()}, open(save_path + ".json", "w"))

    env.close()
    return metrics
