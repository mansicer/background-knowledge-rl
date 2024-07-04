import argparse
import datetime
import json
import os
import random
import time

import gymnasium as gym
import minigrid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from torch.distributions.categorical import Categorical

import wandb
from models.networks import IntrinsicAgent
from models.reward_provider import *
from utils.config import *
from utils import logger
from utils.normalizer import RewardForwardFilter, RunningMeanStd
from utils.env_maker import make_vector_envs, make_crafter_env, make_minigrid_env
from utils.evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    add_exp_arguments(parser)
    args, other_args = parser.parse_known_args()

    # add environment arguments
    if "minigrid" in args.env.lower() or "babyai" in args.env.lower():
        add_minigrid_arguments(parser)
        args.env_type = "minigrid"
        args.log_metrics = []
    elif "crafter" in args.env.lower():
        from crafter import constants

        add_crafter_arguments(parser)
        args.env_type = "crafter"
        args.log_metrics = ["success"] + constants.achievements
    else:
        raise ValueError(f"Unsupported environment: {args.env}")

    # add ppo arguments
    add_ppo_arguments(parser)

    if args.alg == "bk-code":
        add_coding_reward_arguments(parser)
    elif args.alg == "bk-pref":
        add_preference_arguments(parser)
    elif args.alg == "bk-goal":
        add_goal_reward_arguments(parser)
    else:
        raise ValueError(f"Unsupported algorithm: {args.alg}")

    args = parser.parse_args(other_args, namespace=args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


def train(args, agent, reward_provider: RewardProvider, envs, eval_env_fn, device):
    combined_parameters = agent.parameters()
    optimizer = optim.Adam(combined_parameters, lr=args.learning_rate, eps=1e-5)
    reward_rms = RunningMeanStd()
    advantage_rms = RunningMeanStd()
    discounted_reward = RewardForwardFilter(args.gamma)

    obs_shape = envs.single_observation_space["image"].shape
    sequence_length = args.reward_model_horizon + 1

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    intrinsic_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    ext_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    int_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    descriptions = np.empty((args.num_steps, args.num_envs), dtype=object)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    initial_obs = envs.reset()[0]
    next_obs = torch.Tensor(initial_obs["image"]).to(device)
    next_description = initial_obs["description"]
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = (args.total_timesteps // args.batch_size) + 1

    # RM setup
    rm_obs = torch.zeros((args.num_envs, sequence_length) + obs_shape).to(device)
    rm_actions = torch.zeros((args.num_envs, sequence_length) + envs.single_action_space.shape, dtype=torch.long).to(device)
    rm_timesteps = torch.zeros(args.num_envs, dtype=torch.long).to(device)
    moving_obs = torch.zeros((args.num_steps, args.num_envs, sequence_length) + obs_shape).to(device)
    moving_actions = torch.zeros((args.num_steps, args.num_envs, sequence_length) + envs.single_action_space.shape, dtype=torch.long).to(device)
    moving_timesteps = torch.zeros((args.num_steps, args.num_envs, sequence_length), dtype=torch.long).to(device)

    rm_descriptions = np.empty((args.num_envs, sequence_length), dtype=object)
    moving_descriptions = np.empty((args.num_steps, args.num_envs, sequence_length), dtype=object)

    train_metrics = {}
    last_eval_step = 0
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            descriptions[step] = next_description

            # ALGO LOGIC: action logic
            with torch.no_grad():
                value_ext, value_int = agent.get_value(obs[step])
                ext_values[step], int_values[step] = (value_ext.flatten(), value_int.flatten())
                action, logprob, _, _, _ = agent.get_action_and_value(obs[step])

            actions[step] = action
            logprobs[step] = logprob

            rm_obs[rm_timesteps == 0, :] = next_obs[rm_timesteps == 0, None]
            rm_actions[rm_timesteps == 0, :] = action[rm_timesteps == 0, None]
            rm_descriptions[rm_timesteps.cpu().numpy() == 0, :] = np.array(next_description)[rm_timesteps.cpu().numpy() == 0, None]
            rm_obs[:, :-1], rm_obs[:, -1] = rm_obs[:, 1:], next_obs
            rm_actions[:, :-1], rm_actions[:, -1] = rm_actions[:, 1:], action
            rm_descriptions[:, :-1], rm_descriptions[:, -1] = rm_descriptions[:, 1:], next_description
            rm_stacking_timesteps = rm_timesteps[-1, None] - torch.arange(sequence_length - 1, -1, -1)[None, :].to(device)
            rm_stacking_timesteps = torch.max(rm_stacking_timesteps, torch.zeros_like(rm_stacking_timesteps, device=device))

            moving_obs[step] = rm_obs
            moving_actions[step] = rm_actions
            moving_timesteps[step] = rm_stacking_timesteps
            moving_descriptions[step] = rm_descriptions

            # TRY NOT TO MODIFY: execute the game and log data.
            next_observation, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            next_obs = next_observation["image"]
            next_description = next_observation["description"]
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            rm_timesteps += 1
            rm_timesteps[done] = 0
            if "episode" in info:
                train_metrics["train/episodic_return"] = np.concatenate([train_metrics.get("train/episodic_return", []), info["episode"]["r"][done]], axis=0)
                train_metrics["train/episodic_length"] = np.concatenate([train_metrics.get("train/episodic_length", []), info["episode"]["l"][done]], axis=0)
                for metric_name in set(args.log_metrics) & set(info.keys()):
                    train_metrics[f"train/{metric_name}"] = np.concatenate([train_metrics.get(f"train/{metric_name}", []), info[metric_name][done]], axis=0)

        batch_moving_obs = moving_obs.reshape(-1, *moving_obs.shape[2:])
        batch_moving_actions = moving_actions.reshape(-1, *moving_actions.shape[2:])
        batch_moving_timesteps = moving_timesteps.reshape(-1, *moving_timesteps.shape[2:])
        batch_moving_descriptions = moving_descriptions.reshape(-1, sequence_length)

        with torch.no_grad():
            computed_rewards = reward_provider.compute_reward(batch_moving_obs, batch_moving_actions, batch_moving_timesteps, batch_moving_descriptions)
            # computed_rewards, _ = reward_model.compute_reward(batch_moving_obs, batch_moving_actions, batch_moving_timesteps)
            computed_rewards = computed_rewards.reshape(args.num_steps, args.num_envs)
            intrinsic_rewards = torch.cat([intrinsic_rewards[-1:], computed_rewards], dim=0)

        intrinsic_rewards_per_step = np.array([discounted_reward.update(reward_per_step) for reward_per_step in intrinsic_rewards.cpu().data.numpy()])
        reward_rms.update(intrinsic_rewards_per_step.reshape(-1))
        intrinsic_rewards /= np.sqrt(reward_rms.var)
        shaped_rewards = (args.gamma * intrinsic_rewards[1:] - intrinsic_rewards[:-1]).data
        rewards = rewards * args.ext_coef + shaped_rewards

        # bootstrap value if not done
        with torch.no_grad():
            next_value_ext, next_value_int = agent.get_value(next_obs)
            next_value_ext, next_value_int = next_value_ext.reshape(1, -1), next_value_int.reshape(1, -1)
            ext_advantages = torch.zeros_like(rewards, device=device)
            int_advantages = torch.zeros_like(shaped_rewards, device=device)
            ext_lastgaelam = 0
            int_lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    ext_nextnonterminal = 1.0 - next_done
                    int_nextnonterminal = 1.0
                    ext_nextvalues = next_value_ext
                    int_nextvalues = next_value_int
                else:
                    ext_nextnonterminal = 1.0 - dones[t + 1]
                    int_nextnonterminal = 1.0
                    ext_nextvalues = ext_values[t + 1]
                    int_nextvalues = int_values[t + 1]
                ext_delta = rewards[t] + args.gamma * ext_nextvalues * ext_nextnonterminal - ext_values[t]
                int_delta = shaped_rewards[t] + args.gamma * int_nextvalues * int_nextnonterminal - int_values[t]
                ext_advantages[t] = ext_lastgaelam = ext_delta + args.gamma * args.gae_lambda * ext_nextnonterminal * ext_lastgaelam
                int_advantages[t] = int_lastgaelam = int_delta + args.gamma * args.gae_lambda * int_nextnonterminal * int_lastgaelam
            ext_returns = ext_advantages + ext_values
            int_returns = int_advantages + int_values

        # flatten the batch
        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_ext_advantages = ext_advantages.reshape(-1)
        b_int_advantages = int_advantages.reshape(-1)
        b_ext_returns = ext_returns.reshape(-1)
        b_int_returns = int_returns.reshape(-1)
        b_ext_values = ext_values.reshape(-1)
        # b_advantages = b_int_advantages * args.int_coef + b_ext_advantages * args.ext_coef
        b_advantages = b_ext_advantages  # MODIFY

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, new_ext_values, new_int_values = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_ext_values, new_int_values = new_ext_values.view(-1), new_int_values.view(-1)
                if args.clip_vloss:
                    ext_v_loss_unclipped = (new_ext_values - b_ext_returns[mb_inds]) ** 2
                    ext_v_clipped = b_ext_values[mb_inds] + torch.clamp(
                        new_ext_values - b_ext_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    ext_v_loss_clipped = (ext_v_clipped - b_ext_returns[mb_inds]) ** 2
                    ext_v_loss_max = torch.max(ext_v_loss_unclipped, ext_v_loss_clipped)
                    ext_v_loss = 0.5 * ext_v_loss_max.mean()
                else:
                    ext_v_loss = 0.5 * ((new_ext_values - b_ext_returns[mb_inds]) ** 2).mean()

                int_v_loss = 0.5 * ((new_int_values - b_int_returns[mb_inds]) ** 2).mean()
                # v_loss = ext_v_loss + int_v_loss
                v_loss = ext_v_loss  # MODIFY
                entropy_loss = entropy.mean()
                loss = pg_loss - entropy_loss * args.ent_coef + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(combined_parameters, args.max_grad_norm)
                optimizer.step()

        if last_eval_step == 0 or (global_step - last_eval_step) >= args.eval_freq or global_step >= args.total_timesteps:
            last_eval_step = global_step
            eval_metrics = evaluate(agent, eval_env_fn, args.num_eval_episodes, device=device, log_metrics=args.log_metrics)
            metrics = {
                "learner/learning_rate": optimizer.param_groups[0]["lr"],
                "learner/loss": loss.item(),
                "learner/ext_value_loss": ext_v_loss.item(),
                "learner/value_loss": v_loss.item(),
                "learner/policy_loss": pg_loss.item(),
                "learner/entropy_loss": entropy_loss.item(),
                "learner/approx_kl": approx_kl.item(),
                "learner/old_approx_kl": old_approx_kl.item(),
                "learner/clipfrac": np.mean(clipfracs),
                "learner/grad_norm": grad_norm.item(),
                "learner/SPS": global_step / (time.time() - start_time),
                "learner/advantage_ext_mean": b_ext_advantages.mean().item(),
                "train/int_rewards_mean": intrinsic_rewards.mean().item(),
                "train/int_rewards_min": intrinsic_rewards.min().item(),
                "train/int_rewards_max": intrinsic_rewards.max().item(),
                "train/shaped_rewards_mean": shaped_rewards.mean().item(),
                "train/shaped_rewards_min": shaped_rewards.min().item(),
                "train/shaped_rewards_max": shaped_rewards.max().item(),
                **train_metrics,
                **eval_metrics,
            }
            train_metrics.clear()
            log_str = logger.log_metrics(f"Step {global_step} stats", metrics, global_step)
            logger.info(log_str)


if __name__ == "__main__":
    args = parse_args()
    env_name = args.env + (f"-{args.env_note}" if args.env_note is not None else "")
    alg_name = f"{args.alg}" + (f"-{args.alg_note}" if args.alg_note is not None else "")
    run_name = f"{env_name}-{alg_name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-seed{args.seed}"
    run_path = os.path.join("logs", "rl", alg_name, run_name)
    args.run_path = run_path
    os.makedirs(run_path, exist_ok=True)
    logger.setup_output_file(os.path.join(run_path, "console.log"))
    logger.setup_tensorboard(os.path.join(run_path, "tensorboard"))
    if args.wandb:
        logger.setup_wandb(project=args.wandb_project_name, entity=args.wandb_entity, group=env_name, job_type=alg_name, config=vars(args), name=run_name, save_code=True)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # env setup
    if args.env_type == "minigrid":
        envs = make_vector_envs(make_minigrid_env, args.env, args.num_envs, args.seed, image_only=False)
        eval_env_fn = make_minigrid_env(env_id=args.env, image_only=True)
    elif args.env_type == "crafter":
        envs = make_vector_envs(make_crafter_env, args.env, args.num_envs, args.seed, image_only=False)
        eval_env_fn = make_crafter_env(env_id=args.env, image_only=True)
    envs = RecordEpisodeStatistics(envs)

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    obs_shape = envs.single_observation_space["image"].shape
    action_dim = envs.single_action_space.n
    agent = IntrinsicAgent(obs_shape, action_dim).to(device)

    if args.alg == "bk-code":
        reward_provider = CodingReward(args, device)
    elif args.alg == "bk-pref":
        reward_provider = PreferenceModel(args, device)
    elif args.alg == "bk-goal":
        reward_provider = GoalReward(args, device)

    train(args, agent, reward_provider, envs, eval_env_fn, device)

    # Save final models
    os.makedirs(os.path.join(run_path, "models"), exist_ok=True)
    torch.save(agent.state_dict(), os.path.join(run_path, "models", "agent.pt"))

    # Cleanup
    envs.close()
