from argparse import ArgumentParser
from distutils.util import strtobool


def add_exp_arguments(parser: ArgumentParser):
    parser.add_argument("--alg", type=str, default="goal", help="the algorithm to use")
    parser.add_argument("--env", type=str, default="BabyAI-Text-GoToLocal-RedBall-S20", help="the id of the environment")
    parser.add_argument("--env_note", type=str, default=None, help="extra environment note")
    parser.add_argument("--alg_note", type=str, default=None, help="extra algorithm note")
    parser.add_argument("--precollect", action="store_true", help="collect data during evaluation")
    parser.add_argument("--seed", type=int, default=42, help="seed of the experiment")
    parser.add_argument("--wandb", default=False, action="store_true", help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb_project_name", type=str, default="CleanRL-BabyAI", help="the wandb's project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="the entity (team) of wandb's project")


def add_minigrid_arguments(parser: ArgumentParser):
    parser.add_argument("--total_timesteps", type=int, default=5000000, help="total timesteps of the experiments")
    parser.add_argument("--num_envs", type=int, default=8, help="the number of parallel game environments")
    parser.add_argument("--num_steps", type=int, default=128, help="the number of steps to run in each environment per policy rollout")

    parser.add_argument("--eval_freq", type=int, default=20000, help="the interval between two consecutive evaluations")
    parser.add_argument("--num_eval_episodes", type=int, default=50, help="the interval between two consecutive evaluations")


def add_crafter_arguments(parser: ArgumentParser):
    parser.add_argument("--total_timesteps", type=int, default=2000000, help="total timesteps of the experiments")
    parser.add_argument("--num_envs", type=int, default=16, help="the number of parallel game environments")
    parser.add_argument("--num_steps", type=int, default=256, help="the number of steps to run in each environment per policy rollout")

    parser.add_argument("--eval_freq", type=int, default=20000, help="the interval between two consecutive evaluations")
    parser.add_argument("--num_eval_episodes", type=int, default=50, help="the interval between two consecutive evaluations")


def add_ppo_arguments(parser: ArgumentParser):
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="the learning rate of the optimizer")
    parser.add_argument("--anneal_lr", default=False, action="store_true", help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="the lambda for the general advantage estimation")
    parser.add_argument("--num_minibatches", type=int, default=4, help="the number of mini-batches")
    parser.add_argument("--update_epochs", type=int, default=4, help="the K epochs to update the policy")
    parser.add_argument("--norm_adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggles advantages normalization")
    parser.add_argument("--clip_coef", type=float, default=0.1, help="the surrogate clipping coefficient")
    parser.add_argument("--clip_vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="coefficient of the entropy")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="coefficient of the value function")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
    parser.add_argument("--target_kl", type=float, default=None, help="the target KL divergence threshold")


def add_rnd_arguments(parser: ArgumentParser):
    parser.add_argument("--rnd_output_dim", type=int, default=512, help="Intrinsic reward discount rate")
    parser.add_argument("--update_proportion", type=float, default=0.25, help="proportion of exp used for predictor update")
    parser.add_argument("--int_coef", type=float, default=1.0, help="coefficient of extrinsic reward")
    parser.add_argument("--ext_coef", type=float, default=2.0, help="coefficient of intrinsic reward")
    parser.add_argument("--int_gamma", type=float, default=0.99, help="Intrinsic reward discount rate")


def add_preference_arguments(parser: ArgumentParser):
    parser.add_argument("--pretrain_path", type=str, required=True, help="the path to the reward model")
    parser.add_argument("--reward_model_horizon", type=int, default=5, help="the horizon of the reward model")
    parser.add_argument("--ext_coef", type=float, default=10.0, help="coefficient of intrinsic reward")


def add_coding_reward_arguments(parser: ArgumentParser):
    parser.add_argument("--pretrain_path", type=str, required=True, help="path of reward code")
    parser.add_argument("--reward_model_horizon", type=int, default=5, help="the horizon of the reward model")
    parser.add_argument("--ext_coef", type=float, default=10.0, help="coefficient of intrinsic reward")


def add_goal_reward_arguments(parser: ArgumentParser):
    parser.add_argument("--pretrain_path", type=str, required=True, help="name of the pretrained model")
    parser.add_argument("--subgoal_reward_threshold", type=float, default=0.85, help="threshold of cosine similarity")
    parser.add_argument("--lm_model_name", type=str, default="prajjwal1/bert-small", help="sentence transformer model")
    parser.add_argument("--reward_model_horizon", type=int, default=4, help="the horizon of the reward model")
    parser.add_argument("--ext_coef", type=float, default=10.0, help="coefficient of intrinsic reward")
