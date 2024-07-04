import gymnasium as gym

import babyai_envs
import crafter_text
from babyai_envs.wrappers import ImageObsWrapper
from crafter_text.wrappers import ImagePreprocessWrapper, FrameSkipWrapper


def make_minigrid_env(env_id, seed=None, idx=-1, image_only=False):
    def env_fn():
        env = gym.make(env_id, render_mode="rgb_array")
        env = ImageObsWrapper(env, image_only=image_only)
        env.reset(seed=seed)
        return env

    return env_fn


def make_crafter_env(env_id, seed=None, idx=-1, image_only=False):
    def env_fn():
        env = gym.make(env_id, seed=seed)
        env = ImagePreprocessWrapper(env, image_only=image_only)
        # env = FrameSkipWrapper(env, num_skip=4)
        env.reset(seed=seed)
        return env

    return env_fn


def make_vector_envs(env_maker, env_id, num_envs, seed=0, image_only=False):
    envs = gym.vector.SyncVectorEnv([env_maker(env_id, seed + i, i, image_only=image_only) for i in range(num_envs)])
    return envs
