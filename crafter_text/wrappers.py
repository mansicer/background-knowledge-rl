import os
from collections import deque
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActionWrapper, ObservationWrapper, Wrapper
from moviepy.editor import ImageSequenceClip


class ImagePreprocessWrapper(ObservationWrapper):
    """
    The image will be transposed to (C, H, W) and normalized to [0, 1]. Only use image as observation when `image_only=True`.
    """

    def __init__(self, env, image_only=True):
        super().__init__(env)
        env_spaces = {}
        for key, space in self.observation_space.spaces.items():
            if key.endswith("image"):
                image_shape = space.shape
                image_space = spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(3, *image_shape[:2]),  # number of cells
                    dtype=np.float32,
                )
                env_spaces[key] = image_space

        self.image_only = image_only
        if self.image_only:
            self.observation_space = env_spaces["image"]
        else:
            self.observation_space = spaces.Dict({**self.observation_space.spaces, **env_spaces})

    def observation(self, obs):
        image_obs = {}
        for key, value in obs.items():
            if key.endswith("image"):
                image = value.transpose(2, 0, 1).astype(np.float32)
                image = image / 255.0
                image_obs[key] = image

        return {**obs, **image_obs} if not self.image_only else image_obs["image"]


class FrameSkipWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, num_skip: int):
        gym.ObservationWrapper.__init__(self, env)
        self.num_skip = num_skip

        self.frames = deque(maxlen=num_skip)
        self.is_dict_space = isinstance(env.observation_space, spaces.Dict)
        if self.is_dict_space:
            img_space = env.observation_space.spaces["image"]
            self.observation_space = spaces.Dict({"image": spaces.Box(low=0.0, high=1.0, shape=(num_skip * img_space.shape[0], *img_space.shape[1:]), dtype=img_space.dtype), "description": env.observation_space.spaces["description"]})
        else:
            img_space = env.observation_space
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(num_skip * img_space.shape[0], *img_space.shape[1:]), dtype=img_space.dtype)

    def observation(self, observation):
        assert len(self.frames) == self.num_skip, (len(self.frames), self.num_skip)
        if self.is_dict_space:
            return {"image": np.concatenate(self.frames, axis=0), "description": observation["description"]}
        else:
            return np.concatenate(self.frames, axis=0)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.is_dict_space:
            self.frames.append(observation["image"])
        else:
            self.frames.append(observation)
        return self.observation(observation), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.is_dict_space:
            [self.frames.append(obs["image"]) for _ in range(self.num_skip)]
        else:
            [self.frames.append(obs) for _ in range(self.num_skip)]
        return self.observation(obs), info
