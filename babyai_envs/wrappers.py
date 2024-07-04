import os
from collections import deque
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActionWrapper, ObservationWrapper, Wrapper
from minigrid.core.actions import Actions
from minigrid.core.constants import COLOR_TO_IDX, DIR_TO_VEC, OBJECT_TO_IDX
from moviepy.editor import ImageSequenceClip


class FullyObsWrapper(ObservationWrapper):
    def __init__(self, env, append_obs=False):
        super().__init__(env)

        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype=np.uint8,
        )

        self.append_obs = append_obs
        if not self.append_obs:
            self.observation_space = spaces.Dict({**self.observation_space.spaces, "image": new_image_space})
        else:
            self.observation_space = spaces.Dict({**self.observation_space.spaces, "full_image": new_image_space})

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], env.agent_dir])

        return {**obs, "image": full_grid} if not self.append_obs else {**obs, "full_image": full_grid}


class PartiallyObsWrapper(ObservationWrapper):
    def __init__(self, env, append_obs=False):
        env = FullyObsWrapper(env, append_obs)
        super().__init__(env)
        self.sight_range = self.env.unwrapped.agent_view_size
        self.observation_mask = np.zeros((self.env.width, self.env.height), dtype=bool)

        self.append_obs = append_obs
        if self.append_obs:
            self.observation_space.spaces["partial_image"] = self.observation_space.spaces["full_image"]

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.observation_mask.fill(False)
        return self.observation(obs), info

    def observation(self, obs):
        grid = self.unwrapped.grid.encode()
        grid[self.unwrapped.agent_pos[0]][self.unwrapped.agent_pos[1]] = np.array([OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], self.unwrapped.agent_dir])
        agent_positon = np.array(np.where(grid[:, :, 0] == OBJECT_TO_IDX["agent"])).reshape(2)
        agent_direction = grid[agent_positon[0], agent_positon[1], 2]

        # Compute the sight range in front of the agent
        direction_vec = DIR_TO_VEC[agent_direction]
        for i in range(self.sight_range):
            for j in range((self.sight_range + 1) // 2):
                pos1 = agent_positon + direction_vec * i + np.array([direction_vec[1], direction_vec[0]]) * j
                pos2 = agent_positon + direction_vec * i - np.array([direction_vec[1], direction_vec[0]]) * j
                if pos1[0] in range(self.env.width) and pos1[1] in range(self.env.height):
                    self.observation_mask[pos1[0], pos1[1]] = True
                if pos2[0] in range(self.env.width) and pos2[1] in range(self.env.height):
                    self.observation_mask[pos2[0], pos2[1]] = True

        grid[~self.observation_mask] = OBJECT_TO_IDX["unseen"]
        return {**obs, "image": grid} if not self.append_obs else {**obs, "partial_image": grid}


class ImageObsWrapper(ObservationWrapper):
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
                image[0, :, :] /= 10.0
                image[1, :, :] /= 5.0
                image[2, :, :] /= 3.0
                image_obs[key] = image

        return {**obs, **image_obs} if not self.image_only else image_obs["image"]


class FrameStackWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, num_stack: int):
        """Observation wrapper that stacks the observations in a rolling manner.

        Args:
            env (Env): The environment to apply the wrapper
            num_stack (int): The number of frames to stack
        """
        gym.ObservationWrapper.__init__(self, env)

        self.num_stack = num_stack

        self.frames = deque(maxlen=num_stack)
        obs_shape = env.observation_space.shape
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(num_stack * obs_shape[0], *obs_shape[1:]), dtype=np.float32)

    def observation(self, observation):
        """Converts the wrappers current frames to lazy frames.

        Args:
            observation: Ignored

        Returns:
            :class:`LazyFrames` object for the wrapper's frame buffer,  :attr:`self.frames`
        """
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        return np.concatenate(self.frames, axis=0)

    def step(self, action):
        """Steps through the environment, appending the observation to the frame buffer.

        Args:
            action: The action to step through the environment with

        Returns:
            Stacked observations, reward, terminated, truncated, and information from the environment
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation)
        return self.observation(None), reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment with kwargs.

        Args:
            **kwargs: The kwargs for the environment reset

        Returns:
            The stacked observations
        """
        obs, info = self.env.reset(**kwargs)

        [self.frames.append(obs) for _ in range(self.num_stack)]

        return self.observation(None), info


class GoToActionWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(3)

    def action(self, action):
        if action == 0:
            return Actions.left
        elif action == 1:
            return Actions.right
        else:
            return Actions.forward
