import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def build_conv_nets(input_shape, output_dim, hidden_dim=512, num_linear=1):
    C, H, W = input_shape
    if H < 16 or W < 16:
        feature_dim = 64 * H * W
        networks = [
            nn.Conv2d(C, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        ]
    else:
        feature_dim = int(64 * (H // 16) * (W // 16))
        networks = [
            nn.Conv2d(C, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        ]
    if num_linear == 1:
        networks.append(nn.Linear(feature_dim, output_dim))
    else:
        networks.append(nn.Linear(feature_dim, hidden_dim))
        dims = [hidden_dim] * (num_linear - 2) + [output_dim]
        for dim in dims:
            networks.append(nn.ReLU())
            networks.append(nn.Linear(hidden_dim, dim))
    return nn.Sequential(*networks)


class PPOAgent(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        self.network = build_conv_nets(obs_shape, 512)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(512, 512), std=0.1),
            nn.ReLU(),
            layer_init(nn.Linear(512, action_dim), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(512, 512), std=0.1),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=0.01),
        )

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def select_action(self, x):
        hidden = self.network(x)
        logits = self.actor(hidden)
        return torch.argmax(logits, dim=-1)


class IntrinsicAgent(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        self.network = build_conv_nets(obs_shape, 512)
        self.extra_layer = nn.Sequential(layer_init(nn.Linear(512, 512), std=0.1), nn.ReLU())
        self.actor = nn.Sequential(
            layer_init(nn.Linear(512, 512), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(512, action_dim), std=0.01),
        )
        self.critic_ext = layer_init(nn.Linear(512, 1), std=0.01)
        self.critic_int = layer_init(nn.Linear(512, 1), std=0.01)

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        features = self.extra_layer(hidden)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic_ext(features + hidden),
            self.critic_int(features + hidden),
        )

    def get_value(self, x):
        hidden = self.network(x)
        features = self.extra_layer(hidden)
        return self.critic_ext(features + hidden), self.critic_int(features + hidden)

    def select_action(self, x):
        hidden = self.network(x)
        logits = self.actor(hidden)
        return torch.argmax(logits, dim=-1)


class StateProcessor(nn.Module):
    def __init__(self, input_shape, n_embed):
        super(StateProcessor, self).__init__()
        self.input_channels = input_shape
        self.output_dim = n_embed
        self.conv_net = build_conv_nets(input_shape, n_embed, num_linear=2)

    def forward(self, x):
        bs, seq = x.shape[0], x.shape[1]
        x = self.conv_net(x.reshape(bs * seq, *x.shape[2:]))
        return x.reshape(bs, seq, -1)


class ActionProcessor(nn.Module):
    def __init__(self, num_actions, n_embed):
        super(ActionProcessor, self).__init__()
        self.num_actions = num_actions
        self.action_embedding = nn.Embedding(num_actions, n_embed)

    def forward(self, x):
        return self.action_embedding(x)


class RNDModel(nn.Module):
    def __init__(self, obs_shape, output_size):
        super().__init__()
        self.output_size = output_size

        self.predictor = nn.Sequential(
            build_conv_nets(obs_shape, 512),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, output_size)),
        )

        # Target network
        self.target = build_conv_nets(obs_shape, output_size)

        # target network is not trainable
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)
        return predict_feature, target_feature
