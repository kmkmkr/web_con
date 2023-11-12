import torch
import numpy as np
from typing import NamedTuple
class RolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer:
    def __init__(self, buffer_size, n_envs, obs_shape, action_dim, device, gamma = 0.99, gae_lambda = 0.95):
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def reset(self):
        self.observations = torch.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=torch.float32)
        # self.actions = torch.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=torch.int64)
        self.actions = torch.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=torch.int64)
        self.rewards = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32)
        self.returns = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32)
        self.episode_starts = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32)
        self.values = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32)
        self.log_probs = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32)
        self.advantages = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32)
        self.pos = 0
        self.generator_ready = False

    def add(self, obs1, action, reward, episode_start, value, log_prob):
        self.observations[self.pos] = obs1
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.episode_starts[self.pos] = episode_start
        self.values[self.pos] = value.cpu().flatten()
        self.log_probs[self.pos] = log_prob.cpu()
        self.pos += 1

    def compute_returns_and_advantage(self, last_values, dones):
        last_values = last_values.cpu().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.to(torch.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    @staticmethod
    def swap_and_flatten(arr):
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def get(self, batch_size):
        indices = np.random.permutation(self.buffer_size * self.n_envs)

        if not self.generator_ready:
            self.observations = self.swap_and_flatten(self.observations)
            self.actions = self.swap_and_flatten(self.actions)
            self.values = self.swap_and_flatten(self.values)
            self.log_probs = self.swap_and_flatten(self.log_probs)
            self.advantages = self.swap_and_flatten(self.advantages)
            self.returns = self.swap_and_flatten(self.returns)
            self.generator_ready = True

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def to_torch(self, array):
        return torch.as_tensor(array, device=self.device)

    def _get_samples(
        self,
        batch_inds
    ):
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
