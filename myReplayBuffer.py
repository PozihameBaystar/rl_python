import numpy as np
import torch

# ----------------------------
# Replay Buffer（オフポリシー用）
# ----------------------------
class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int, device: torch.device):
        self.device = device
        self.size = size
        self.ptr = 0
        self.full = False

        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs_next = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.int64)
        self.rew = np.zeros((size, 1), dtype=np.float32)
        self.done = np.zeros((size, 1), dtype=np.float32)

    def __len__(self) -> int:
        return self.size if self.full else self.ptr

    def add(self, obs: np.ndarray, act: int, rew: float, next_obs: np.ndarray, done: bool) -> None:
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.obs_next[self.ptr] = next_obs
        self.done[self.ptr] = float(done)

        self.ptr += 1
        if self.ptr >= self.size:
            self.ptr = 0
            self.full = True

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, len(self), size=batch_size)

        obs = torch.tensor(self.obs[idx], device=self.device)
        act = torch.tensor(self.act[idx], device=self.device)
        rew = torch.tensor(self.rew[idx], device=self.device)
        obs_next = torch.tensor(self.obs_next[idx], device=self.device)
        done = torch.tensor(self.done[idx], device=self.device)

        return {"obs": obs, "act": act, "rew": rew, "obs_next": obs_next, "done": done}