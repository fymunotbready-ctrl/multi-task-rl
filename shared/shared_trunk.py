"""Shared-trunk model: per-task projection (obs->64) -> SHARED trunk
64->64->64 -> per-task heads (SB3 heads with net_arch=[]).
Routing: width 9 = joint (one-hot selects projection, per-sample);
otherwise fixed single-task mode."""
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

TRUNK_DIM = 64
TASKS = {"basketball": 6, "driving": 5, "aiming": 6}
TASK_NAMES = list(TASKS.keys())
N_TASKS = len(TASK_NAMES)
OBS_PAD = max(TASKS.values())
JOINT_DIM = OBS_PAD + N_TASKS


class SharedTrunk(nn.Module):
    def __init__(self, dim=TRUNK_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.Tanh(),
            nn.Linear(dim, dim), nn.Tanh(),
            nn.Linear(dim, dim), nn.Tanh())

    def forward(self, x):
        return self.net(x)


class TrunkFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, tasks: dict, active_task: str):
        super().__init__(observation_space, TRUNK_DIM)
        self.tasks = dict(tasks)
        self.active_task = active_task
        assert active_task in self.tasks
        self.joint_dim = max(self.tasks.values()) + len(self.tasks)
        self.proj = nn.ModuleDict({
            name: nn.Linear(dim, TRUNK_DIM) for name, dim in self.tasks.items()})
        self.trunk = SharedTrunk()

    def forward(self, obs):
        if obs.shape[-1] == self.joint_dim:
            xs = obs[:, :OBS_PAD]
            ids = obs[:, OBS_PAD:].argmax(dim=1)
            out = torch.zeros(obs.shape[0], TRUNK_DIM, device=obs.device)
            for i, name in enumerate(TASK_NAMES):
                mask = ids == i
                if mask.any():
                    z = torch.tanh(self.proj[name](xs[mask, :self.tasks[name]]))
                    out[mask] = self.trunk(z)
            return out
        return self.trunk(torch.tanh(self.proj[self.active_task](obs)))


def encode_obs(obs, task_idx):
    padded = np.zeros(OBS_PAD, dtype=np.float32)
    padded[:len(obs)] = obs
    onehot = np.zeros(N_TASKS, dtype=np.float32)
    onehot[task_idx] = 1.0
    return np.concatenate([padded, onehot])


def trunk_policy_kwargs(active_task: str) -> dict:
    return dict(
        features_extractor_class=TrunkFeaturesExtractor,
        features_extractor_kwargs=dict(tasks=TASKS, active_task=active_task),
        net_arch=[], ortho_init=True)


def save_trunk(model, path: str):
    torch.save(model.policy.features_extractor.state_dict(), path)


def load_trunk(model, path: str):
    sd = torch.load(path, map_location=model.device)
    model.policy.features_extractor.load_state_dict(sd, strict=True)
