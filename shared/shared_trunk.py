"""Shared-trunk multi-task model.

Per-task input projection (obs -> 64) -> SHARED trunk MLP (64->64->64)
-> per-task output heads (SB3 action_net/value_net with net_arch=[]).

Transfer rule: only the trunk weights (+ banked projections) move between
tasks. Each task's PPO instance owns fresh heads.
"""
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

TRUNK_DIM = 64

# Global registry: every model instance carries ALL projections, so a
# checkpoint saved by task A loads into task B with strict=True and simply
# banks A's projection weights alongside the trunk.
TASKS = {
    "basketball": 6,
    "driving": 5,     # v3 env (sin/cos heading + relative target)
    "aiming": 6,
}


class SharedTrunk(nn.Module):
    def __init__(self, dim=TRUNK_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.Tanh(),
            nn.Linear(dim, dim), nn.Tanh(),
            nn.Linear(dim, dim), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class TrunkFeaturesExtractor(BaseFeaturesExtractor):
    """SB3 hook: routes obs through the active task's projection, then the
    shared trunk. Output features_dim=64 feeds the per-task heads."""

    def __init__(self, observation_space, tasks: dict, active_task: str):
        super().__init__(observation_space, TRUNK_DIM)
        self.tasks = dict(tasks)
        self.active_task = active_task
        self.task_names = list(self.tasks)
        self.obs_pad = max(self.tasks.values())
        self.joint_dim = self.obs_pad + len(self.tasks)
        self.proj = nn.ModuleDict({
            name: nn.Linear(dim, TRUNK_DIM) for name, dim in self.tasks.items()
        })
        self.trunk = SharedTrunk()

    def forward(self, obs):
        if obs.shape[-1] == self.joint_dim:
            task_ids = obs[:, self.obs_pad:].argmax(dim=1)
            out = obs.new_zeros((obs.shape[0], TRUNK_DIM))
            for idx, name in enumerate(self.task_names):
                mask = task_ids == idx
                if mask.any():
                    task_obs = obs[mask, :self.tasks[name]]
                    out[mask] = self.trunk(torch.tanh(self.proj[name](task_obs)))
            return out
        z = torch.tanh(self.proj[self.active_task](obs))
        return self.trunk(z)

    def forward_task(self, obs, task: str):
        """Hook for the future joint multi-task env (per-episode routing)."""
        z = torch.tanh(self.proj[task](obs))
        return self.trunk(z)


def trunk_policy_kwargs(active_task: str) -> dict:
    return dict(
        features_extractor_class=TrunkFeaturesExtractor,
        features_extractor_kwargs=dict(tasks=TASKS, active_task=active_task),
        net_arch=[],          # action_net/value_net become the per-task heads
        ortho_init=True,
    )


def save_trunk(model, path: str):
    torch.save(model.policy.features_extractor.state_dict(), path)


def load_trunk(model, path: str):
    sd = torch.load(path, map_location=model.device, weights_only=True)
    model.policy.features_extractor.load_state_dict(sd, strict=True)
