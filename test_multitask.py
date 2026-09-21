import numpy as np
import torch
from gymnasium import spaces

from shared.shared_trunk import TASKS, TRUNK_DIM, TrunkFeaturesExtractor


extractor = TrunkFeaturesExtractor(
    spaces.Box(-np.inf, np.inf, (9,), np.float32), TASKS, "basketball"
)
obs = torch.zeros((len(TASKS), 9), dtype=torch.float32)
for idx, (name, dim) in enumerate(TASKS.items()):
    obs[idx, :dim] = torch.arange(1, dim + 1, dtype=torch.float32)
    obs[idx, 6 + idx] = 1.0

actual = extractor(obs)
expected = torch.stack(
    [
        extractor.trunk(torch.tanh(extractor.proj[name](obs[idx, :dim])))
        for idx, (name, dim) in enumerate(TASKS.items())
    ]
)
assert actual.shape == (len(TASKS), TRUNK_DIM)
assert torch.equal(actual, expected)

actual.sum().backward()
assert all(extractor.proj[name].weight.grad is not None for name in TASKS)
print("OK - joint observations route through every task projection")
