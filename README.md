# Multi-Task RL Portfolio

One shared neural network (PPO / Stable-Baselines3) trained to solve three
distinct 2D physics control tasks. Built entirely on a phone (Termux +
GitHub) with free Google Colab compute. Zero budget.

## Results (held-out seeds, deterministic policy)

| Task         | Single-skill model | Shared multi-task model |
|--------------|--------------------|-------------------------|
| Basketball   | 100%               | 100%                    |
| Driving      | 100%               | 100%                    |
| Aiming       | 94%                | 84%                     |

Success = scored / reached / hit. Evaluated on seeds the agents never
trained on (`eval_all.py`, `eval_multitask.py`).

## Architecture

- Each task is a custom Gymnasium env with a 6-dim observation and a
  2-dim continuous action (deliberately uniform so one network fits all).
- `envs/meta_env.py`: each episode samples one task at random; a one-hot
  task ID is appended to the observation (9-dim) so the shared policy
  knows which skill it's playing.
- Driving and aiming rewards are scaled by 0.2 inside the meta-env so no
  task dominates the shared gradient.

## Key lessons (the debugging story)

1. **Sparse rewards fail silently.** First basketball run converged to
   -0.110 reward: the agent gave up and shot lazily. Fix: shape rewards
   by closest approach.
2. **Progress rewards can be gamed.** Driving's "distance closed" reward
   paid more for *hovering near* the target than for reaching it (8%
   true success at 0.914 avg reward — the average lied). Fix: terminal
   bonus (5.0) that beats any farming strategy.
3. **Multi-task needs task IDs.** Shared model v1 without a one-hot task
   vector scored 0%/4%/2% — the policy couldn't tell which game it was
   playing. v2 with task conditioning + reward normalization: 100/100/84.
4. **Trust success rate, never shaped-reward averages.**

## Reproduce

1. Clone, then `pip install -r requirements.txt` (Colab has PyTorch preinstalled).
2. Train any 2D skill: `python training/train_basketball.py` (same for driving/aiming/multitask).
3. Validate the 2D skills: `python eval_all.py` / `python eval_multitask.py`.
4. Smoke-test the reference-free ragdoll: `python smoke_test_ragdoll.py --steps 1000 --check-env`.

### Squat imitation

```bash
python motions/generate_squat.py
python training/train_squat.py --steps 100000
python eval_squat.py --episodes 50 --visual assets/squat_eval.png
```

The generated motion contains 120 frames of 28 joint-angle targets. In
reference mode, each normalized policy action becomes a small offset from the
current reference pose and PyBullet's position controller supplies PD balance.
This avoids asking PPO to discover floating-base balance directly from raw
torques; reference-free ragdoll mode retains the original torque controls.

Gate 3 evaluation used 50 fixed-seed, deterministic episodes. The trained model
completed 50/50 motions (100%), with 0.664022 mean per-frame imitation reward
and 120.00/120 mean/reference episode length. `assets/squat_eval.png` is the
saved side-view frame strip.

### Optional C++ reward

The environment automatically uses the pybind11 reward extension when it is
available and otherwise keeps the NumPy implementation. Build and verify it:

```bash
python3 -m pip install pybind11
./build_reward_cpp.sh
python3 test_reward.py
python3 benchmark_reward.py --calls 100000 --steps 10000
```

The build script runs the equivalent of:

```bash
c++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) \
  reward.cpp -o reward_cpp$(python3-config --extension-suffix)
```

## Roadmap

Squat imitation is available. Command-selected motions and the live interface
remain future phases.

## Honest limitations

- All tasks are 2D, single-episode, fully observed toy physics — not
  image-based or partially observable RL.
- "84% on aiming" means the shared model still misses roughly 1 in 6 moving targets.
- The command interface (upcoming) maps fixed strings to trained motions;
  it is not open-ended language understanding.
- The squat gate covers one generated motion in PyBullet DIRECT mode; GUI playback,
  command mapping, multiple motions, and a live UI are not included.

## Stack

Python · Gymnasium · Stable-Baselines3 (PPO) · PyTorch · PyBullet · NumPy ·
GitHub Actions-free, phone-built workflow (Termux + Colab).
