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
2. Train any skill: `python training/train_basketball.py` (same for driving/aiming/multitask).
3. Validate: `python eval_all.py` / `python eval_multitask.py`.

## Roadmap

Physics ragdoll motion agent (PyBullet humanoid, DeepMimic-style imitation
reward, command-selected motions, live Gradio interface). See phases in
repo history / `logs/`.

## Honest limitations

- All tasks are 2D, single-episode, fully observed toy physics — not
  image-based or partially observable RL.
- "84% on aiming" means the shared model still misses roughly 1 in 6 moving targets.
- The command interface (upcoming) maps fixed strings to trained motions;
  it is not open-ended language understanding.

## Stack

Python · Gymnasium · Stable-Baselines3 (PPO) · PyTorch · NumPy · GitHub Actions-free,
phone-built workflow (Termux + Colab).
