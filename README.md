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

### Disturbed squat recovery

Disturbance training is opt-in; the reference-free environment still has its
71-value observation and direct torque actions. The recovery configuration adds
the 28 target joint angles, base linear/angular velocity, and motion phase to
make a 106-value observation. It applies one seeded horizontal base push between
frames 25 and 80 at 960–1200 N for 4–8 control frames, adds Gaussian action
noise with standard deviation 0.02, and scales PD force to 80%. The imitation
reward and its C++/NumPy implementations are unchanged.

```bash
python calibrate_disturbances.py --episodes 50
python training/train_disturbed_squat.py --steps 500000 --seed 37 --force-max 1000
python training/train_disturbed_squat.py --input-model models/squat_disturbed_ppo.zip --steps 300000 --seed 73 --force-max 1200
python training/train_disturbed_squat.py --input-model models/squat_disturbed_ppo.zip --steps 20000 --seed 73
python training/train_disturbed_squat.py --input-model models/squat_disturbed_ppo.zip --steps 20000 --seed 73
python eval_disturbed_squat.py --episodes 50
```

Calibration on seeds 10000–10049 produced zero/random completion rates of
100/28%, 94/28%, 88/22%, 80/24%, and 70/20% at maximum push forces of 0, 400,
700, 1000, and 1200 N. The selected 1200 N level is the first calibrated level
at the 70% zero-action ceiling. The final PPO checkpoint contains 843,776
timesteps.

The same 50 disturbance seeds produced 86% PPO, 70% zero-action, and 20% random
completion: a passing 16 percentage-point PPO advantage. Without disturbance,
the rates were 100%, 100%, and 94%. `assets/squat_disturbed_eval.png` shows a
completed disturbed PPO episode before, during, and after its push.

### Motion-conditioned commands

One PPO policy handles squat, shoot, and dunk-reach references. Each episode
samples a motion and appends its three-value one-hot ID to the recovery
observation, producing 109 values. The exact commands `squat down`, `shoot the
target`, and `dunk it` select those IDs through `commands.py`; this is fixed
command routing, not open-ended language understanding. Dunk is an overhead
reach with toe rise, not an airborne jump.

```bash
python motions/generate_squat.py
python motions/generate_shoot.py
python motions/generate_dunk.py
python training/train_motion_conditioned.py --steps 600000 --seed 97
python training/train_motion_conditioned.py --input-model models/motion_conditioned_ppo.zip --steps 398000 --seed 97
python eval_motion_conditioned.py --episodes 50 --fidelity-episodes 20
python test_commands.py
```

The selected checkpoint used 999,424 actual timesteps in one configuration.
On held-out seeds 10000–10049, undisturbed PPO completion was 100% for every
motion. Disturbed PPO/zero completion was 82/70% for squat, 86/90% for shoot,
and 82/16% for dunk-reach. Squat and dunk-reach pass the recovery gate; shoot
does not, with a -4 percentage-point PPO gap. Motion fidelity selected the
commanded reference in 20/20 disturbed episodes for each motion. The three
`assets/*_conditioned_eval.png` strips show representative disturbed rollouts.

### Live command interface

`app.py` provides a Gradio text box, Run command button, and Surprise me
button for squat, shoot, dunk-reach, and intentional miss. It loads the
motion-conditioned policy for the first three commands and the dedicated Miss
checkpoint for `miss the target`. The callback is a generator: it advances
PyBullet once and yields the newly rendered frame, so the browser sees the
simulation as it runs rather than receiving a pre-built video or frame list.

```bash
python app.py
python test_app.py
```

### Basketball shoot mechanics

`BasketballShootEnv` adds a 0.62 kg dynamic ball, a fixed physical rim, and
downward rim-plane make detection to the shoot motion only. The ball follows
the right wrist until the documented release at frame 65, then receives the
measured wrist velocity and continues under PyBullet physics. A make requires
the ball center to cross downward through the rim plane within the 0.33 m
clearance left by the 0.45 m rim and 0.12 m ball radii.

The observation now appends the hoop vector from the shooting hand and its
distance to the original 109 motion features. Stage 1c can linearly reduce
imitation weight only during the 25 frames before release while retaining full
imitation elsewhere, and can randomize the training start position and yaw.
The Stage 1b policy is expanded from 109 to 113 inputs with zero-initialized
new columns, preserving its behavior before fine-tuning.

The separate shot term combines a one-time make reward, normalized
closest-approach progress after release, and a one-time release-quality term
based on the predicted ballistic miss distance and velocity alignment. The
default weights are 30 for the combined shot term, 1.0 for progress, and 0.5
for release quality. Evaluation always uses the physical 0.45 m rim.

```bash
python training/train_basketball_shoot.py --steps 1248000 --seed 251
python eval_basketball_shoot.py --model models/basketball_shoot_stage1c.zip \
  --episodes 50 --seed-start 10000 --visual assets/basketball_shoot_stage1c_made.png
python test_basketball_shoot.py
```

Stage 1c evaluated midpoint and final checkpoints from two configurations on
held-out seeds 20000–20049. Both annealed release-window imitation from 1.0 to
0.1; the selected configuration also randomized starts by ±0.25 m and ±5
degrees. Its final checkpoint made 4/50 held-out shots after 1,249,280
additional timesteps. The two configurations used 2,498,560 timesteps total.
On fixed gate seeds 10000–10049, the selected checkpoint made 3/50 shots (6%)
and completed 36/50 motions, versus 7/50 (14%) and 44/50 for Stage 1b and 1/50
(2%) and 46/50 for zero-offset replay. Its mean release speed was 4.321 m/s and
mean elevation angle was 4.13 degrees. `assets/basketball_shoot_stage1c_made.png`
is a real made-shot rollout. Stage 1c is an accepted limitation: it regressed
from Stage 1b and remains below the 30–40% demo-ready target, so no further
retraining is proposed within this stage.

### Intentional miss

`BasketballMissEnv` reuses the shoot mechanics and motion reference, but mirrors
the sparse outcome objective: a made basket receives the negative of the shoot
make reward, a released non-make receives +1 at episode end, and failing to
release receives -1. With outcome weight $w$, the command-specific adjustment
is `-2 * w * shoot_make + w * terminal_miss_outcome`. Imitation, release
quality, and physical ball/rim dynamics are unchanged.

```bash
python training/train_basketball_shoot.py --objective miss --steps 248000 \
  --input-model models/basketball_shoot_stage1b.zip --output models/basketball_miss_ppo
python eval_basketball_shoot.py --objective miss \
  --model models/basketball_miss_ppo.zip --episodes 50 --seed-start 10000 \
  --visual assets/basketball_miss_rollout.png
python test_basketball_miss.py
```

Two configurations used 249,856 actual timesteps each, 499,712 total, starting
from the stronger Stage 1b weights on the Stage 1c code lineage. On held-out
seeds 20000–20049, the selected seed-307, weight-30 final checkpoint produced
47/50 intentional misses, released 50/50 balls, and completed 46/50 motions.
On fixed gate seeds 10000–10049 it produced 48/50 intentional misses (96%),
released 50/50 balls, and completed 41/50 motions. The Stage 1b shoot policy
missed 43/50 (86%) on the same seeds, so the five-point reduction in makes is
attributable to the mirrored objective rather than non-release. The 100%
release rate and `assets/basketball_miss_rollout.png` show a normal shot motion
that deliberately sends the ball clear of the rim.

### Optional C++ reward

The environment automatically uses the pybind11 reward extension when it is
available and otherwise keeps the NumPy implementation. Build and verify it:

```bash
python -m pip install pybind11
./build_reward_cpp.sh
python test_reward.py
python benchmark_reward.py --calls 100000 --steps 10000
```

The build script runs the equivalent of:

```bash
c++ -O3 -Wall -shared -std=c++17 -fPIC $(python -m pybind11 --includes) \
  reward.cpp -o reward_cpp$(python-config --extension-suffix)
```

## Roadmap
Squat, shoot, dunk-reach, and intentional miss are available through fixed
command routing and a live-streaming Gradio interface. Basketball shoot has
physical ball/rim mechanics but remains low-reliability; Stage 1b remains the
stronger fixed-seed shooting checkpoint at a 14% make rate.

## Honest limitations

- All tasks are 2D, single-episode, fully observed toy physics — not
  image-based or partially observable RL.
- "84% on aiming" means the shared model still misses roughly 1 in 6 moving targets.
- The command interface maps four fixed strings to trained behaviors; it is not
  open-ended language understanding.
- Motions are hand-authored references. Dunk-reach has no airborne phase, and
  the shoot recovery gate did not beat its zero-action baseline.
- Basketball shooting improved from 8% to 14% in Stage 1b. Stage 1c added
  hoop-relative observation, release-window imitation annealing, and randomized
  starts, but regressed to 6% after its 2,498,560-timestep budget. Both remain
  below the 30–40% demo-ready threshold.
- Evaluation and the Gradio stream use PyBullet DIRECT-mode RGB rendering.

## Stack

Python · Gymnasium · Stable-Baselines3 (PPO) · PyTorch · PyBullet · NumPy ·
GitHub Actions-free, phone-built workflow (Termux + Colab).
