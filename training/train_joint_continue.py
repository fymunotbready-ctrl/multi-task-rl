import sys, os, csv
sys.path.insert(0, ".")
import torch
torch.set_num_threads(1)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from envs.meta_env import MultiTaskEnv, TASK_NAMES
from envs.basketball.basketball_env import BasketballEnv
from envs.driving.driving_env import DrivingEnv
from envs.aiming.aiming_env import AimingEnv
from envs.meta_env import encode_obs

EVAL_ENVS = [BasketballEnv(), DrivingEnv(), AimingEnv()]
EXTRA_STEPS = 500_000
EVAL_EVERY = 50_000
N_EVAL = 40


def task_success(model, env, idx, n):
    hits = 0
    for i in range(n):
        obs, _ = env.reset(seed=200_000 + i)
        obs = encode_obs(obs, idx)
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            obs = encode_obs(obs, idx)
            done = term or trunc
        if term and r >= 0.99:
            hits += 1
    return hits / n


class JointCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.window, self.best, self._last_eval = [], -1.0, 0
        with open("logs/joint_train_v2.csv", "w", newline="") as f:
            csv.writer(f).writerow(["steps", "task", "ep_rew_mean", "success"])
        with open("logs/joint_eval_v2.csv", "w", newline="") as f:
            csv.writer(f).writerow(["steps"] + TASK_NAMES + ["mean"])

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.window.append(
                    (info["task"], info["episode"]["r"], bool(info.get("success", False))))
        if self.num_timesteps - self._last_eval >= EVAL_EVERY:
            self._last_eval = self.num_timesteps
            rates = [task_success(self.model, EVAL_ENVS[i], i, N_EVAL)
                     for i in range(len(TASK_NAMES))]
            mean = sum(rates) / len(rates)
            with open("logs/joint_eval_v2.csv", "a", newline="") as f:
                csv.writer(f).writerow([self.num_timesteps] + rates + [mean])
            print("  EVAL v2:", {TASK_NAMES[i]: f"{rates[i]*100:.0f}%" for i in range(3)},
                  f"mean {mean*100:.0f}%")
            if mean > self.best:
                self.best = mean
                self.model.save("models/joint_trunk_best_v2")
                print("  >>> new best v2, banked")
        return True

    def _on_rollout_end(self):
        if not self.window:
            return
        with open("logs/joint_train_v2.csv", "a", newline="") as f:
            w = csv.writer(f)
            for t in range(len(TASK_NAMES)):
                rows = [x for x in self.window if x[0] == t]
                if rows:
                    w.writerow([self.num_timesteps, TASK_NAMES[t],
                                sum(x[1] for x in rows) / len(rows),
                                sum(x[2] for x in rows) / len(rows)])
        self.window = []


env = MultiTaskEnv()
model = PPO.load(
    "models/joint_trunk_best.zip",
    env=env,
    custom_objects={"policy_class": ActorCriticPolicy, "clip_range": 0.2},
)
model.set_env(env)
model.verbose = 1
model.learning_rate = 3e-4
model.n_steps = 2048
model.batch_size = 256

print(f"=== JOINT v2: +{EXTRA_STEPS} steps from best checkpoint ===")
model.learn(total_timesteps=EXTRA_STEPS, callback=JointCallback(), reset_num_timesteps=False)
model.save("models/joint_trunk_final_v2")
for i, name in enumerate(TASK_NAMES):
    print(f"JOINT v2 {name:12s} success: {task_success(model, EVAL_ENVS[i], i, 100)*100:5.1f}%")
