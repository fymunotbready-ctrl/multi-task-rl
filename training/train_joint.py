import sys, os, csv
sys.path.insert(0, ".")
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from envs.meta_env import MultiTaskEnv, TASK_NAMES
from envs.basketball.basketball_env import BasketballEnv
from envs.driving.driving_env import DrivingEnv
from envs.aiming.aiming_env import AimingEnv
from shared.shared_trunk import trunk_policy_kwargs, load_trunk, encode_obs

EVAL_ENVS = [BasketballEnv(), DrivingEnv(), AimingEnv()]
TOTAL_STEPS = 1_000_000
EVAL_EVERY = 50_000
N_EVAL = 40


def task_success(model, env, idx, n):
    hits = 0
    for i in range(n):
        obs, _ = env.reset(seed=100_000 + i)
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
        with open("logs/joint_train.csv", "w", newline="") as f:
            csv.writer(f).writerow(["steps", "task", "ep_rew_mean", "success"])
        with open("logs/joint_eval.csv", "w", newline="") as f:
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
            with open("logs/joint_eval.csv", "a", newline="") as f:
                csv.writer(f).writerow([self.num_timesteps] + rates + [mean])
            print("  EVAL:", {TASK_NAMES[i]: f"{rates[i]*100:.0f}%" for i in range(3)},
                  f"mean {mean*100:.0f}%")
            if mean > self.best:
                self.best = mean
                self.model.save("models/joint_trunk_best")
                print("  >>> new best, banked models/joint_trunk_best")
        return True

    def _on_rollout_end(self):
        if not self.window:
            return
        with open("logs/joint_train.csv", "a", newline="") as f:
            w = csv.writer(f)
            for t in range(len(TASK_NAMES)):
                rows = [x for x in self.window if x[0] == t]
                if rows:
                    w.writerow([self.num_timesteps, TASK_NAMES[t],
                                sum(x[1] for x in rows) / len(rows),
                                sum(x[2] for x in rows) / len(rows)])
        self.window = []


env = MultiTaskEnv()
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4,
            n_steps=2048, batch_size=256, seed=5,
            policy_kwargs=trunk_policy_kwargs("basketball"))
if os.path.exists("models/trunk_basketball.pt"):
    load_trunk(model, "models/trunk_basketball.pt")
    print(">>> warm-started from basketball trunk")
else:
    print(">>> no trunk found, from scratch")
print(f"=== JOINT agent: 3 tasks, {TOTAL_STEPS} steps ===")
model.learn(total_timesteps=TOTAL_STEPS, callback=JointCallback())
model.save("models/joint_trunk_final")
for i, name in enumerate(TASK_NAMES):
    print(f"JOINT  {name:12s} success: {task_success(model, EVAL_ENVS[i], i, 100)*100:5.1f}%")
