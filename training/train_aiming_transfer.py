import sys, csv, argparse
sys.path.insert(0, ".")
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from envs.aiming.aiming_env import AimingEnv
from shared.shared_trunk import trunk_policy_kwargs, load_trunk

p = argparse.ArgumentParser()
p.add_argument("--init", choices=["basketball", "random"], required=True)
p.add_argument("--steps", type=int, default=300_000)
args = p.parse_args()
LOG = f"logs/aiming_from_{args.init}.csv"


class MetricsLogger(BaseCallback):
    def __init__(self, path):
        super().__init__()
        self.path, self.window = path, []
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(["steps", "ep_rew_mean", "success_rate"])

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.window.append((info["episode"]["r"], bool(info.get("success", False))))
        return True

    def _on_rollout_end(self):
        if self.window:
            rs = [w[0] for w in self.window]
            ss = [w[1] for w in self.window]
            with open(self.path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [self.num_timesteps, sum(rs)/len(rs), sum(ss)/len(ss)])
            self.window = []


env = AimingEnv()
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4,
            n_steps=1024, batch_size=64, seed=11,
            policy_kwargs=trunk_policy_kwargs("aiming"))
if args.init == "basketball":
    load_trunk(model, "models/trunk_basketball.pt")
    print(">>> loaded shared trunk from basketball")
else:
    print(">>> fresh random trunk")
print(f"=== aiming from {args.init} trunk ({args.steps} steps) ===")
model.learn(total_timesteps=args.steps, callback=MetricsLogger(LOG))
model.save(f"models/aiming_from_{args.init}")
print(f"CSV -> {LOG}")
