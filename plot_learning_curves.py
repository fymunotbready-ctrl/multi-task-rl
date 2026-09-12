import sys, csv
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    steps, rews, succ = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["steps"]))
            rews.append(float(row["ep_rew_mean"]))
            succ.append(float(row["success_rate"]))
    return steps, rews, succ


def smooth(xs, w=10):
    return [sum(xs[max(0, i-w+1):i+1]) / len(xs[max(0, i-w+1):i+1])
            for i in range(len(xs))]


arms = {}
for label, path in [("aiming from basketball trunk", "logs/aiming_from_basketball.csv"),
                    ("aiming from random trunk", "logs/aiming_from_random.csv")]:
    arms[label] = load(path)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#1f77b4", "#d62728"]
for (label, (steps, rews, succ)), c in zip(arms.items(), colors):
    ax.plot(steps, rews, color=c, alpha=0.15)
    ax.plot(steps, smooth(rews), color=c, label=label + " (smoothed)")
ax.set_xlabel("timesteps"); ax.set_ylabel("ep_rew_mean")
ax.set_title("Aiming: pretrained shared trunk vs fresh trunk")
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout()
plt.savefig("assets/learning_curves.png", dpi=150)
print("saved assets/learning_curves.png")
for label, (steps, rews, succ) in arms.items():
    s = smooth(succ)
    cross = next((steps[i] for i, v in enumerate(s) if v >= 0.8), None)
    print(f"{label:32s} final success {succ[-1]*100:5.1f}%  "
          f"first smoothed >=80% success at: {cross}")
