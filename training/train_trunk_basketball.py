import sys
sys.path.insert(0, ".")
from stable_baselines3 import PPO
from envs.basketball.basketball_env import BasketballEnv
from shared.shared_trunk import trunk_policy_kwargs, save_trunk

env = BasketballEnv()
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4,
            n_steps=1024, batch_size=64, seed=42,
            policy_kwargs=trunk_policy_kwargs("basketball"))
print("=== basketball builds the trunk (150k) ===")
model.learn(total_timesteps=150_000)
model.save("models/trunk_basketball")
save_trunk(model, "models/trunk_basketball.pt")

env = BasketballEnv()
hits = 0
for i in range(100):
    obs, _ = env.reset(seed=1000 + i)
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(a)
        done = term or trunc
    if term and r >= 0.99:
        hits += 1
print(f"\nTRUNK GATE: basketball {hits}/100 (need 100)")
