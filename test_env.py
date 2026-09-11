from envs.basketball.basketball_env import BasketballEnv

env = BasketballEnv()
obs, _ = env.reset()
print("obs:", obs)
total = 0.0
for i in range(10):
    obs, r, term, trunc, _ = env.step(env.action_space.sample())
    total += r
    if term or trunc:
        print(f"episode ended, total reward: {total:.2f}")
        obs, _ = env.reset()
        total = 0.0
print("OK - env runs")
