from envs.basketball.basketball_env import BasketballEnv, DT, G

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

# Episode metadata must agree with the terminal scoring reward used by trainers.
env.reset(seed=1)
env.launched = True
env.hoop[:] = [6.0, 3.0]
env.ball[:] = [5.9, 3.0]
env.vel[:] = [3.0, G * DT]
_, reward, terminated, truncated, info = env.step([0.0, 0.0])
assert terminated and not truncated
assert reward == 1.0
assert info["success"]
