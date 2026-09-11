from envs.basketball.basketball_env import BasketballEnv

env = BasketballEnv()
obs, _ = env.reset()
print(f"Hoop is at x={obs[4]:.2f}, y={obs[5]:.2f}")

for shot in range(5):
    angle = float(input("Angle (15-85): "))
    power = float(input("Power (3-12): "))
    a0 = (angle - 50) / 35.0   # map back to [-1, 1]
    a1 = (power - 7.5) / 4.5
    env.reset()
    env.hoop = obs[4], obs[5]  # keep same hoop while practicing
    total, done = 0.0, False
    closest = 999.0
    while not done:
        obs, r, term, trunc, _ = env.step([a0, a1])
        a0, a1 = 0.0, 0.0  # action only matters on first step
        total += r
        done = term or trunc
        d = abs(obs[1] - obs[5]) if abs(obs[0] - obs[4]) < 0.5 else 999.0
        closest = min(closest, d)
    print(f"Reward: {total:.2f} | closest vertical miss: {closest:.2f}m")
    obs, _ = env.reset()
    print(f"\nNew hoop at x={obs[4]:.2f}, y={obs[5]:.2f}")
