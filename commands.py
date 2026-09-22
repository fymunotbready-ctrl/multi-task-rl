COMMAND_MAP = {
    "squat down": 0,
    "shoot the target": 1,
    "miss the target": 1,
    "dunk it": 2,
}


def resolve_command(command_text):
    normalized = command_text.strip().lower()
    if normalized not in COMMAND_MAP:
        valid = ", ".join(f'"{command}"' for command in COMMAND_MAP)
        raise ValueError(f"unknown command {command_text!r}; valid commands: {valid}")
    return COMMAND_MAP[normalized]


def run_command(model, env, command_text):
    motion_idx = resolve_command(command_text)
    observation, _ = env.reset(motion_idx=motion_idx)
    first_frame = env.render()
    if first_frame is None:
        raise ValueError("run_command requires an environment with rgb_array rendering")
    frames = [first_frame]
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        done = terminated or truncated
    return frames
