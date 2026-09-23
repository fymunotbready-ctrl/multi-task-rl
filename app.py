import random
import time
from functools import lru_cache

import gradio as gr
from stable_baselines3 import PPO

from commands import COMMAND_MAP, resolve_command
from basketball_shoot import BASKETBALL_ENV_KWARGS
from envs.ragdoll import BasketballDunkEnv, BasketballMissEnv, RagdollEnv
from envs.vehicle import VehicleEnv
from multi_motion import MOTION_ENV_KWARGS

MODEL_PATH = "models/motion_conditioned_ppo.zip"
MISS_MODEL_PATH = "models/basketball_miss_ppo.zip"
DRIFT_MODEL_PATH = "models/drift_ppo.zip"
DUNK_MODEL_PATH = "models/basketball_dunk_ppo.zip"
FRAME_INTERVAL_SECONDS = 1.0 / 30.0


@lru_cache(maxsize=5)
def load_model(path=MODEL_PATH):
    return PPO.load(path)


def stream_rollout(model, env, command_text, frame_interval=FRAME_INTERVAL_SECONDS):
    motion_idx = resolve_command(command_text)
    if motion_idx is None:
        observation, _ = env.reset()
    else:
        observation, _ = env.reset(motion_idx=motion_idx)
    yield env.render()

    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(action)
        frame = env.render()
        if frame_interval:
            time.sleep(frame_interval)
        yield frame
        done = terminated or truncated


def stream_command(command_text):
    normalized = command_text.strip().lower()
    if normalized == "miss the target":
        env = BasketballMissEnv(
            render_mode="rgb_array",
            **BASKETBALL_ENV_KWARGS,
        )
        model_path = MISS_MODEL_PATH
    elif normalized == "dunk it":
        env = BasketballDunkEnv(
            render_mode="rgb_array",
            **BASKETBALL_ENV_KWARGS,
        )
        model_path = DUNK_MODEL_PATH
    elif normalized == "drift":
        env = VehicleEnv(render_mode="rgb_array")
        model_path = DRIFT_MODEL_PATH
    else:
        env = RagdollEnv(render_mode="rgb_array", **MOTION_ENV_KWARGS)
        model_path = MODEL_PATH
    try:
        yield from stream_rollout(load_model(model_path), env, normalized)
    finally:
        env.close()


def surprise_command(rng=None):
    chooser = rng if rng is not None else random
    return chooser.choice(tuple(COMMAND_MAP))


def build_demo():
    with gr.Blocks(title="Ragdoll Commands") as demo:
        gr.Markdown("# Ragdoll Commands\nEnter one of the five trained commands.")
        command = gr.Textbox(
            label="Command",
            placeholder="squat down, shoot the target, miss the target, dunk it, or drift",
        )
        with gr.Row():
            run = gr.Button("Run command", variant="primary")
            surprise = gr.Button("Surprise me")
        frame = gr.Image(label="Live simulation", streaming=True)

        run.click(stream_command, inputs=command, outputs=frame)
        command.submit(stream_command, inputs=command, outputs=frame)
        surprise.click(surprise_command, outputs=command).then(
            stream_command, inputs=command, outputs=frame
        )
    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
