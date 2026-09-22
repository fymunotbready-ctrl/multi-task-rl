import random
import time
from functools import lru_cache

import gradio as gr
from stable_baselines3 import PPO

from commands import COMMAND_MAP, resolve_command
from envs.ragdoll import RagdollEnv
from multi_motion import MOTION_ENV_KWARGS

MODEL_PATH = "models/motion_conditioned_ppo.zip"
FRAME_INTERVAL_SECONDS = 1.0 / 30.0


@lru_cache(maxsize=1)
def load_model():
    return PPO.load(MODEL_PATH)


def stream_rollout(model, env, command_text, frame_interval=FRAME_INTERVAL_SECONDS):
    motion_idx = resolve_command(command_text)
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
    env = RagdollEnv(render_mode="rgb_array", **MOTION_ENV_KWARGS)
    try:
        yield from stream_rollout(load_model(), env, command_text)
    finally:
        env.close()


def surprise_command(rng=None):
    chooser = rng if rng is not None else random
    return chooser.choice(tuple(COMMAND_MAP))


def build_demo():
    with gr.Blocks(title="Ragdoll Commands") as demo:
        gr.Markdown("# Ragdoll Commands\nEnter one of the three trained commands.")
        command = gr.Textbox(
            label="Command",
            placeholder="squat down, shoot the target, or dunk it",
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
