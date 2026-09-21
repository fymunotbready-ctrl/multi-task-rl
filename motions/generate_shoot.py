from pathlib import Path

import numpy as np


N_FRAMES = 120
N_JOINT_ANGLES = 28


def build_shoot():
    keyframes = np.array([0, 24, 54, 82, 119])
    lift = np.interp(np.arange(N_FRAMES), keyframes, [0.0, 0.25, 1.0, 1.0, 0.0])
    extend = np.interp(np.arange(N_FRAMES), keyframes, [0.0, 0.0, 0.45, 1.0, 0.0])
    frames = np.zeros((N_FRAMES, N_JOINT_ANGLES), dtype=np.float32)

    frames[:, 0] = -0.08 * lift
    frames[:, 8] = -1.15 * lift
    frames[:, 9] = 1.25 * (lift - 0.75 * extend)
    frames[:, 12] = -1.15 * lift
    frames[:, 13] = -1.25 * (lift - 0.75 * extend)
    frames[:, 16] = 0.10 * lift
    frames[:, 17] = -0.20 * lift
    frames[:, 20] = 0.10 * lift
    frames[:, 23] = 0.06 * lift
    frames[:, 24] = -0.12 * lift
    frames[:, 27] = 0.06 * lift
    return frames


if __name__ == "__main__":
    output = Path(__file__).with_name("shoot.npy")
    motion = build_shoot()
    np.save(output, motion)
    print(f"saved {output}: shape={motion.shape}, dtype={motion.dtype}")
