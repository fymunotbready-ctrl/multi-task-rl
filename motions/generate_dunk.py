from pathlib import Path

import numpy as np


N_FRAMES = 120
N_JOINT_ANGLES = 28


def build_dunk_reach():
    keyframes = np.array([0, 30, 55, 72, 105, 119])
    crouch = np.interp(
        np.arange(N_FRAMES), keyframes, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    )
    reach = np.interp(
        np.arange(N_FRAMES), keyframes, [0.0, 0.15, 1.0, 1.0, 0.0, 0.0]
    )
    frames = np.zeros((N_FRAMES, N_JOINT_ANGLES), dtype=np.float32)

    frames[:, 8] = -1.20 * reach
    frames[:, 9] = 0.15 * reach
    frames[:, 12] = -1.20 * reach
    frames[:, 13] = -0.15 * reach
    for hip_z, knee, ankle_z in ((16, 17, 20), (23, 24, 27)):
        frames[:, hip_z] = 0.45 * crouch - 0.06 * reach
        frames[:, knee] = -0.90 * crouch
        frames[:, ankle_z] = 0.45 * crouch - 0.10 * reach
    return frames


if __name__ == "__main__":
    output = Path(__file__).with_name("dunk.npy")
    motion = build_dunk_reach()
    np.save(output, motion)
    print(f"saved {output}: shape={motion.shape}, dtype={motion.dtype}")
