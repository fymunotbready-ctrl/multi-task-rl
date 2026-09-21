from pathlib import Path

import numpy as np


N_FRAMES = 120
N_JOINT_ANGLES = 28


def build_squat():
    phase = np.linspace(0.0, np.pi, N_FRAMES, dtype=np.float32)
    depth = np.sin(phase) ** 2
    frames = np.zeros((N_FRAMES, N_JOINT_ANGLES), dtype=np.float32)

    for hip_z, knee, ankle_z in ((16, 17, 20), (23, 24, 27)):
        frames[:, hip_z] = 0.45 * depth
        frames[:, knee] = -0.9 * depth
        frames[:, ankle_z] = 0.45 * depth

    return frames


if __name__ == "__main__":
    output = Path(__file__).with_name("squat.npy")
    motion = build_squat()
    np.save(output, motion)
    print(f"saved {output}: shape={motion.shape}, dtype={motion.dtype}")
