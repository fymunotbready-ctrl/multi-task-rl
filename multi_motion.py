from disturbed_squat import DISTURBED_ENV_KWARGS


MOTION_PATHS = (
    "motions/squat.npy",
    "motions/shoot.npy",
    "motions/dunk.npy",
)
MOTION_NAMES = ("squat", "shoot", "dunk-reach")
MOTION_LABELS = {
    "squat": "squat",
    "shoot": "shoot",
    "dunk-reach": "dunk-reach (no airborne phase)",
}

MOTION_ENV_KWARGS = {
    **DISTURBED_ENV_KWARGS,
    "reference_motions": MOTION_PATHS,
}
UNDISTURBED_MOTION_ENV_KWARGS = {
    **MOTION_ENV_KWARGS,
    "push_count": 0,
    "action_noise_std": 0.0,
    "control_strength": 1.0,
}
