DISTURBED_ENV_KWARGS = {
    "push_count": 1,
    "push_frame_range": (25, 80),
    "push_force_range": (960.0, 1200.0),
    "push_duration_range": (4, 8),
    "action_noise_std": 0.02,
    "control_strength": 0.8,
    "include_reference_observation": True,
}

TRAINING_ENV_KWARGS = {
    **DISTURBED_ENV_KWARGS,
    "push_force_range": (0.0, 1200.0),
}

UNDISTURBED_ENV_KWARGS = {
    "include_reference_observation": True,
}
