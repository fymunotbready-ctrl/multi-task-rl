# Project summary

## What this project is

This is a small reinforcement-learning portfolio project built with Stable-Baselines3 and PyBullet. It combines simple 2D control benchmarks with a simulated humanoid, a basketball and hoop, and a small vehicle. A Gradio page lets a viewer run five fixed demonstrations.

The interface recognizes only these exact commands:

- `squat down`
- `shoot the target`
- `miss the target`
- `dunk it`
- `drift`

This is command routing, not a general chatbot or natural-language system.

## What works well

### Squat and recovery

The humanoid can follow a hand-authored squat motion. In the fixed 50-episode disturbance evaluation, the trained recovery policy completed 86% of episodes versus 70% for the zero-action baseline. Without disturbances, it completed all 50 episodes.

### Intentional miss

The intentional-miss policy uses a real simulated ball and hoop. It released the ball in all 50 fixed-seed episodes and deliberately missed 48 of them, a 96% intentional-miss rate.

### Drift

The vehicle policy produces controlled sideways motion without flipping. Across 50 fixed-seed episodes it stayed safe in all 50 and averaged 28.11 degrees of slip, compared with 3.97 degrees for random actions.

### Airborne dunk motion

The dunk uses the simulated humanoid, ball, and hoop. It achieved a real airborne phase in all 50 fixed-seed episodes and landed after the jump in 48 of them. This is a genuine jump rather than the earlier toe-rise reach.

### Basic 2D benchmarks

The shared 2D policy reached 100% on the basketball and driving benchmark evaluations. These are small, fully observed toy environments rather than visual or real-world tasks.

## Known limitations

### Basketball shooting is unreliable

The strongest separately evaluated physical-shoot checkpoint made 7 of 50 shots, or 14%. A later experiment with hoop-relative observations and randomized starts regressed to 6%. The project keeps the stronger 14% checkpoint and treats this as a known limitation.

The live `shoot the target` command demonstrates the learned shooting body motion from the motion-conditioned policy. It is not evidence of reliable physical basket scoring.

### The dunk rarely scores

The selected airborne dunk policy made 1 of 50 fixed-seed attempts, or 2%. It reliably jumps, but it does not reliably put the ball through the hoop. Zero-action replay scored more often in that evaluation, so the trained policy did not improve scoring reliability.

### Shared aiming stopped at 84%

The shared multi-task 2D model reached 84% on aiming. That is below the project's 90% target and means it still misses roughly one in six targets.

### Not all recovery results beat a baseline

The motion-conditioned shoot recovery result was 86% completion under disturbance, while zero-offset replay reached 90% on the same fixed seeds. The learned policy therefore did not beat that baseline for this motion.

### The scope is intentionally small

The motions are hand-authored references. The simulations are fully observed, run one episode at a time, and use PyBullet's direct RGB renderer. They do not demonstrate camera-based perception, open-ended language understanding, real robot control, or production-grade driving.

## Bottom line

The strongest demonstrations are disturbed squat recovery, intentional miss, controlled drift, and a visibly airborne dunk motion. The basketball scoring policies and shared aiming model are useful documented experiments, but their measured success rates are not demo-ready and are not presented as such.
