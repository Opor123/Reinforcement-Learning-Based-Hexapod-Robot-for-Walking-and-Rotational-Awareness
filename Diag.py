"""
Detailed diagnosis tool to understand what the robot is doing
"""
import numpy as np
from pathlib import Path
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from spider_env import SpiderWalkEnv


def diagnose_model(model_path, vecnorm_path, steps=500):
    """Run detailed diagnostics on what the model is actually doing"""

    print("=" * 80)
    print("DETAILED MODEL DIAGNOSIS")
    print("=" * 80)

    # Load model
    def make_env():
        env = SpiderWalkEnv(render_mode="human")
        return Monitor(env)

    base_env = DummyVecEnv([make_env])
    vec_env = VecNormalize.load(vecnorm_path, base_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(model_path, env=vec_env, device="cpu")

    # Reset
    obs = vec_env.reset()
    robot = vec_env.envs[0].unwrapped.robot

    # Track metrics
    positions = []
    heights = []
    velocities = []
    action_history = []
    reward_breakdown = []

    print("\nRunning diagnosis for {} steps...\n".format(steps))

    for step in range(steps):
        # Get action
        action, _ = model.predict(obs, deterministic=True)
        action_history.append(action[0].copy())

        # Step
        obs, reward, done, info = vec_env.step(action)

        # Get state
        pos, ori = p.getBasePositionAndOrientation(robot)
        vel, ang_vel = p.getBaseVelocity(robot)

        positions.append(pos[0])
        heights.append(pos[2])
        velocities.append(vel[0])
        reward_breakdown.append(info[0])

        if step % 100 == 0:
            print(f"Step {step:3d}: x={pos[0]:6.3f}m, z={pos[2]:6.3f}m, vx={vel[0]:6.3f}m/s, r={reward[0]:6.2f}")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    actions = np.array(action_history)

    print(f"\n📊 Position:")
    print(f"  Start X: {positions[0]:.3f}m")
    print(f"  End X:   {positions[-1]:.3f}m")
    print(f"  Total distance: {positions[-1] - positions[0]:.3f}m")
    print(f"  Max X reached: {max(positions):.3f}m")

    print(f"\n📊 Height:")
    print(f"  Average: {np.mean(heights):.3f}m")
    print(f"  Std Dev: {np.std(heights):.3f}m")
    print(f"  Min: {min(heights):.3f}m")
    print(f"  Max: {max(heights):.3f}m")

    print(f"\n📊 Velocity:")
    print(f"  Average vx: {np.mean(velocities):.3f}m/s")
    print(f"  Max vx: {max(velocities):.3f}m/s")
    print(f"  Min vx: {min(velocities):.3f}m/s")

    print(f"\n📊 Actions:")
    print(f"  Mean: {np.mean(actions):.3f}")
    print(f"  Std:  {np.std(actions):.3f}")
    print(f"  Min:  {np.min(actions):.3f}")
    print(f"  Max:  {np.max(actions):.3f}")
    print(f"  % at +1.0: {(np.abs(actions - 1.0) < 0.01).mean() * 100:.1f}%")
    print(f"  % at -1.0: {(np.abs(actions + 1.0) < 0.01).mean() * 100:.1f}%")
    print(f"  % at  0.0: {(np.abs(actions) < 0.01).mean() * 100:.1f}%")

    # Check if actions are varying or static
    action_changes = np.abs(np.diff(actions, axis=0)).mean()
    print(f"  Action change per step: {action_changes:.4f}")
    if action_changes < 0.01:
        print("  ⚠️  WARNING: Actions are nearly static! Robot may be frozen.")

    # Reward breakdown
    print(f"\n📊 Average Reward Components:")
    keys = ['velocity_reward', 'distance_reward', 'stability_reward', 'forward_reward']
    for key in keys:
        if key in reward_breakdown[0]:
            values = [r[key] for r in reward_breakdown]
            print(f"  {key:20s}: {np.mean(values):7.3f}")

    # Check for reward hacking
    print(f"\n🔍 Potential Issues:")
    if positions[-1] - positions[0] < 0.1:
        print("  ❌ Robot is NOT moving forward")
        if np.mean([r.get('velocity_reward', 0) for r in reward_breakdown]) > 0:
            print("     But getting positive velocity rewards! (reward hacking)")
        if np.std(heights) < 0.01:
            print("     Height is very stable - robot might be stuck in place")
        if action_changes < 0.01:
            print("     Actions are static - robot found a 'do nothing' policy")

    if np.mean([r.get('contacts', 0) for r in reward_breakdown]) < 2:
        print("  ❌ Very few ground contacts - robot may be falling")

    print("\n" + "=" * 80)
    print("Press Q to close")
    print("=" * 80)

    # Keep GUI open
    while p.isConnected():
        keys = p.getKeyboardEvents()
        if ord('q') in keys or ord('Q') in keys:
            break
        import time
        time.sleep(0.01)

    vec_env.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
    else:
        # Auto-find latest
        training_runs = Path("training_runs")
        runs = sorted(training_runs.glob("spider_ppo_*"), key=lambda x: x.stat().st_mtime, reverse=True)
        if runs:
            run_dir = runs[0]
            print(f"Using latest run: {run_dir.name}\n")
        else:
            print("No training runs found!")
            sys.exit(1)

    model_path = run_dir / "final_model.zip"
    vecnorm_path = run_dir / "vecnorm_final.pkl"

    diagnose_model(str(model_path), str(vecnorm_path))