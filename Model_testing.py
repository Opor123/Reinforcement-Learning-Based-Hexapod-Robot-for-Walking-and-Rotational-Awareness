"""
🧪 Spider Robot Model Evaluation Script
Comprehensive testing and diagnostics for trained models

Usage:
    python evaluate_model.py                    # Auto-find latest model
    python evaluate_model.py --run-dir path/to/run
    python evaluate_model.py --model model.zip --vecnorm vecnorm.pkl
    python evaluate_model.py --compare-random   # Compare with random baseline
"""

import sys
import time
from pathlib import Path
import argparse

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from spider_env import SpiderWalkEnv


# ============================================================================
# Pretty Output
# ============================================================================

class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}\n")


def print_section(text):
    print(f"\n{Colors.BOLD}{Colors.CYAN}▶ {text}{Colors.END}")


def print_metric(label, value, good_threshold=None, bad_threshold=None):
    """Print a metric with color coding"""
    color = Colors.END

    if good_threshold is not None and bad_threshold is not None:
        if isinstance(value, (int, float)):
            if value >= good_threshold:
                color = Colors.GREEN
            elif value <= bad_threshold:
                color = Colors.RED
            else:
                color = Colors.YELLOW

    print(f"  {label:.<40} {color}{value}{Colors.END}")


# ============================================================================
# Testing Functions
# ============================================================================

def test_random_policy(env, num_episodes=3, max_steps=1000, render=False):
    """
    Test with completely random actions (baseline comparison)
    """
    print_section("Testing Random Policy (Baseline)")

    returns = []
    distances = []
    heights = []
    contacts = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        start_x = info.get('base_height', 0)  # Will track actual position
        max_height = 0
        total_contacts = 0

        done = False
        while not done and steps < max_steps:
            # Random action
            action = np.random.uniform(-1, 1, env.action_space.shape)
            obs, reward, done, truncated, info = env.step(action)

            total_reward += reward
            max_height = max(max_height, info.get('base_height', 0))
            total_contacts += info.get('contacts', 0)
            steps += 1

            if truncated:
                break

        # Get final position
        final_pos, _ = env.unwrapped.robot if hasattr(env, 'unwrapped') else (None, None)
        try:
            import pybullet as p
            if env.unwrapped.robot is not None:
                final_pos, _ = p.getBasePositionAndOrientation(env.unwrapped.robot)
                distance = final_pos[0]  # X distance traveled
            else:
                distance = 0
        except:
            distance = 0

        returns.append(total_reward)
        distances.append(distance)
        heights.append(max_height)
        contacts.append(total_contacts / max(steps, 1))

        print(f"  Episode {ep+1}: Reward={total_reward:>6.2f} | Distance={distance:>6.3f}m | Steps={steps}")

    print(f"\n{Colors.BOLD}Random Policy Results:{Colors.END}")
    print_metric("Avg Return", f"{np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print_metric("Avg Distance", f"{np.mean(distances):.3f}m ± {np.std(distances):.3f}m")
    print_metric("Avg Max Height", f"{np.mean(heights):.3f}m")
    print_metric("Avg Contacts/Step", f"{np.mean(contacts):.2f}/6")

    return {
        'returns': returns,
        'distances': distances,
        'heights': heights,
        'contacts': contacts
    }


def test_trained_model(model_path, vecnorm_path, num_episodes=5, max_steps=1000, render=True, deterministic=True):
    """
    Test a trained PPO model with detailed diagnostics
    """
    print_section(f"Testing Trained Model {'(Deterministic)' if deterministic else '(Stochastic)'}")

    # Check files exist
    if not Path(model_path).exists():
        print(f"{Colors.RED}✗ Model not found: {model_path}{Colors.END}")
        return None
    if not Path(vecnorm_path).exists():
        print(f"{Colors.RED}✗ VecNormalize not found: {vecnorm_path}{Colors.END}")
        return None

    print(f"  Model: {model_path}")
    print(f"  VecNorm: {vecnorm_path}")

    # Create environment
    def _make_env():
        env = SpiderWalkEnv(render_mode="human" if render else None)
        return Monitor(env)

    base_env = DummyVecEnv([_make_env])
    vec_env = VecNormalize.load(vecnorm_path, base_env)
    vec_env.training = False
    vec_env.norm_reward = False

    # Load model
    try:
        model = PPO.load(model_path, env=vec_env, device="cpu", print_system_info=False)
        print(f"{Colors.GREEN}✓ Model loaded successfully{Colors.END}\n")
    except Exception as e:
        print(f"{Colors.RED}✗ Failed to load model: {e}{Colors.END}")
        return None

    # Test episodes
    returns = []
    distances = []
    heights = []
    contacts_per_step = []
    action_magnitudes = []

    print(f"{Colors.BOLD}Running {num_episodes} episodes...{Colors.END}\n")

    for ep in range(num_episodes):
        obs = vec_env.reset()
        done = np.array([False])
        total_reward = 0
        steps = 0
        max_height = 0
        total_contacts = 0
        action_mags = []

        # Get starting position
        import pybullet as p
        robot = vec_env.envs[0].unwrapped.robot
        start_pos, _ = p.getBasePositionAndOrientation(robot)
        start_x = start_pos[0]

        while not done.any() and steps < max_steps:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = vec_env.step(action)

            total_reward += float(reward[0])
            max_height = max(max_height, info[0].get('base_height', 0))
            total_contacts += info[0].get('contacts', 0)
            action_mags.append(np.abs(action).mean())
            steps += 1

        # Get final position
        final_pos, _ = p.getBasePositionAndOrientation(robot)
        distance = final_pos[0] - start_x

        returns.append(total_reward)
        distances.append(distance)
        heights.append(max_height)
        contacts_per_step.append(total_contacts / max(steps, 1))
        action_magnitudes.append(np.mean(action_mags))

        status = f"{Colors.GREEN}✓{Colors.END}" if distance > 0.5 else f"{Colors.RED}✗{Colors.END}"
        print(f"  {status} Episode {ep+1}: "
              f"Reward={total_reward:>7.2f} | "
              f"Distance={distance:>6.3f}m | "
              f"Steps={steps:>4} | "
              f"Contacts={total_contacts/max(steps,1):.1f}/6")

    # Summary statistics
    print(f"\n{Colors.BOLD}Trained Model Results:{Colors.END}")
    print_metric("Avg Return", f"{np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print_metric("Avg Distance", f"{np.mean(distances):.3f}m ± {np.std(distances):.3f}m",
                 good_threshold=1.0, bad_threshold=0.1)
    print_metric("Max Distance", f"{np.max(distances):.3f}m")
    print_metric("Min Distance", f"{np.min(distances):.3f}m")
    print_metric("Avg Max Height", f"{np.mean(heights):.3f}m")
    print_metric("Avg Contacts/Step", f"{np.mean(contacts_per_step):.2f}/6",
                 good_threshold=3.0, bad_threshold=1.0)
    print_metric("Avg Action Magnitude", f"{np.mean(action_magnitudes):.3f}",
                 good_threshold=0.3, bad_threshold=0.05)

    # Diagnosis
    print(f"\n{Colors.BOLD}Diagnosis:{Colors.END}")

    avg_dist = np.mean(distances)
    avg_action = np.mean(action_magnitudes)

    if avg_dist < 0.1:
        print(f"  {Colors.RED}⚠ Robot is NOT moving forward!{Colors.END}")
        if avg_action < 0.1:
            print(f"    • Actions are very small (avg={avg_action:.3f})")
            print(f"    • Model may have collapsed to near-zero policy")
            print(f"    • Try: Lower entropy coefficient, check reward shaping")
        else:
            print(f"    • Actions are active (avg={avg_action:.3f}) but not productive")
            print(f"    • Try: Check reward function, increase training time")
    elif avg_dist < 0.5:
        print(f"  {Colors.YELLOW}⚠ Robot is moving but slowly{Colors.END}")
        print(f"    • Average distance: {avg_dist:.3f}m")
        print(f"    • Needs more training or reward tuning")
    else:
        print(f"  {Colors.GREEN}✓ Robot is walking successfully!{Colors.END}")
        print(f"    • Average distance: {avg_dist:.3f}m")

    if np.mean(contacts_per_step) < 2.0:
        print(f"  {Colors.RED}⚠ Low ground contact (avg={np.mean(contacts_per_step):.1f}/6){Colors.END}")
        print(f"    • Robot may be falling or unstable")
        print(f"    • Check spawn height and stance configuration")

    if render:
        print(f"\n{Colors.YELLOW}GUI is open. Press Q to close.{Colors.END}")
        _keep_gui_open()

    vec_env.close()

    return {
        'returns': returns,
        'distances': distances,
        'heights': heights,
        'contacts': contacts_per_step,
        'actions': action_magnitudes
    }


def _keep_gui_open():
    """Keep PyBullet GUI open"""
    try:
        import pybullet as p
        if p.isConnected():
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            try:
                p.resetDebugVisualizerCamera(distance=1.5, yaw=45, pitch=-30, targetPosition=[0, 0, 0.1])
            except:
                pass

            while p.isConnected():
                keys = p.getKeyboardEvents()
                if ord('q') in keys or ord('Q') in keys:
                    break
                time.sleep(0.01)
    except:
        pass


def compare_policies(trained_results, random_results):
    """
    Compare trained model vs random baseline
    """
    if trained_results is None or random_results is None:
        return

    print_section("Comparison: Trained vs Random")

    improvement_reward = (np.mean(trained_results['returns']) - np.mean(random_results['returns'])) / max(abs(np.mean(random_results['returns'])), 1e-6)
    improvement_distance = np.mean(trained_results['distances']) - np.mean(random_results['distances'])

    print(f"\n  {'Metric':<25} {'Random':>12} {'Trained':>12} {'Improvement':>15}")
    print(f"  {'-'*70}")
    print(f"  {'Return':<25} {np.mean(random_results['returns']):>12.2f} {np.mean(trained_results['returns']):>12.2f} {improvement_reward:>14.1%}")
    print(f"  {'Distance (m)':<25} {np.mean(random_results['distances']):>12.3f} {np.mean(trained_results['distances']):>12.3f} {improvement_distance:>14.3f}m")
    print(f"  {'Contacts/Step':<25} {np.mean(random_results['contacts']):>12.2f} {np.mean(trained_results['contacts']):>12.2f}")

    if improvement_distance < 0.3:
        print(f"\n  {Colors.RED}⚠ Model is not significantly better than random!{Colors.END}")
        print(f"    • Training may not be working properly")
        print(f"    • Check: reward function, network size, training duration")
    elif improvement_distance < 1.0:
        print(f"\n  {Colors.YELLOW}⚠ Model shows some improvement but needs more training{Colors.END}")
    else:
        print(f"\n  {Colors.GREEN}✓ Model is significantly better than random!{Colors.END}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test Spider Robot Model")
    parser.add_argument("--model", type=str, default=None, help="Path to model (e.g., final_model.zip)")
    parser.add_argument("--vecnorm", type=str, default=None, help="Path to vecnorm (e.g., vecnorm_final.pkl)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--no-render", action="store_true", help="Don't show GUI")
    parser.add_argument("--compare-random", action="store_true", help="Also test random policy for comparison")
    parser.add_argument("--run-dir", type=str, default=None, help="Auto-find latest run in this directory")

    args = parser.parse_args()

    print_header("🧪 SPIDER ROBOT MODEL TESTING")

    # Auto-find latest model if run-dir provided
    if args.run_dir:
        run_path = Path(args.run_dir)
        if not run_path.exists():
            print(f"{Colors.RED}✗ Run directory not found: {run_path}{Colors.END}")
            return

        args.model = str(run_path / "final_model.zip")
        args.vecnorm = str(run_path / "vecnorm_final.pkl")
        print(f"Auto-detected model from: {run_path}\n")

    # Auto-find latest in training_runs if nothing specified
    if args.model is None:
        training_runs = Path("training_runs")
        if training_runs.exists():
            runs = sorted(training_runs.glob("spider_ppo_*"), key=lambda x: x.stat().st_mtime, reverse=True)
            if runs:
                latest = runs[0]
                args.model = str(latest / "final_model.zip")
                args.vecnorm = str(latest / "vecnorm_final.pkl")
                print(f"Auto-detected latest model: {latest.name}\n")

    if args.model is None:
        print(f"{Colors.RED}✗ No model specified. Use --model or --run-dir{Colors.END}")
        print("\nUsage examples:")
        print("  python test_model.py --run-dir training_runs/spider_ppo_20241104_143522")
        print("  python test_model.py --model path/to/model.zip --vecnorm path/to/vecnorm.pkl")
        return

    # Test random baseline if requested
    random_results = None
    if args.compare_random:
        temp_env = SpiderWalkEnv(render_mode=None)
        random_results = test_random_policy(temp_env, num_episodes=3, max_steps=args.max_steps, render=False)
        temp_env.close()

    # Test trained model
    trained_results = test_trained_model(
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        render=not args.no_render,
        deterministic=True
    )

    # Compare
    if random_results and trained_results:
        compare_policies(trained_results, random_results)

    print_header("✅ TESTING COMPLETE")


if __name__ == "__main__":
    main()