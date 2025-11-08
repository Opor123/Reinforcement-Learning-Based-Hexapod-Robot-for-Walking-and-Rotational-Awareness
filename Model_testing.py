import random,torch
import argparse
import sys
import time
from pathlib import Path
import numpy as np
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from spider_env import SpiderWalkEnv
import pybullet as p

def reseed_vecenv(vec_env, base_seed, ep):
    """Seed underlying env(s) and action spaces (works with VecNormalize/DummyVecEnv)."""
    if base_seed is None:
        return
    seed_ep = int(base_seed) * 1000 + int(ep)
    try:
        # Works through wrappers (VecNormalize) down to DummyVecEnv -> envs[i].seed(...)
        vec_env.env_method("seed", seed_ep)
    except Exception:
        # Fallback: touch each sub-env directly
        for i, e in enumerate(getattr(vec_env, "envs", [])):
            try:
                e.reset(seed=seed_ep + i)
            except TypeError:
                try:
                    e.seed(seed=seed_ep + i)
                except Exception:
                    pass
    # Seed action spaces for determinism
    try:
        for i, e in enumerate(getattr(vec_env, "envs", [])):
            e.action_space.seed(seed_ep + i)
    except Exception:
        pass



class Colors:
    """Pretty terminal colors"""
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def find_all_runs():
    """Find all training runs sorted by date"""
    runs_dir = Path("training_runs")
    if not runs_dir.exists():
        return []

    runs = []
    for run_path in runs_dir.glob("spider_ppo_*"):
        if run_path.is_dir() and (run_path / "final_model.zip").exists():
            # Parse timestamp from folder name
            try:
                timestamp_str = run_path.name.replace("spider_ppo_", "")
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                runs.append((run_path, timestamp))
            except:
                runs.append((run_path, datetime.min))

    # Sort by timestamp (newest first)
    runs.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in runs]


def list_available_runs():
    """Display all available trained models"""
    runs = find_all_runs()

    if not runs:
        print(f"{Colors.RED}❌ No trained models found in training_runs/{Colors.END}")
        print("\nRun 'python Training.py' to train a model first.")
        return

    print(f"\n{Colors.BOLD}{Colors.CYAN}Available Trained Models:{Colors.END}\n")

    for i, run in enumerate(runs, 1):
        # Get timestamp
        timestamp_str = run.name.replace("spider_ppo_", "")
        try:
            dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            date_str = "Unknown date"

        # Check if vecnorm exists
        has_vecnorm = (run / "vecnorm_final.pkl").exists()
        vecnorm_str = f"{Colors.GREEN}✓{Colors.END}" if has_vecnorm else f"{Colors.YELLOW}⚠{Colors.END}"

        marker = f"{Colors.GREEN}➤{Colors.END}" if i == 1 else " "
        print(f"  {marker} [{i}] {run.name}")
        print(f"      Date: {date_str}")
        print(f"      VecNormalize: {vecnorm_str}")

    print(f"\n{Colors.BOLD}Latest model will be tested by default{Colors.END}")
    print(f"Use: python test.py --run {runs[0].name}")
    print()


def test_model(run_path, episodes=5, render=True, deterministic=True, verbose=True,seed=None):
    """Test a trained model"""

    model_path = run_path / "final_model.zip"
    vecnorm_path = run_path / "vecnorm_final.pkl"



    if not model_path.exists():
        print(f"{Colors.RED}❌ Model not found: {model_path}{Colors.END}")
        return None

    if verbose:
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'🧪 TESTING SPIDER ROBOT'.center(80)}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.END}\n")

        print(f"{Colors.CYAN}Model:{Colors.END} {run_path.name}")
        print(f"{Colors.CYAN}Episodes:{Colors.END} {episodes}")
        print(f"{Colors.CYAN}Mode:{Colors.END} {'Deterministic' if deterministic else 'Stochastic'}")
        print(f"{Colors.CYAN}Render:{Colors.END} {'Yes (GUI)' if render else 'No (Headless)'}")
        print()

    # Create environment
    env = SpiderWalkEnv(render_mode="human" if render else None,
                        enable_watchdog=False,
                        eval_mode=True)
    vec_env = DummyVecEnv([lambda: env])

    # Load VecNormalize if available
    if vecnorm_path.exists():
        vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        _=vec_env.reset()
        if verbose:
            print(f"{Colors.GREEN}✓{Colors.END} Loaded VecNormalize statistics")
    else:
        if verbose:
            print(f"{Colors.YELLOW}⚠{Colors.END} No VecNormalize found (training may have been interrupted)")

    if seed is not None:
        set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        try:
            torch.manual_seed(seed)
        except:
            pass
        try:
            vec_env.seed(seed)
        except:
            pass

    # Load model
    model = PPO.load(str(model_path), device='cpu')
    if verbose:
        print(f"{Colors.GREEN}✓{Colors.END} Loaded model\n")
        print(f"{Colors.BOLD}Running episodes...{Colors.END}\n")

    # Track results
    results = {
        'returns': [],
        'distances': [],
        'lengths': [],
        'contacts': [],
        'success': []  # Episodes with >0.5m distance
    }

    # Run episodes
    for ep in range(1, episodes + 1):
        reseed_vecenv(vec_env, seed, ep)
        obs = vec_env.reset()
        done = np.array([False])

        ep_return = 0.0
        ep_length = 0

        total_contacts = 0

        last_distance = 0.0
        last_info = {}

        max_distance=0.0

        # Run episode
        while not done.any():
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = vec_env.step(action)

            ep_return += float(reward[0])
            ep_length += 1

            last_info=info[0] if isinstance(info,(list,tuple)) and len(info) else {}

            if "distance_traveled" in last_info:
                last_distance = float(last_info["distance_traveled"])

            if "contacts" in last_info:
                total_contacts += float(last_info["contacts"])

            if "distance_traveled" in last_info:
                last_distance=float(last_info["distance_traveled"])
                if last_distance>max_distance:
                    max_distance=last_distance


        # Get final position
        distance = last_distance
        avg_contacts = total_contacts / ep_length if ep_length > 0 else 0

        # Store results
        results['returns'].append(ep_return)
        results['distances'].append(distance)
        results.setdefault('max_distances', []).append(max_distance)
        results['lengths'].append(ep_length)
        results['contacts'].append(avg_contacts)
        results['success'].append(distance > 0.5)

        # Print episode result
        status = (f"{Colors.GREEN}✓ GREAT{Colors.END}" if distance > 1.0
                  else f"{Colors.GREEN}✓ Good{Colors.END} " if distance > 0.5
                  else f"{Colors.YELLOW}○ Okay{Colors.END} " if distance > 0.1
                  else f"{Colors.RED}✗ Poor{Colors.END} ")

        term = last_info.get("termination_reason", "")
        term_str = f" | Term={term}" if term else ""

        print(f"  {status} Ep {ep:2d}: Distance={last_distance:6.3f}m | "
              f"MaxDist={max_distance:6.3f}m | Reward={ep_return:7.2f} | "
              f"Steps={ep_length:4d} | Contacts={avg_contacts:.1f}/6{term_str}")

    # Print summary
    if verbose:
        print(f"\n{Colors.BOLD}{'=' * 80}{Colors.END}")
        print(f"{Colors.BOLD}SUMMARY{Colors.END}")
        print(f"{Colors.BOLD}{'=' * 80}{Colors.END}\n")

        avg_dist = np.mean(results['distances'])
        max_dist = np.max(results['distances'])
        min_dist = np.min(results['distances'])
        success_rate = sum(results['success']) / len(results['success']) * 100
        avg_max_dist = np.mean(results.get('max_distances', results['distances']))
        max_of_max = np.max(results.get('max_distances', results['distances']))
        min_of_max = np.min(results.get('max_distances', results['distances']))

        print(f"  {'Distance (avg)':<25} {avg_dist:6.3f}m ± {np.std(results['distances']):5.3f}m")
        print(f"  {'Distance (max)':<25} {max_dist:6.3f}m")
        print(f"  {'Distance (min)':<25} {min_dist:6.3f}m")
        print(f"  {'Success Rate (>0.5m)':<25} {success_rate:5.1f}%")
        print(f"  {'Avg Return':<25} {np.mean(results['returns']):7.2f} ± {np.std(results['returns']):6.2f}")
        print(f"  {'Avg Episode Length':<25} {np.mean(results['lengths']):6.0f} steps")
        print(f"  {'Avg Contacts':<25} {np.mean(results['contacts']):4.2f}/6")

        print(f"  {'Max Distance (avg)':<25} {avg_max_dist:6.3f}m")
        print(f"  {'Max Distance (max)':<25} {max_of_max:6.3f}m")
        print(f"  {'Max Distance (min)':<25} {min_of_max:6.3f}m")

        print()

        # Diagnosis
        if avg_dist > 1.0:
            print(f"{Colors.GREEN}✓ EXCELLENT{Colors.END} - Robot walks consistently well!")
        elif avg_dist > 0.5:
            print(f"{Colors.GREEN}✓ GOOD{Colors.END} - Robot walks, but could be more consistent")
        elif avg_dist > 0.1:
            print(f"{Colors.YELLOW}⚠ FAIR{Colors.END} - Robot moves but struggles")
            print("  Consider: More training time or reward tuning")
        else:
            print(f"{Colors.RED}✗ POOR{Colors.END} - Robot not moving forward reliably")
            print("  Consider: Check reward function, increase training")

        if success_rate < 50:
            print(f"\n{Colors.YELLOW}⚠{Colors.END} Low success rate - high variance in performance")
            print("  The robot learned to walk but needs more training for consistency")

        print(f"\n{Colors.BOLD}{'=' * 80}{Colors.END}\n")

    if render:
        print(f"{Colors.YELLOW}Press Q in the PyBullet window to close{Colors.END}\n")

    try:
        vec_env.close()
        p.disconnect()
    except Exception:
        pass

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Test trained Spider Robot models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test.py                    # Test latest model with GUI
  python test.py --episodes 10      # Test with 10 episodes  
  python test.py --no-render        # Test without GUI (faster)
  python test.py --list             # List all available models
  python test.py --run spider_ppo_20251107_123255  # Test specific model
        """
    )

    parser.add_argument('--run', type=str, default=None,
                        help='Specific run to test (default: latest)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to test (default: 5)')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable GUI rendering (faster testing)')
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic policy instead of deterministic')
    parser.add_argument('--list', action='store_true',
                        help='List all available trained models')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for env & policy')


    args = parser.parse_args()

    # List models if requested
    if args.list:
        list_available_runs()
        return

    # Find runs
    runs = find_all_runs()

    if not runs:
        print(f"{Colors.RED}❌ No trained models found!{Colors.END}")
        print("\nTo train a model, run: python Training.py")
        sys.exit(1)

    # Select run
    if args.run:
        # User specified a run
        run_path = Path("training_runs") / args.run
        if not run_path.exists():
            # Try exact path
            run_path = Path(args.run)

        if not run_path.exists() or not (run_path / "final_model.zip").exists():
            print(f"{Colors.RED}❌ Run not found: {args.run}{Colors.END}")
            print(f"\nAvailable runs:")
            for r in runs[:5]:  # Show first 5
                print(f"  - {r.name}")
            sys.exit(1)
    else:
        # Use latest run
        run_path = runs[0]
        print(f"{Colors.CYAN}Using latest model: {run_path.name}{Colors.END}")

    # Test the model
    try:
        test_model(
            run_path=run_path,
            episodes=args.episodes,
            render=not args.no_render,
            deterministic=not args.stochastic,
            verbose=True,
            seed=args.seed,
        )
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Testing interrupted by user{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error during testing: {e}{Colors.END}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()