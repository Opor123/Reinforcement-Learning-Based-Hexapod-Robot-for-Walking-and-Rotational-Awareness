"""
🕷️ Spider Robot PPO Training - Clean & Readable Output
One-click training script with beautiful console output
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import time
import warnings

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from spider_env import SpiderWalkEnv


# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Run identification
    RUN_NAME = f"spider_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    OUTPUT_DIR = Path("training_runs") / RUN_NAME
    TENSORBOARD_DIR = OUTPUT_DIR / "tensorboard"

    # Training settings
    N_ENVS = 8
    TOTAL_TIMESTEPS = 3_000_000
    N_STEPS = max(64, 2048 // max(1, N_ENVS))
    BATCH_SIZE = 512
    N_EPOCHS = 10

    # PPO hyperparameters
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    LEARNING_RATE = 3e-4
    CLIP_RANGE = 0.2
    ENT_COEF = 0.0
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    POLICY_LAYERS = [256, 256]

    # Testing
    TEST_EPISODES = 3
    TEST_RENDER = True
    TEST_MAX_STEPS = 10_000
    KEEP_GUI_OPEN = True


# ============================================================================
# Pretty Console Output Utilities
# ============================================================================

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

    @staticmethod
    def disable():
        Colors.HEADER = ''
        Colors.BLUE = ''
        Colors.CYAN = ''
        Colors.GREEN = ''
        Colors.YELLOW = ''
        Colors.RED = ''
        Colors.BOLD = ''
        Colors.UNDERLINE = ''
        Colors.END = ''


def print_header(text, char="="):
    """Print a formatted header"""
    width = 80
    print(f"\n{Colors.BOLD}{Colors.CYAN}{char * width}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(width)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{char * width}{Colors.END}\n")


def print_section(title):
    """Print a section title"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}▶ {title}{Colors.END}")


def print_info(key, value, indent=2):
    """Print key-value info"""
    spaces = " " * indent
    print(f"{spaces}{Colors.CYAN}{key}:{Colors.END} {value}")


def print_success(message):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")


def print_warning(message):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")


def print_error(message):
    """Print error message"""
    print(f"{Colors.RED}✗ {message}{Colors.END}")


def format_number(num):
    """Format large numbers with commas"""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    return str(num)


def format_time(seconds):
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


# ============================================================================
# Custom Training Callback with Clean Output
# ============================================================================

class CleanProgressCallback(BaseCallback):
    """
    Custom callback that prints clean, structured training progress
    """

    def __init__(self, total_timesteps, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.start_time = None
        self.last_print_time = 0
        self.print_interval = 5  # Print every 5 seconds

    def _on_training_start(self):
        self.start_time = time.time()
        print_header("🎓 TRAINING IN PROGRESS")
        print(f"{Colors.BOLD}Progress will update every {self.print_interval} seconds...{Colors.END}\n")

        # Print table header
        header = (
            f"{'Step':>10} │ {'Progress':>8} │ {'Ep.Len':>7} │ {'Ep.Reward':>10} │ "
            f"{'FPS':>6} │ {'ETA':>8}"
        )
        separator = "─" * len(header)
        print(f"{Colors.BOLD}{header}{Colors.END}")
        print(separator)

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        """Called after each rollout - print progress"""
        current_time = time.time()

        # Only print at intervals
        if current_time - self.last_print_time < self.print_interval:
            return True

        self.last_print_time = current_time

        # Get current stats
        timesteps = self.num_timesteps
        progress = (timesteps / self.total_timesteps) * 100

        # Get episode stats from logger
        ep_len = self.model.ep_info_buffer[-1]['l'] if self.model.ep_info_buffer else 0
        ep_rew = self.model.ep_info_buffer[-1]['r'] if self.model.ep_info_buffer else 0

        # Calculate FPS and ETA
        elapsed = current_time - self.start_time
        fps = timesteps / elapsed if elapsed > 0 else 0
        remaining_steps = self.total_timesteps - timesteps
        eta_seconds = remaining_steps / fps if fps > 0 else 0

        # Format output
        row = (
            f"{format_number(timesteps):>10} │ "
            f"{progress:>7.1f}% │ "
            f"{ep_len:>7.0f} │ "
            f"{ep_rew:>10.2f} │ "
            f"{fps:>6.0f} │ "
            f"{format_time(eta_seconds):>8}"
        )

        # Color code based on progress
        if progress < 33:
            color = Colors.RED
        elif progress < 66:
            color = Colors.YELLOW
        else:
            color = Colors.GREEN

        print(f"{color}{row}{Colors.END}")
        sys.stdout.flush()

        return True

    def _on_training_end(self):
        """Print final summary"""
        elapsed = time.time() - self.start_time
        print("\n" + "─" * 80)
        print_success(f"Training completed in {format_time(elapsed)}")

        # Final stats
        if self.model.ep_info_buffer:
            recent_rewards = [ep['r'] for ep in list(self.model.ep_info_buffer)[-100:]]
            recent_lengths = [ep['l'] for ep in list(self.model.ep_info_buffer)[-100:]]

            print(f"\n{Colors.BOLD}Final Statistics (last 100 episodes):{Colors.END}")
            print_info("Average Reward", f"{np.mean(recent_rewards):.2f} ± {np.std(recent_rewards):.2f}")
            print_info("Average Length", f"{np.mean(recent_lengths):.0f} ± {np.std(recent_lengths):.0f}")


# ============================================================================
# Environment Setup
# ============================================================================

def make_env(rank: int, seed: int = 42, render_mode=None):
    """Create a single environment"""

    def _init():
        env = SpiderWalkEnv(render_mode=render_mode)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def make_train_vecenv(n_envs: int):
    """Create vectorized training environment"""
    print_section("Creating Training Environment")

    if n_envs > 1:
        venv = SubprocVecEnv([make_env(i, seed=42, render_mode=None) for i in range(n_envs)])
        print_info("Environment Type", "Multi-process (parallel)")
    else:
        venv = DummyVecEnv([make_env(0, seed=42, render_mode=None)])
        print_info("Environment Type", "Single-process")

    print_info("Number of Envs", n_envs)

    venv = VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    print_success("Environment created and normalized")
    return venv


# ============================================================================
# Model Setup
# ============================================================================

def create_model(venv, config):
    """Create PPO model"""
    import torch as th

    print_section("Initializing PPO Model")

    device = "cpu"
    policy_kwargs = dict(
        net_arch=dict(pi=config.POLICY_LAYERS, vf=config.POLICY_LAYERS),
        activation_fn=th.nn.ReLU,
    )

    print_info("Device", device)
    print_info("Policy Network", f"{config.POLICY_LAYERS}")
    print_info("Steps per Update", config.N_STEPS)
    print_info("Batch Size", config.BATCH_SIZE)
    print_info("Learning Rate", config.LEARNING_RATE)

    model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=config.LEARNING_RATE,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        n_epochs=config.N_EPOCHS,
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        clip_range=config.CLIP_RANGE,
        ent_coef=config.ENT_COEF,
        vf_coef=config.VF_COEF,
        max_grad_norm=config.MAX_GRAD_NORM,
        verbose=0,  # Disable default verbose output
        tensorboard_log=str(config.TENSORBOARD_DIR),
        policy_kwargs=policy_kwargs,
        device=device,
    )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print_info("Total Parameters", f"{total_params:,}")

    print_success("Model initialized")
    return model


# ============================================================================
# Testing
# ============================================================================

def run_test(run_dir: Path, config):
    """Test the trained model"""
    print_header("🎮 TESTING TRAINED MODEL")

    model_path = run_dir / "final_model.zip"
    vec_path = run_dir / "vecnorm_final.pkl"

    if not model_path.exists() or not vec_path.exists():
        print_error("Model or normalization file not found")
        return

    print_info("Model", str(model_path))
    print_info("VecNormalize", str(vec_path))
    print_info("Episodes", config.TEST_EPISODES)
    print_info("Render", "Yes (GUI)" if config.TEST_RENDER else "No (headless)")

    # Create test environment
    def _make_test_env():
        env = SpiderWalkEnv(render_mode="human" if config.TEST_RENDER else None)
        return Monitor(env)

    base = DummyVecEnv([_make_test_env])
    venv = VecNormalize.load(str(vec_path), base)
    venv.training = False
    venv.norm_reward = False

    model = PPO.load(str(model_path), env=venv, device="cpu", print_system_info=False)

    print(f"\n{Colors.BOLD}Running Episodes...{Colors.END}\n")

    returns = []
    lengths = []

    for ep in range(1, config.TEST_EPISODES + 1):
        obs = venv.reset()
        done = np.array([False])
        total_reward = 0.0
        steps = 0

        while not done.any():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = venv.step(action)
            total_reward += float(reward[0])
            steps += 1

            if steps > config.TEST_MAX_STEPS:
                warnings.warn("Episode hit safety limit")
                break

        returns.append(total_reward)
        lengths.append(steps)

        print(f"  Episode {ep}: {Colors.GREEN}Return = {total_reward:>8.2f}{Colors.END}  |  Length = {steps:>4}")

    # Summary
    print(f"\n{Colors.BOLD}Test Summary:{Colors.END}")
    print_info("Episodes", config.TEST_EPISODES)
    print_info("Avg Return", f"{np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print_info("Avg Length", f"{np.mean(lengths):.0f} ± {np.std(lengths):.0f}")

    if config.TEST_RENDER and config.KEEP_GUI_OPEN:
        print(f"\n{Colors.YELLOW}GUI is open. Press Q to close and exit.{Colors.END}")
        _keep_gui_open()

    venv.close()


def _keep_gui_open():
    """Keep PyBullet GUI open until user presses Q"""
    try:
        import pybullet as p
        if p.isConnected():
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            try:
                p.resetDebugVisualizerCamera(distance=1.2, yaw=45, pitch=-30, targetPosition=[0, 0, 0.1])
            except:
                pass

            while p.isConnected():
                keys = p.getKeyboardEvents()
                if ord('q') in keys or ord('Q') in keys:
                    break
                time.sleep(0.01)
    except:
        pass


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    """Main training pipeline"""

    # Disable colors on Windows if terminal doesn't support ANSI
    if sys.platform == "win32":
        try:
            import colorama
            colorama.init()
        except ImportError:
            Colors.disable()

    config = Config()

    # Create output directories
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)

    # Print banner
    print_header("🕷️  SPIDER ROBOT - PPO TRAINING")

    print_section("Configuration")
    print_info("Run Name", config.RUN_NAME)
    print_info("Output Directory", str(config.OUTPUT_DIR))
    print_info("Total Timesteps", format_number(config.TOTAL_TIMESTEPS))
    print_info("Parallel Environments", config.N_ENVS)
    print_info("Test After Training", f"{config.TEST_EPISODES} episodes")

    # Create environment
    vec_env = make_train_vecenv(config.N_ENVS)

    # Create model
    model = create_model(vec_env, config)

    # Create callback
    callback = CleanProgressCallback(total_timesteps=config.TOTAL_TIMESTEPS, verbose=1)

    # Train
    try:
        model.learn(
            total_timesteps=config.TOTAL_TIMESTEPS,
            callback=callback,
            log_interval=9999,  # Disable default logging
        )
    except KeyboardInterrupt:
        print_warning("\nTraining interrupted by user")
    finally:
        # Save model
        print_section("Saving Model")
        model.save(str(config.OUTPUT_DIR / "final_model"))
        vec_env.save(str(config.OUTPUT_DIR / "vecnorm_final.pkl"))
        print_success(f"Model saved to {config.OUTPUT_DIR}")

        vec_env.close()

        # Run test
        try:
            run_test(config.OUTPUT_DIR, config)
        except Exception as e:
            print_error(f"Test failed: {e}")

    # Final instructions
    print_header("✅ ALL DONE")
    print(f"\n{Colors.BOLD}To view training curves:{Colors.END}")
    print(f"  tensorboard --logdir \"{config.TENSORBOARD_DIR}\"\n")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    # Optimize CPU threading
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # Run training
    main()