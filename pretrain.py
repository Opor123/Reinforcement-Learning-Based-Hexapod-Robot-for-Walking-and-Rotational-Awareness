"""
Generate expert demonstrations from the working tripod gait
Then use behavior cloning to pre-train the policy
"""

import numpy as np
import pickle
from pathlib import Path
from spider_env import SpiderWalkEnv


def collect_expert_demonstrations(num_episodes=50, steps_per_episode=200):
    """
    Collect demonstrations using the hand-crafted tripod gait
    """
    print(f"Collecting {num_episodes} expert demonstrations...")

    env = SpiderWalkEnv(render_mode=None)

    demonstrations = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'episode_starts': []
    }

    # Tripod gait parameters (from your working example)
    tripod_A = {0, 3, 4}
    tripod_B = {1, 2, 5}
    A = np.array([0.05, 0.20, 0.22, 0.14], dtype=np.float32)
    DC = np.array([0.02, 0.06, -0.02, -0.04], dtype=np.float32)
    duty = 0.58
    phase_lead_tibia = np.deg2rad(12)
    phase_lead_tarsus = np.deg2rad(6)

    def rect_wave(phase, duty):
        x = (phase % (2 * np.pi)) / (2 * np.pi)
        return 2.0 * (x < duty) - 1.0

    total_distance = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        demonstrations['episode_starts'].append(len(demonstrations['observations']))

        ep_distance = 0
        start_x = info.get('distance_traveled', 0)

        for t in range(steps_per_episode):
            # Generate expert action (tripod gait)
            s = 2 * np.pi * (t / 60)  # ~1Hz gait

            act = np.zeros(env.num_joints, dtype=np.float32)

            for leg in range(env.n_legs):
                base = leg * env.joints_per_leg
                leg_phase = s + (np.pi if leg in tripod_B else 0.0)
                gate = rect_wave(leg_phase, duty)
                in_stance = (gate > 0.0)

                femur = A[1] * (0.6 * gate + 0.4 * np.sin(leg_phase))
                tibia = A[2] * (0.6 * gate + 0.4 * np.sin(leg_phase + phase_lead_tibia))
                tarsus = A[3] * (0.5 * gate + 0.5 * np.sin(leg_phase + phase_lead_tarsus))
                coxa = (A[0] * 0.4) if in_stance else 0.0

                act[base + 0] = DC[0] + coxa
                act[base + 1] = DC[1] + femur
                act[base + 2] = DC[2] + tibia
                act[base + 3] = DC[3] + tarsus

            # Store demonstration
            demonstrations['observations'].append(obs)
            demonstrations['actions'].append(act)

            # Step environment
            obs, reward, done, trunc, info = env.step(act)
            demonstrations['rewards'].append(reward)

            if done or trunc:
                break

        ep_distance = info.get('distance_traveled', 0) - start_x
        total_distance += ep_distance

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{num_episodes}: distance = {ep_distance:.3f}m")

    env.close()

    avg_distance = total_distance / num_episodes
    print(f"\n✓ Collected {len(demonstrations['observations'])} transitions")
    print(f"  Average distance per episode: {avg_distance:.3f}m")

    # Convert to arrays
    demonstrations['observations'] = np.array(demonstrations['observations'])
    demonstrations['actions'] = np.array(demonstrations['actions'])
    demonstrations['rewards'] = np.array(demonstrations['rewards'])
    demonstrations['episode_starts'] = np.array(demonstrations['episode_starts'])

    return demonstrations


def pretrain_with_demonstrations(demonstrations, epochs=20, policy_layers=[256, 256]):
    """
    Pre-train policy using behavior cloning on expert demonstrations

    Args:
        demonstrations: Expert demonstration data
        epochs: Number of training epochs
        policy_layers: Network architecture (MUST match Training.py config!)
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    import torch

    print("\n" + "="*80)
    print("PRE-TRAINING WITH BEHAVIOR CLONING")
    print("="*80)

    # Create dummy environment for policy structure
    def make_env():
        env = SpiderWalkEnv(render_mode=None)
        return Monitor(env)

    vec_env = DummyVecEnv([make_env])

    # CRITICAL: Match the network architecture from Training.py
    policy_kwargs = dict(
        net_arch=dict(pi=policy_layers, vf=policy_layers),
        activation_fn=torch.nn.ReLU,
    )

    print(f"Creating model with network architecture: {policy_layers}")

    # Create PPO model with matching architecture
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=1e-3,
        verbose=0,
        device='cpu',
        policy_kwargs=policy_kwargs  # ← This is the critical addition!
    )

    # Behavior cloning training (ensure CPU)
    device = torch.device('cpu')
    obs = torch.FloatTensor(demonstrations['observations']).to(device)
    actions = torch.FloatTensor(demonstrations['actions']).to(device)

    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3)

    print(f"\nTraining on {len(obs)} expert transitions for {epochs} epochs...")

    batch_size = 256
    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0

        # Shuffle data
        indices = torch.randperm(len(obs))

        for i in range(0, len(obs), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_obs = obs[batch_indices]
            batch_actions = actions[batch_indices]

            # Forward pass
            predicted_actions, _, _ = model.policy(batch_obs)

            # MSE loss (behavior cloning)
            loss = torch.nn.functional.mse_loss(predicted_actions, batch_actions)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss = {epoch_loss/n_batches:.6f}")

    vec_env.close()

    print("\n✓ Behavior cloning complete!")
    return model


def main():
    """
    Main imitation learning pipeline
    """
    print("="*80)
    print("IMITATION LEARNING BOOTSTRAP FOR SPIDER ROBOT")
    print("="*80)

    # IMPORTANT: Match network architecture from Training.py
    POLICY_LAYERS = [256, 256]  # ← Must match Config.POLICY_LAYERS in Training.py

    print(f"\nNetwork architecture: {POLICY_LAYERS}")
    print("(This MUST match your Training.py configuration!)\n")

    # Step 1: Collect expert demonstrations
    demo_file = Path("expert_demonstrations.pkl")

    if not demo_file.exists():
        print("\n📊 Collecting expert demonstrations from tripod gait...")
        demonstrations = collect_expert_demonstrations(num_episodes=50, steps_per_episode=200)

        # Save demonstrations
        with open(demo_file, 'wb') as f:
            pickle.dump(demonstrations, f)
        print(f"✓ Saved demonstrations to {demo_file}")
    else:
        print(f"\n📁 Loading existing demonstrations from {demo_file}")
        with open(demo_file, 'rb') as f:
            demonstrations = pickle.load(f)
        print(f"✓ Loaded {len(demonstrations['observations'])} transitions")

    # Step 2: Pre-train policy with behavior cloning (with correct architecture!)
    print("\n🧠 Pre-training policy with behavior cloning...")
    model = pretrain_with_demonstrations(
        demonstrations,
        epochs=20,
        policy_layers=POLICY_LAYERS  # ← Pass architecture
    )

    # Step 3: Save pre-trained model
    output_dir = Path("pretrained_model")
    output_dir.mkdir(exist_ok=True)
    model.save(output_dir / "pretrained_policy")
    print(f"\n✓ Saved pre-trained model to {output_dir}/pretrained_policy.zip")

    print("\n" + "="*80)
    print("✅ IMITATION LEARNING COMPLETE!")
    print("="*80)
    print("\nThe pre-trained model now matches your Training.py architecture.")
    print("You can now run Training.py and it will successfully load the weights!")
    print("\nThe robot will start with knowledge of the tripod gait. 🎉")


if __name__ == "__main__":
    main()