import numpy as np
from spider_env import SpiderWalkEnv
import time


def test_simple_forward_push():
    """Test if pushing coxa joints forward moves the robot"""

    print("=" * 80)
    print("TESTING PHYSICAL FORWARD MOTION")
    print("=" * 80)

    env = SpiderWalkEnv(render_mode="human")
    obs, info = env.reset()

    print("\nTest 1: Constant forward push on all coxa joints")
    print("Robot should slide/shuffle forward if physics allows it")
    print("-" * 80)

    start_x = info.get('distance_traveled', 0)
    total_reward = 0

    for step in range(300):  # 5 seconds
        # Simple action: push all coxa joints forward
        action = np.zeros(env.action_space.shape)
        for leg in range(6):
            action[leg * 4] = 0.5  # Moderate forward push on coxa

        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward

        if step % 60 == 0:  # Every second
            vx = info.get('fwd_speed', 0)
            dist = info.get('distance_traveled', 0) - start_x
            print(f"t={step / 60:.1f}s: vx={vx:+.4f} m/s, distance={dist:+.4f}m, reward={reward:+.2f}")

        if done or trunc:
            print(f"\n⚠️ Episode ended early at step {step}")
            break

    final_dist = info.get('distance_traveled', 0) - start_x

    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print(f"Final distance: {final_dist:.4f}m")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward/step: {total_reward / max(step, 1):.2f}")

    if abs(final_dist) < 0.001:
        print("\n❌ CRITICAL PROBLEM: Robot did NOT move at all!")
        print("   This suggests:")
        print("   1. Physics issue (friction too high, robot too heavy)")
        print("   2. URDF issue (joints not configured correctly)")
        print("   3. Action scaling issue (actions too small)")
        print("\n   The RL algorithm can't learn if the robot can't physically move!")
    elif final_dist < -0.01:
        print("\n⚠️  Robot moved BACKWARD")
        print("   Check if coxa joint directions are correct in URDF")
    elif final_dist < 0.05:
        print("\n⚠️  Robot barely moved (< 5cm)")
        print("   Friction or torques might be too weak")
    else:
        print("\n✅ Robot CAN move forward!")
        print("   Physics is OK - problem is in RL training")

    time.sleep(2)  # Let you see the result
    env.close()

    return final_dist


def test_alternating_legs():
    """Test a simple alternating leg pattern"""

    print("\n" + "=" * 80)
    print("TEST 2: ALTERNATING LEG PATTERN")
    print("=" * 80)

    env = SpiderWalkEnv(render_mode="human")
    obs, info = env.reset()

    start_x = info.get('distance_traveled', 0)

    for step in range(360):  # 6 seconds
        action = np.zeros(env.action_space.shape)

        # Alternate which legs push forward
        phase = (step // 30) % 2  # Switch every 0.5 seconds

        for leg in range(6):
            if leg % 2 == phase:
                # Push this leg forward
                action[leg * 4] = 0.6
            else:
                # Pull this leg back
                action[leg * 4] = -0.3

        obs, reward, done, trunc, info = env.step(action)

        if step % 60 == 0:
            dist = info.get('distance_traveled', 0) - start_x
            vx = info.get('fwd_speed', 0)
            print(f"t={step / 60:.1f}s: vx={vx:+.4f} m/s, distance={dist:+.4f}m")

        if done or trunc:
            break

    final_dist = info.get('distance_traveled', 0) - start_x
    print(f"\nFinal distance with alternating: {final_dist:.4f}m")

    time.sleep(2)
    env.close()

    return final_dist


if __name__ == "__main__":
    print("\n🧪 PHYSICAL MOVEMENT TEST\n")
    print("This will test if your robot can physically move forward")
    print("Watch the PyBullet window!\n")

    dist1 = test_simple_forward_push()
    dist2 = test_alternating_legs()

    print("\n" + "=" * 80)
    print("FINAL DIAGNOSIS")
    print("=" * 80)

    if abs(dist1) < 0.01 and abs(dist2) < 0.01:
        print("\n❌ ROBOT CANNOT MOVE - PHYSICS/URDF PROBLEM")
        print("\nPossible fixes:")
        print("1. Reduce friction: p.changeDynamics(plane, -1, lateralFriction=0.5)")
        print("2. Increase torque: self.max_torque = 5.0 (currently 3.0)")
        print("3. Check URDF joint limits allow forward motion")
        print("4. Increase action_scale for coxa joints")
    elif max(dist1, dist2) > 0.1:
        print("\n✅ ROBOT CAN MOVE - RL TRAINING PROBLEM")
        print("\nThe physics works! Solutions:")
        print("1. Increase exploration: ENT_COEF = 0.05")
        print("2. Add forward bias to initial actions")
        print("3. Use imitation learning from manual gait")
    else:
        print("\n⚠️  ROBOT CAN BARELY MOVE")
        print("\nTweaks needed:")
        print("1. Slightly reduce friction")
        print("2. Slightly increase torques")
        print("3. Increase exploration in RL")

    print("=" * 80)