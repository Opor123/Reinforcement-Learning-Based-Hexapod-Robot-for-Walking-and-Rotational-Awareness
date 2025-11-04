import os, json
import numpy as np
import pybullet as p
import pybullet_data

# Recalculate correct spawn height
client = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

plane = p.loadURDF("plane.urdf")

urdf_path = os.path.abspath("kit1_spider.urdf")

# Load robot at a safe high position
robot = p.loadURDF(urdf_path, [0, 0, 1.0])

# Get joints
joint_indices = []
joint_limits = []
for j in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, j)
    if info[2] == p.JOINT_REVOLUTE:
        joint_indices.append(j)
        joint_limits.append((info[8], info[9]))

# Set to grounded stance
n_legs = 6
joints_per_leg = 4
stance = np.zeros(len(joint_indices))

# More aggressive leg bending to ensure feet reach down
femur_d, tibia_d, tarsus_d = -1.2, 1.1, -0.8

for leg in range(n_legs):
    b = leg * joints_per_leg
    coxa_yaw = 0.15 if (leg % 2 == 0) else -0.15
    stance[b+0] = coxa_yaw
    stance[b + 1] = max(femur_d, joint_limits[b+1][0])  # Clamp to limits
    stance[b + 2] = min(tibia_d, joint_limits[b+2][1])
    stance[b + 3] = max(tarsus_d, joint_limits[b+3][0])

print(f"Stance: {stance}")

# Set stance
for i, q in zip(joint_indices, stance):
    p.resetJointState(robot, i, float(q))

# Get the lowest point of the entire robot
base_pos, _ = p.getBasePositionAndOrientation(robot)
print(f"Base at: z={base_pos[2]:.4f}m")

aabb_min, aabb_max = p.getAABB(robot, -1)
lowest_z = aabb_min[2]
print(f"Robot AABB: min_z={lowest_z:.4f}m, max_z={aabb_max[2]:.4f}m")

# Check each foot
foot_links = [leg * joints_per_leg + (joints_per_leg - 1) for leg in range(n_legs)]
foot_positions = []
for foot_link in foot_links:
    if foot_link < p.getNumJoints(robot):
        link_state = p.getLinkState(robot, foot_link)
        foot_pos = link_state[0][2]  # z position
        foot_positions.append(foot_pos)
        print(f"  Foot link {foot_link} at z={foot_pos:.4f}m")

# The lowest foot should be the reference
lowest_foot = min(foot_positions)
print(f"\nLowest foot at: z={lowest_foot:.4f}m")

# Calculate spawn height: we want lowest point at target_clearance above ground
target_clearance = 0.005  # 5mm above ground
base_to_lowest = base_pos[2] - lowest_z
spawn_height = target_clearance + base_to_lowest

print(f"\n📏 Calculations:")
print(f"  Base z: {base_pos[2]:.4f}m")
print(f"  Lowest point: {lowest_z:.4f}m")
print(f"  Offset (base to lowest): {base_to_lowest:.4f}m")
print(f"  Target clearance: {target_clearance:.4f}m")
print(f"  → Spawn height: {spawn_height:.4f}m")

# Save to metadata
meta_file = urdf_path.replace('.urdf', '_meta.json')
with open(meta_file, 'w') as f:
    json.dump({'spawn_height': float(spawn_height)}, f, indent=2)

print(f"\n✅ Saved spawn height {spawn_height:.4f}m to {meta_file}")

# Verify by loading at the calculated height
p.removeBody(robot)
robot = p.loadURDF(urdf_path, [0, 0, spawn_height])

# Set stance again
for i, q in zip(joint_indices, stance):
    p.resetJointState(robot, i, float(q))

# Check clearance
aabb_min_new, _ = p.getAABB(robot, -1)
actual_clearance = aabb_min_new[2]

print(f"\n🔍 Verification:")
print(f"  Loaded robot at spawn_height={spawn_height:.4f}m")
print(f"  Actual lowest point: {actual_clearance:.4f}m")
print(f"  Target was: {target_clearance:.4f}m")
print(f"  Error: {abs(actual_clearance - target_clearance):.4f}m")

if abs(actual_clearance - target_clearance) < 0.002:
    print("  ✅ Spawn height is correct!")
else:
    print("  ⚠️  Small error, but should be OK")

p.disconnect()