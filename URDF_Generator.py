def generator(
        n_legs=6,
        joints_per_leg=4,
        body_radius=0.08,
        body_height=0.03,
        body_mass=1.0,
        leg_segments=[0.06, 0.05, 0.04, 0.04],
        leg_masses=[0.02, 0.015, 0.01, 0.01],
        joint_limits=[(-0.785, 0.785), (-1.57, 0.785), (-0.785, 1.57), (-0.785, 1.57)],
        filename="kit1_spider.urdf",
        target_clearance=0.05
):
    import math, tempfile, pybullet as p, pybullet_data, os, json

    # ---------- Step 1: Generate raw URDF text ----------
    if len(leg_segments) < joints_per_leg:
        leg_segments = leg_segments + [leg_segments[-1]] * (joints_per_leg - len(leg_segments))
    if len(leg_masses) < joints_per_leg:
        leg_masses = leg_masses + [leg_masses[-1]] * (joints_per_leg - len(leg_masses))
    if len(joint_limits) < joints_per_leg:
        joint_limits = joint_limits + [joint_limits[-1]] * (joints_per_leg - len(joint_limits))

    xml = ['<?xml version="1.0"?>', '<robot name="spider_bot">']

    # Body
    xml.append('  <link name="base_link">')
    xml.append('    <inertial>')
    xml.append(f'      <mass value="{body_mass}"/>')
    ixx = iyy = body_mass * (3 * body_radius ** 2 + body_height ** 2) / 12
    izz = body_mass * body_radius ** 2 / 2
    xml.append(f'      <inertia ixx="{ixx}" iyy="{iyy}" izz="{izz}" ixy="0" ixz="0" iyz="0"/>')
    xml.append('    </inertial>')
    xml.append('    <visual>')
    xml.append('      <geometry>')
    xml.append(f'        <cylinder radius="{body_radius}" length="{body_height}"/>')
    xml.append('      </geometry>')
    xml.append('      <material name="body_color"><color rgba="0.3 0.3 0.3 1"/></material>')
    xml.append('    </visual>')
    xml.append('    <collision>')
    xml.append('      <geometry>')
    xml.append(f'        <cylinder radius="{body_radius}" length="{body_height}"/>')
    xml.append('      </geometry>')
    xml.append('    </collision>')
    xml.append('  </link>')

    # Legs
    for leg_idx in range(n_legs):
        angle = 2 * math.pi * leg_idx / n_legs
        rim = body_radius + 0.005
        attach_x = rim * math.cos(angle)
        attach_y = rim * math.sin(angle)
        attach_z = -body_height * 0.5
        parent_link = "base_link"

        for seg_idx in range(joints_per_leg):
            link_name = f"leg{leg_idx}_seg{seg_idx}"
            joint_name = f"leg{leg_idx}_joint{seg_idx}"
            seg_length = leg_segments[seg_idx]
            seg_mass = leg_masses[seg_idx]
            seg_radius = 0.008

            xml.append(f'  <joint name="{joint_name}" type="revolute">')
            xml.append(f'    <parent link="{parent_link}"/>')
            xml.append(f'    <child link="{link_name}"/>')

            if seg_idx == 0:
                xml.append(f'    <origin xyz="{attach_x} {attach_y} {attach_z}" rpy="0 0 {angle}"/>')
                xml.append('    <axis xyz="0 0 1"/>')
            elif seg_idx == 1:
                prev_len = leg_segments[seg_idx - 1]
                xml.append(f'    <origin xyz="{prev_len} 0 0" rpy="0 0 0"/>')
                xml.append('    <axis xyz="0 1 0"/>')
            else:
                prev_len = leg_segments[seg_idx - 1]
                xml.append(f'    <origin xyz="{prev_len} 0 0" rpy="0 0 0"/>')
                xml.append('    <axis xyz="0 1 0"/>')

            lower, upper = joint_limits[seg_idx]
            xml.append(f'    <limit lower="{lower}" upper="{upper}" effort="6.0" velocity="20.0"/>')
            xml.append('    <dynamics damping="0.06" friction="0.03"/>')
            xml.append('  </joint>')

            # Link definition
            xml.append(f'  <link name="{link_name}">')
            xml.append('    <inertial>')
            xml.append(f'      <origin xyz="{seg_length / 2} 0 0"/>')
            xml.append(f'      <mass value="{seg_mass}"/>')
            ixx_seg = seg_mass * seg_radius ** 2 / 2
            iyy_seg = izz_seg = seg_mass * (3 * seg_radius ** 2 + seg_length ** 2) / 12
            xml.append(f'      <inertia ixx="{ixx_seg}" iyy="{iyy_seg}" izz="{izz_seg}" ixy="0" ixz="0" iyz="0"/>')
            xml.append('    </inertial>')
            xml.append('    <visual>')
            xml.append(f'      <origin xyz="{seg_length / 2} 0 0" rpy="0 1.5708 0"/>')
            xml.append('      <geometry>')
            xml.append(f'        <cylinder radius="{seg_radius}" length="{seg_length}"/>')
            xml.append('      </geometry>')
            xml.append(
                f'      <material name="leg{seg_idx}_color"><color rgba="{0.8 - seg_idx * 0.2} 0.4 0.2 1"/></material>')
            xml.append('    </visual>')
            xml.append('    <collision>')
            xml.append(f'      <origin xyz="{seg_length / 2} 0 0" rpy="0 1.5708 0"/>')
            xml.append('      <geometry>')
            xml.append(f'        <cylinder radius="{seg_radius}" length="{seg_length}"/>')
            xml.append('      </geometry>')
            xml.append('    </collision>')

            # Add foot sphere on last segment
            if seg_idx == joints_per_leg - 1:
                sole_r = 0.025
                sole_drop = 0.120
                tip_x = leg_segments[seg_idx] + 0.03

                xml.append('    <!-- Foot sole: visual -->')
                xml.append('    <visual>')
                xml.append(f'      <origin xyz="{tip_x} 0 {-sole_drop}" rpy="0 0 0"/>')
                xml.append('      <geometry>')
                xml.append(f'        <sphere radius="{sole_r}"/>')
                xml.append('      </geometry>')
                xml.append('      <material name="sole_color"><color rgba="0.1 0.6 0.1 1"/></material>')
                xml.append('    </visual>')

                xml.append('    <!-- Foot sole: collision -->')
                xml.append('    <collision>')
                xml.append(f'      <origin xyz="{tip_x} 0 {-sole_drop}" rpy="0 0 0"/>')
                xml.append('      <geometry>')
                xml.append(f'        <sphere radius="{sole_r}"/>')
                xml.append('      </geometry>')
                xml.append('    </collision>')

            xml.append('  </link>')
            parent_link = link_name

    xml.append('</robot>')

    # ---------- Step 2: Auto-calibrate base height ----------
    tmpfile = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False).name
    with open(tmpfile, "w") as f:
        f.write("\n".join(xml))

    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    plane = p.loadURDF("plane.urdf")
    robot = p.loadURDF(tmpfile, [0, 0, 1.0])  # Start very high to measure leg extension

    # Set all joints to lower limits (legs down)
    num_joints = p.getNumJoints(robot)
    for j in range(num_joints):
        info = p.getJointInfo(robot, j)
        if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            lower_limit = info[8]
            p.resetJointState(robot, j, lower_limit)

    # Measure without physics - just get the geometry
    base_pos, _ = p.getBasePositionAndOrientation(robot)
    base_z = base_pos[2]

    # Get lowest point of the robot in this configuration
    aabb_min, aabb_max = p.getAABB(robot, -1)
    lowest_z = aabb_min[2]

    # Calculate the offset from base to lowest point
    base_to_lowest = base_z - lowest_z

    # Spawn height should position the base so that lowest point is at target_clearance
    spawn_height = target_clearance + base_to_lowest

    p.disconnect()
    os.unlink(tmpfile)

    print(f"✅ Calculated spawn height = {spawn_height:.4f} m (base_to_lowest offset = {base_to_lowest:.4f}m)")

    # ---------- Step 3: Save URDF and metadata ----------
    with open(filename, "w") as f:
        f.write("\n".join(xml))

    # Save spawn height to companion file
    meta_file = filename.replace('.urdf', '_meta.json')
    with open(meta_file, 'w') as f:
        json.dump({'spawn_height': spawn_height}, f)

    print(f"✓ Generated {filename} with {n_legs} legs ({joints_per_leg} joints each)")
    print(f"✓ Saved spawn height to {meta_file}")
    return filename, spawn_height


if __name__ == "__main__":
    generator(filename="kit1_spider.urdf")