from URDF_Generator import generator
generator(
    n_legs=6,
    joints_per_leg=4,
    leg_segments=[0.08, 0.07, 0.06, 0.06],
    leg_masses=[0.02, 0.015, 0.01, 0.01],
    joint_limits=[
        (-0.785, 0.785),   # coxa yaw
        (-1.57,  0.60),    # femur: allow real “down”
        (-1.57,  1.20),    # tibia: allow big flexion
        (-1.40,  1.20),    # tarsus
    ],
    filename="kit1_spider.urdf"
)
