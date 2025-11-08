import os, json
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data


class SpiderWalkEnv(gym.Env):
    """
    Hexapod walking env with heading-aligned progress, thrash watchdog,
    and PD torque control. Compatible with Gymnasium.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode="human", enable_watchdog=True, eval_mode=False):
        super().__init__()
        self.debug = False
        self.render_mode = render_mode
        self.client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Ground
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client)
        p.changeDynamics(self.plane, -1, lateralFriction=1.5, physicsClientId=self.client)

        # Robot URDF + spawn height
        self.urdf_path = os.path.abspath("kit1_spider.urdf")
        meta = self.urdf_path.replace(".urdf", "_meta.json")
        if os.path.exists(meta):
            with open(meta, "r") as f:
                self.spawn_height = float(json.load(f)["spawn_height"])
        else:
            self.spawn_height = 0.10
        print(f"📖 Loaded spawn height: {self.spawn_height:.4f} m")

        # Kinematics / actuation
        self.n_legs, self.joints_per_leg = 6, 4
        self.num_joints = self.n_legs * self.joints_per_leg
        self.kp, self.kd, self.max_torque = 1.0, 0.25, 2.0
        self.err_clip, self.qd_clip = 0.25, 4.0

        # Action/Obs spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32)
        # obs: (q_n:24) + (base_z:1) + (ori_quat:4) + (qd_n:24) + (lin:3) + (ang:3) = 59
        obs_dim = self.num_joints * 2 + 1 + 4 + 6
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Control / sim timing
        self.control_hz = 50
        self.sim_hz = 250
        self.inner_step = self.sim_hz // self.control_hz
        self.dt = 1.0 / self.control_hz

        # Action scaling (coxa, femur, tibia, tarsus) per-leg
        self.action_scale = np.array([0.20, 0.30, 0.30, 0.30] * self.n_legs, dtype=np.float32)

        # Runtime
        self.curriculum_stage = 0
        self.max_steps = int(self.control_hz * 20)  # 20 seconds max
        self._step = 0

        # Reward weights
        self.w_fwd = 10.0              # forward velocity
        self.w_fwd_dist = 25.0        # forward displacement
        self.w_backward = -20.0        # penalty coef for backward
        self.w_alive = 0.3
        self.w_tilt = 1.0
        self.w_energy = 0.0010
        self.w_smooth = 0.0040
        self.w_height = 0.8
        self.w_contact = 0.15
        self.w_lat = 0.15
        self.w_yawrate = 0.03
        self.w_vz = 0.8
        self.z_nominal = float(np.clip(self.spawn_height + 0.03, 0.05, 0.18))

        # Minor gait helpers
        self.swing_scale = np.array([1.0, 0.45, 0.35, 0.35], dtype=np.float32)
        self.stance_scale = np.ones(4, dtype=np.float32)

        # Caches and buffers
        self.stance = np.zeros(self.num_joints, dtype=np.float32)        # nominal stance pose
        self.last_action = np.zeros(self.num_joints, dtype=np.float32)   # smoothing memory
        self.foot_links = [leg * self.joints_per_leg + (self.joints_per_leg - 1)
                           for leg in range(self.n_legs)]

        # Progress windows (seconds → samples)
        self.control_window_s = 4.0
        self.progress_window = int(self.control_hz * self.control_window_s)

        # min total forward distance over window to not be "backward"
        self.min_progress_threshold = 0.025 * (self.progress_window / max(1, self.control_hz))
        self.no_progress_window_s = 2.0
        self.no_progress_eps = 0.02

        # Watchdog flags
        self.enable_watchdog = enable_watchdog and (self.render_mode != "human")
        self.eval_mode = eval_mode

        # Smoothing
        self.action_alpha = 0.3  # blend with previous action

        print(f"✅ SpiderWalkEnv initialized ({self.num_joints} joints, spawn_height={self.spawn_height:.3f}m)")

    # ---------------- core API ----------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # (Re)connect if needed
        if not p.isConnected(self.client):
            self.client = p.connect(p.GUI if self.render_mode == "human" else p.DIRECT)

        # Physics world
        p.resetSimulation(physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client)
        p.changeDynamics(self.plane, -1, lateralFriction=3.0, physicsClientId=self.client)

        # Engine params (auto-detect supported keys)
        p.setTimeStep(1.0 / self.sim_hz, physicsClientId=self.client)
        self._set_physics_params_autodetect(self.client)
        try:
            p.setPhysicsEngineParameter(
                fixedTimeStep=1.0 / self.sim_hz,
                deterministicOverlappingPairs=1,
                physicsClientId=self.client
            )
        except Exception:
            pass

        # Safe spawn height
        start_z = max(0.08, min(0.22, float(self.spawn_height)))
        self.robot = p.loadURDF(self.urdf_path, [0, 0, start_z], useFixedBase=False)

        # Heading baseline (world XY)
        pos, ori = p.getBasePositionAndOrientation(self.robot)
        self._start_pos_xy = np.array([pos[0], pos[1]], dtype=np.float32)
        self._last_pos_xy = self._start_pos_xy.copy()
        self._yaw0 = p.getEulerFromQuaternion(ori)[2]
        self._fwd0 = np.array([np.cos(self._yaw0), np.sin(self._yaw0)], dtype=np.float32)

        # Friction and slope / gravity
        mu = float(np.random.uniform(0.7, 1.5)) if not self.eval_mode else 1.0
        p.changeDynamics(self.plane, -1, lateralFriction=mu)

        if not self.eval_mode:
            slope_roll = float(np.deg2rad(np.random.uniform(-3, 3)))
            slope_pitch = float(np.deg2rad(np.random.uniform(-3, 3)))
            gx = 9.81 * np.sin(slope_pitch)
            gy = -9.81 * np.sin(slope_roll)
        else:
            gx = gy = 0.0
        gz = -9.81
        p.setGravity(gx, gy, gz)

        # Randomize masses slightly (train-time only)
        for j in range(-1, p.getNumJoints(self.robot)):
            m = p.getDynamicsInfo(self.robot, j)[0]
            m_use = float(m if self.eval_mode else (m * np.random.uniform(0.8, 1.2)))
            p.changeDynamics(self.robot, j, mass=m_use)

        # Damping etc.
        for link_idx in range(-1, p.getNumJoints(self.robot)):
            p.changeDynamics(
                self.robot, link_idx,
                linearDamping=0.06,
                angularDamping=0.12,
                restitution=0.0
            )

        # Joints & stance
        self._cache_joints()
        self.stance = self._get_grounded_stance().astype(np.float32)
        for i, q in zip(self.joint_indices, self.stance):
            p.resetJointState(self.robot, i, float(q))

        # Feet friction
        for link in self.foot_links:
            if link < p.getNumJoints(self.robot):
                p.changeDynamics(
                    self.robot, link,
                    lateralFriction=4.0,
                    rollingFriction=0.001,
                    spinningFriction=0.001,
                    restitution=0.0,
                    frictionAnchor=1
                )

        # Disable velocity motors
        for j in self.joint_indices:
            p.setJointMotorControl2(self.robot, j, p.VELOCITY_CONTROL, force=0)

        # Let it settle into stance for 1 second
        for _ in range(self.sim_hz):
            for i, qd in zip(self.joint_indices, self.stance):
                p.setJointMotorControl2(
                    self.robot, i, p.POSITION_CONTROL,
                    targetPosition=float(qd), positionGain=0.6, velocityGain=0.2, force=3.0
                )
            p.stepSimulation()

        p.resetBaseVelocity(self.robot, [0, 0, 0], [0, 0, 0])

        # Runtime caches
        self.last_action[:] = 0.0
        self._step = 0

        # Progress buffers (heading-aligned)
        self._proj_hist = []  # sliding window of per-step projected deltas
        self._dx_hist = []    # |delta| for thrash detection

        return self._get_obs(), {"base_height": p.getBasePositionAndOrientation(self.robot)[0][2]}

    def step(self, action):
        # --- Clamp and smooth action ---
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        action = self.action_alpha * action + (1.0 - self.action_alpha) * self.last_action

        # --- Read joint states ---
        q, qd = self._get_joint_state()

        # --- Target positions ---
        target = self._clip_to_limit(self.stance + self.action_scale * action)

        # --- PD torques ---
        err = np.clip(target - q, -self.err_clip, self.err_clip)
        qd = np.clip(qd, -self.qd_clip, self.qd_clip)
        tau = self.kp * err - self.kd * qd
        tau = np.clip(tau, -self.max_torque, self.max_torque)

        # --- Apply torques and simulate ---
        for i, tval in zip(self.joint_indices, tau):
            p.setJointMotorControl2(self.robot, i, p.TORQUE_CONTROL, force=float(tval))
        for _ in range(self.inner_step):
            p.stepSimulation()

        # --- Base state ---
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        pos_xy = np.array([pos[0], pos[1]], dtype=np.float32)
        base_z = float(pos[2])
        roll, pitch, yaw = self._base_rpy()
        (vx, vy, vz), (wx, wy, wz) = p.getBaseVelocity(self.robot)

        # --- Heading-aligned progress ---
        distance_delta, _ = self._projected_step(pos_xy)
        self._last_pos_xy = pos_xy.copy()
        contacts = int(self._count_leg_contacts())

        # --- Update progress windows ---
        self._proj_hist.append(distance_delta)
        if len(self._proj_hist) > self.progress_window:
            self._proj_hist.pop(0)

        self._dx_hist.append(abs(distance_delta))
        max_len = int(self.control_hz * self.no_progress_window_s)
        if len(self._dx_hist) > max_len:
            self._dx_hist.pop(0)

        # --- Progress checks ---
        no_progress = False
        if len(self._proj_hist) == self.progress_window:
            total_progress = sum(self._proj_hist)
            if total_progress < -self.min_progress_threshold:
                no_progress = True

        thrash_no_progress = (len(self._dx_hist) == max_len) and (sum(self._dx_hist) < self.no_progress_eps)
        no_progress = (no_progress or thrash_no_progress) and (not self.eval_mode) and self.enable_watchdog

        # === Reward shaping ===
        forward_vel = self._forward_speed_body()

        backstep_pen = 0.0
        if distance_delta < 0.0:
            backstep_pen += self.w_backward * (-distance_delta) * 2.0
        if forward_vel < 0.0:
            backstep_pen += self.w_backward * (-forward_vel) * self.dt * 2.0

        # Progress reward (velocity + displacement)
        velocity_reward = self.w_fwd * max(0.0, forward_vel) * self.dt
        distance_reward = self.w_fwd_dist * max(0.0, distance_delta)

        # Encourage staying near initial heading (not strict; still RL-friendly)
        w_heading = 0.05
        heading_bonus = w_heading * np.cos(yaw - self._yaw0) * self.dt

        # If moving but encoder says zero displacement this tick, discourage stalling
        if forward_vel > 0.05 and abs(distance_delta) <= 1e-4:
            distance_reward -= 0.02

        # Stability/height/contacts
        tilt = abs(roll) + abs(pitch)
        stability_reward = -self.w_tilt * tilt * self.dt
        height_error = abs(base_z - self.z_nominal)
        height_penalty = -self.w_height * height_error * self.dt
        contact_ratio = contacts / 6.0
        contact_bonus = self.w_contact * (contact_ratio - 0.5) * self.dt

        # Energetics & smoothness
        mech_power = float(np.mean(np.abs(tau * qd)))
        energy_penalty = -self.w_energy * mech_power * self.dt
        smoothness_penalty = -self.w_smooth * np.linalg.norm(action - self.last_action)

        # Lateral/yaw/vertical velocity penalties
        lateral_penalty = -self.w_lat * abs(vy) * self.dt
        yaw_penalty = -self.w_yawrate * abs(wz) * self.dt
        vz_penalty = -self.w_vz * abs(vz) * self.dt

        # Alive bonus
        alive_bonus = self.w_alive * self.dt

        # Total reward
        reward = (
            velocity_reward + distance_reward + stability_reward + height_penalty +
            contact_bonus + energy_penalty + smoothness_penalty + lateral_penalty +
            yaw_penalty + vz_penalty + backstep_pen + alive_bonus + heading_bonus
        )
        reward = float(np.clip(reward, -10.0, 50.0))

        # --- Termination / truncation ---
        done = (
            base_z < 0.008 or
            base_z > 1.0 or
            abs(roll) > 2.5 or
            abs(pitch) > 2.5 or
            (no_progress and self.enable_watchdog)
        )
        term = ""
        if base_z < 0.010:
            term = "fell_low"
        elif base_z > 0.8:
            term = "flew_high"
        elif abs(roll) > 2.0 or abs(pitch) > 2.0:
            term = "tilt"
        elif no_progress:
            term = "no_progress"

        proj_dist = float((pos_xy - self._start_pos_xy).dot(self._fwd0))

        self._step += 1
        truncated = (self._step >= self.max_steps)

        # --- Info dict ---
        info = {
            "base_height": float(base_z),
            "fwd_speed": float(forward_vel),
            "tilt": float(tilt),
            "mech_power": float(mech_power),
            "contacts": int(contacts),
            "velocity_reward": float(velocity_reward),
            "distance_reward": float(distance_reward),
            "stability_reward": float(stability_reward),
            "forward_reward": float(velocity_reward + distance_reward),
            "distance_traveled": proj_dist,             # along initial heading
            "contacts_per_step": float(contacts) / 6.0,
            "action_mag": float(np.mean(np.abs(action))),
            "torque_mag": float(np.mean(np.abs(tau))),
            "termination_reason": term,
        }

        # Cache
        self.last_action = action

        # small completion bonus when only truncated
        if truncated and not done:
            reward += 8.0

        # Gymnasium API: (obs, reward, terminated, truncated, info)
        return self._get_obs(), reward, bool(done), bool(truncated), info


    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)

    # ---------------- helpers ----------------

    def _set_physics_params_autodetect(self, client):
        desired = dict(
            numSolverIterations=180,
            solverResidualThreshold=1e-6,
            enableConeFriction=1,
            erp=0.15,
            contactERP=0.02,
            restitutionVelocityThreshold=0.01,  # silently skipped if unknown
        )
        supported = p.getPhysicsEngineParameters(physicsClientId=client)
        for k, v in desired.items():
            if k in supported:
                p.setPhysicsEngineParameter(physicsClientId=client, **{k: v})

    def _cache_joints(self):
        self.joint_indices = []
        lows, highs = [], []
        for j in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, j)
            if info[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(j)
                lows.append(info[8])
                highs.append(info[9])
        self.joint_limits = np.stack([lows, highs], axis=1)

        lo, hi = self.joint_limits[:, 0], self.joint_limits[:, 1]
        span = np.clip(hi - lo, 0.1, 6.28)
        base_frac = np.array([0.40, 0.30, 0.30, 0.30] * self.n_legs, dtype=np.float32)
        self.action_scale = (base_frac * span / 2.0).astype(np.float32)

    def _get_grounded_stance(self):
        stance = np.zeros(self.num_joints, dtype=np.float32)
        femur_d, tibia_d, tarsus_d = -0.8, 0.9, -0.6
        for leg in range(self.n_legs):
            b = leg * self.joints_per_leg
            coxa_yaw = 0.20 if (leg % 2 == 0) else -0.20
            stance[b + 0] = coxa_yaw
            stance[b + 1], stance[b + 2], stance[b + 3] = femur_d, tibia_d, tarsus_d
        return stance

    def _get_joint_state(self):
        states = p.getJointStates(self.robot, self.joint_indices)
        q = np.array([s[0] for s in states], np.float32)
        qd = np.array([s[1] for s in states], np.float32)
        return q, qd

    def _get_obs(self):
        q, qd = self._get_joint_state()
        pos, ori = p.getBasePositionAndOrientation(self.robot)
        (vx, vy, vz), (wx, wy, wz) = p.getBaseVelocity(self.robot)

        lo, hi = self.joint_limits[:, 0], self.joint_limits[:, 1]
        mid = 0.5 * (lo + hi)
        span = np.maximum(hi - lo, 1e-3)
        q_n = (q - mid) / (0.5 * span)
        qd_n = np.tanh(qd / 6.0)

        lin = np.clip([vx, vy, vz], -3.0, 3.0) / 3.0
        ang = np.clip([wx, wy, wz], -6.0, 6.0) / 6.0
        base_height = np.array([pos[2]], dtype=np.float32)
        ori_array = np.array(ori, dtype=np.float32)

        obs = np.concatenate([q_n, base_height, ori_array, qd_n, lin, ang], dtype=np.float32)
        if not self.eval_mode:
            obs = obs + np.random.normal(0.0, 0.005, size=obs.shape).astype(np.float32)

        return obs

    # --- reward helpers / kinematics ---

    def _clip_to_limit(self, target):
        lo, hi = self.joint_limits[:, 0], self.joint_limits[:, 1]
        return np.minimum(np.maximum(target, lo), hi)

    def _base_rpy(self):
        _, quat = p.getBasePositionAndOrientation(self.robot)
        return p.getEulerFromQuaternion(quat)

    def _forward_speed_body(self):
        (vx, vy, _), _ = p.getBaseVelocity(self.robot)
        _, _, yaw = self._base_rpy()
        hx, hy = np.cos(yaw), np.sin(yaw)
        return vx * hx + vy * hy

    def _count_leg_contacts(self):
        hits = 0
        for link in self.foot_links:
            pts = p.getContactPoints(self.robot, self.plane, linkIndexA=link)
            hits += int(len(pts) > 0)
        return hits

    def _projected_step(self, pos_xy):
        disp = pos_xy - self._last_pos_xy
        return float(disp.dot(self._fwd0)), disp


# ---------------- quick manual demo ----------------
if __name__ == "__main__":
    env = SpiderWalkEnv("human")
    obs, info = env.reset()
    print("Initial:", info)

    # Tripod gait: stance-only forward coxa, gentle phase leads
    T = 3.0
    hz = env.control_hz
    steps = int(T * hz)

    tripod_A = {0, 3, 4}
    tripod_B = {1, 2, 5}

    # (coxa, femur, tibia, tarsus)
    A = np.array([0.05, 0.20, 0.22, 0.14], dtype=np.float32)
    DC = np.array([0.02, 0.06, -0.02, -0.04], dtype=np.float32)
    duty = 0.58
    phase_lead_tibia = np.deg2rad(12)
    phase_lead_tarsus = np.deg2rad(6)

    def rect_wave(phase, duty_cycle):
        x = (phase % (2 * np.pi)) / (2 * np.pi)
        return 2.0 * (x < duty_cycle) - 1.0  # +1 stance, -1 swing

    rew_sum = 0.0
    for t in range(steps * 4):
        s = 2 * np.pi * (t / steps)  # 0..2π per cycle
        act = np.zeros(env.num_joints, dtype=np.float32)

        for leg in range(env.n_legs):
            base = leg * env.joints_per_leg
            leg_phase = s + (np.pi if leg in tripod_B else 0.0)

            gate = rect_wave(leg_phase, duty)
            in_stance = (gate > 0.0)

            femur = A[1] * (0.6 * gate + 0.4 * np.sin(leg_phase))
            tibia = A[2] * (0.6 * gate + 0.4 * np.sin(leg_phase + phase_lead_tibia))
            tarsus = A[3] * (0.5 * gate + 0.5 * np.sin(leg_phase + phase_lead_tarsus))
            coxa = (A[0] * 0.4) if in_stance else 0.0  # only push forward in stance

            act[base + 0] = DC[0] + coxa
            act[base + 1] = DC[1] + femur
            act[base + 2] = DC[2] + tibia
            act[base + 3] = DC[3] + tarsus

        _, r, done, trunc, info = env.step(act)
        rew_sum += r
        if t % 60 == 0:
            print(f"t={t / env.control_hz:.1f}s  vx≈{info['fwd_speed']:.3f}  "
                  f"z={info['base_height']:.3f}  c={info['contacts']}")
        if done or trunc:
            break

    print("avg step reward:", rew_sum / max(1, t + 1))
    env.close()
