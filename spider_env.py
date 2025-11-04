import os, time, json
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data

class SpiderWalkEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode="human"):
        super().__init__()
        self.debug=False
        self.render_mode = render_mode
        self.client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client)
        p.changeDynamics(self.plane, -1, lateralFriction=1.5, physicsClientId=self.client)

        self.urdf_path = os.path.abspath("kit1_spider.urdf")
        meta = self.urdf_path.replace(".urdf", "_meta.json")
        if os.path.exists(meta):
            self.spawn_height = json.load(open(meta))["spawn_height"]
        else:
            self.spawn_height = 0.10
        print(f"📖 Loaded spawn height: {self.spawn_height:.4f} m")

        self.n_legs, self.joints_per_leg = 6, 4
        self.num_joints = self.n_legs * self.joints_per_leg
        self.kp, self.kd, self.max_torque = 1.2, 0.30, 3.0

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_joints,), dtype=np.float32)
        obs_dim = self.num_joints * 2 + 7 + 6
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        #control & scales
        self.control_hz=60
        self.sim_hz=240
        self.inner_step=self.sim_hz//self.control_hz
        self.dt=1.0/self.control_hz

        self.action_scale=np.array([0.25,0.35,0.35,0.35]*self.n_legs,dtype=np.float32)

        self.max_steps=int(self.control_hz*10)
        self._step=0

        # Reward weight (tune later)
        self.w_fwd = 2.0  # forward speed reward
        self.w_alive = 0.1  # small alive bonus
        self.w_tilt = 0.4  # penalize roll/pitch
        self.w_energy = 0.0015  # penalize |tau * qdot|
        self.w_smooth = 0.015 # penalize action changes
        self.w_height = 0.5  # penalize too-low base
        self.z_nominal=float(np.clip(self.spawn_height+0.03,0.05,0.18))
        self.w_contact = 0.06  # encourage 3+ feet on ground
        self.w_lat=0.20
        self.w_yawrate=0.05
        self.w_vz=1.2
        self.swing_scale=np.array([1.0,0.5,0.35,0.35],dtype=np.float32)
        self.stance_scale=np.ones(4,dtype=np.float32)

        #runtime state caches
        self.stance=np.zeros(self.num_joints,dtype=np.float32)
        self.last_action=np.zeros(self.num_joints,dtype=np.float32)
        self.foot_links=[leg*self.joints_per_leg+(self.joints_per_leg-1) for leg in range(self.n_legs)]

        self.progress_window=int(self.control_hz*2.0)
        self.min_progress_threshold=0.03
        self._x_hist=[]
        self._x_last=0.0


        self.enable_watchdog=True

        self._a_prev=np.zeros(self.num_joints,dtype=np.float32)
        self.action_alpha=0.2

        self._tau_prev=np.zeros(self.num_joints,dtype=np.float32)
        self.tau_alpha=0.25
        self.err_clip=0.30
        self.qd_clip=5.0

        self.z_normal=float(np.clip(self.spawn_height+0.05,0.07,0.20))

        print(f"✅ SpiderWalkEnv initialized ({self.num_joints} joints, spawn_height={self.spawn_height:.3f}m)")

    def reset(self, seed=None, options=None):

        def set_physics_params_autodetect(client):
            desired = dict(
                numSolverIterations=180,
                solverResidualThreshold=1e-6,
                enableConeFriction=1,
                erp=0.15,
                contactERP=0.02,
                restitutionVelocityThreshold=0.01,  # will be skipped if unknown
            )
            supported = p.getPhysicsEngineParameters(physicsClientId=client)  # returns a dict
            for k, v in desired.items():
                if k in supported:  # only set if this build knows the key
                    p.setPhysicsEngineParameter(physicsClientId=client, **{k: v})

        super().reset(seed=seed)
        if not p.isConnected(self.client):
            self.client = p.connect(p.GUI if self.render_mode == "human" else p.DIRECT)

        p.resetSimulation(physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client)
        p.changeDynamics(self.plane, -1, lateralFriction=2.5, physicsClientId=self.client)

        p.setGravity(0,0,-9.81)
        p.setTimeStep(1.0/self.sim_hz)
        set_physics_params_autodetect(self.client)

        # safe spawn height
        start_z = max(0.08, min(0.22, float(self.spawn_height)))
        self.robot = p.loadURDF(self.urdf_path, [0, 0, start_z], useFixedBase=False)

        for link_idx in range(-1,p.getNumJoints(self.robot)):
            p.changeDynamics(
                self.robot, link_idx,
                linearDamping=0.06,
                angularDamping=0.12,
                restitution=0.0,
            )

        self._cache_joints()
        self.stance = self._get_grounded_stance().astype(np.float32)
        for i, q in zip(self.joint_indices, self.stance):
            p.resetJointState(self.robot, i, float(q))

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

        for j in self.joint_indices:
            p.setJointMotorControl2(self.robot,j,p.VELOCITY_CONTROL,force=0)

        for _ in range(self.sim_hz):
            for i,qd in zip(self.joint_indices,self.stance):
                p.setJointMotorControl2(
                    self.robot,i,p.POSITION_CONTROL,
                    targetPosition=float(qd),positionGain=0.6,velocityGain=0.2,force=3.0
                )
            p.stepSimulation()

        p.resetBaseVelocity(self.robot, [0,0,0],[0,0,0])

        # ramp gravity
        self.last_action[:]=0.0
        self._x_hist=[]
        self._x_prev=p.getBasePositionAndOrientation(self.robot)[0][0]
        pos,_=p.getBasePositionAndOrientation(self.robot)
        self._x_last=float(pos[0])
        self._step=0

        if self.render_mode == "human":
            self.enable_watchdog=False
        else:
            self.enable_watchdog=True

        return self._get_obs(), {"base_height": p.getBasePositionAndOrientation(self.robot)[0][2]}

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

        lo,hi=self.joint_limits[:,0], self.joint_limits[:,1]
        span=np.clip(hi-lo,0.1,6.28)
        base_frac=np.array([0.40,0.30,0.30,0.30]*self.n_legs,dtype=np.float32)
        self.action_scale=(base_frac*span/2.0).astype(np.float32)

    def _get_grounded_stance(self):
        stance = np.zeros(self.num_joints)
        femur_d, tibia_d, tarsus_d = -0.6, 1.1, -0.7
        for leg in range(self.n_legs):
            b = leg * self.joints_per_leg
            coxa_yaw=0.15 if (leg%2==0) else -0.15
            stance[b+0]=coxa_yaw
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

        lo,hi=self.joint_limits[:,0],self.joint_limits[:,1]
        mid=0.5*(lo+hi)
        span=np.maximum(hi-lo,1e-3)
        q_n=(q-mid)/(0.5*span)
        qd_n=np.tanh(qd/6.0)

        lin=np.clip([vx,vy,vz],-3.0,3.0)/3.0
        ang=np.clip([wx,wy,wz],-6.0,6.0)/6.0

        return np.concatenate([q_n, qd_n, pos, ori, lin, ang], dtype=np.float32)

    def step(self, action):
        # --- minimal: clamp action, plain PD torque, no smoothing, no contact shaping ---
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # read state
        q, qd = self._get_joint_state()

        # target joint positions (stance + scaled action), clipped to limits
        target = self._clip_to_limit(self.stance + self.action_scale * action)

        # plain PD (light clips to avoid numeric spikes)
        err = np.clip(target - q, -0.5, 0.5)
        qd = np.clip(qd, -6.0, 6.0)
        tau = self.kp * err - self.kd * qd
        tau = np.clip(tau, -self.max_torque, self.max_torque)

        # apply torques
        for i, tval in zip(self.joint_indices, tau):
            p.setJointMotorControl2(self.robot, i, p.TORQUE_CONTROL, force=float(tval))

        # integrate
        for _ in range(self.inner_step):
            p.stepSimulation()

        # observe
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        x_now=pos[0]
        base_z = pos[2]
        roll, pitch, _ = self._base_rpy()
        (vx, vy, vz), (wx, wy, wz) = p.getBaseVelocity(self.robot)

        dx=max(0.0,x_now-self._x_last)
        self._x_last=x_now

        contacts=int(self._count_leg_contacts())
        contact_term=contacts/6.0

        self._x_hist.append(x_now)
        if len(self._x_hist)>self.progress_window:
            self._x_hist.pop(0)
        no_progress=(
                len(self._x_hist)==self.progress_window and
                (self._x_hist[-1]-self._x_hist[0]<self.min_progress_threshold)
        )



        # super-simple reward: forward speed – tilt – vertical speed – tiny energy
        fwd = max(vx, 0.0)
        tilt=abs(roll)+abs(pitch)
        mech_power = float(np.mean(np.abs(tau * qd)))  # coarse estimate is fine here

        w_fwd=6.0
        w_dx=25.0
        w_tilt=0.20
        w_vz=0.50
        w_pow=0.0005
        w_cnt=0.04

        x_prog=max(0.0,x_now-self._x_hist[0])
        self._x_prev=x_now

        still=1.0-np.tanh(fwd/0.10)

        reward = (
                2.0 * w_fwd * fwd * self.dt
                + 1.5 * w_dx * dx
                - w_tilt * tilt * self.dt
                - w_vz * abs(vz) * self.dt
                - w_pow * mech_power * self.dt
                + 0.5 * w_cnt * contact_term
        )

        # simple termination
        done = (base_z < 0.02) or (base_z > 0.5) or (abs(roll) > 1.0) or (abs(pitch) > 1.0 or no_progress)
        self._step += 1
        truncated = (self._step >= self.max_steps)

        # debug print
        if self.debug:
            print(f"[min] vx={fwd:.3f} z={base_z:.3f} tilt={(abs(roll) + abs(pitch)):.3f}")

        info = {
            "base_height": float(base_z),
            "fwd_speed": float(np.clip(fwd, -2.0, 2.0)),
            "tilt": float(abs(roll) + abs(pitch)),
            "mech_power": mech_power,
            "contacts":contacts,
        }
        self.last_action = action
        return self._get_obs(), float(reward), bool(done), bool(truncated), info

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)

    #--- Helper ---
    def _clip_to_limit(self,target):
        lo,hi=self.joint_limits[:,0],self.joint_limits[:,1]
        return np.minimum(np.maximum(target,lo),hi)

    def _base_rpy(self):
        _,quat=p.getBasePositionAndOrientation(self.robot)
        return p.getEulerFromQuaternion(quat)

    def _forward_speed_body(self):
        (vx,vy,_),_=p.getBaseVelocity(self.robot)
        _,_,yaw=self._base_rpy()
        hx,hy=np.cos(yaw),np.sin(yaw)
        return vx*hx+vy*hy

    def _count_leg_contacts(self):
        hits=0
        for link in self.foot_links:
            pts=p.getContactPoints(self.robot, self.plane,linkIndexA=link)
            hits+=int(len(pts)>0)
        return hits

    def _yaw_rate(self):
        _,ang=p.getBaseVelocity(self.robot)
        return abs(ang[2])

    def _lateral_speed_body(self):
        (vx, vy, _), _ = p.getBaseVelocity(self.robot)
        _, _, yaw = self._base_rpy()
        # body axes
        hx, hy = np.cos(yaw), np.sin(yaw)  # forward
        rx, ry = -hy, hx  # lateral (right)
        return vx * rx + vy * ry

    def _up_dot(self):
        _,quat=p.getBasePositionAndOrientation(self.robot)
        mat=p.getMatrixFromQuaternion(quat)
        body_z_world=np.array([mat[6],mat[7],mat[8]])
        return float(body_z_world[2])

    def _forward_speed_world_x(self):
        (vx,_,_),_=p.getBaseVelocity(self.robot)
        return vx

    def _leg_contact_mask(self):
        mask=np.zeros(self.n_legs,dtype=np.bool_)
        for leg in range(self.n_legs):
            foot=self.foot_links[leg]
            pts=p.getContactPoints(self.robot, self.plane,linkIndexA=foot)
            mask[leg]=(len(pts)>0)
        return mask

if __name__ == "__main__":
    env = SpiderWalkEnv("human")
    obs, info = env.reset()
    print("Initial:", info)

    # --- Tripod gait with stance duty cycle, per-joint phasing, and stance-only forward push
    T = 3.0
    hz = env.control_hz
    steps = int(T * hz)

    # Legs in tripod A: 0,3,4  (coarse choice works well on many hexapods)
    tripod_A = {0, 3, 4}
    tripod_B = {1, 2, 5}

    # Joint amplitudes (coxa, femur, tibia, tarsus)
    A = np.array([0.05, 0.20, 0.22, 0.14], dtype=np.float32)  # keep coxa small

    # DC posture (light crouch)
    DC = np.array([0.02, 0.06, -0.02, -0.04], dtype=np.float32)

    duty = 0.58  # stance fraction
    phase_lead_tibia = np.deg2rad(12)  # tibia leads for push-off
    phase_lead_tarsus = np.deg2rad(6)  # small lead


    def rect_wave(phase, duty):
        """1 during stance, -1 during swing (centered), continuous using a smoothstep."""
        # phase in [0, 2π)
        x = (phase % (2 * np.pi)) / (2 * np.pi)
        return 2.0 * (x < duty) - 1.0


    rew_sum = 0.0
    for t in range(steps * 4):
        s = 2 * np.pi * (t / steps)  # 0..2π per cycle

        act = np.zeros(env.num_joints, dtype=np.float32)

        for leg in range(env.n_legs):
            base = leg * env.joints_per_leg

            # Tripod phase: B is π shifted relative to A
            leg_phase = s + (np.pi if leg in tripod_B else 0.0)

            # Stance vs swing gate (rect), in {-1, +1}; stance when +1
            gate = rect_wave(leg_phase, duty)
            in_stance = (gate > 0.0)

            # Femur: lift in swing (negative), push in stance (positive)
            femur = A[1] * (0.6 * gate + 0.4 * np.sin(leg_phase))

            # Tibia: lead femur for push-off
            tibia = A[2] * (0.6 * gate + 0.4 * np.sin(leg_phase + phase_lead_tibia))

            # Tarsus: small lead, helps clearance
            tarsus = A[3] * (0.5 * gate + 0.5 * np.sin(leg_phase + phase_lead_tarsus))

            # Coxa: only push forward in stance (no backward during swing)
            coxa = (A[0] * 0.4) if in_stance else 0.0

            act[base + 0] = DC[0] + coxa
            act[base + 1] = DC[1] + femur
            act[base + 2] = DC[2] + tibia
            act[base + 3] = DC[3] + tarsus

        _, r, done, trunc, info = env.step(act)
        rew_sum += r
        if t % 60 == 0:
            print(
                f"t={t / env.control_hz:.1f}s  vx≈{info['fwd_speed']:.3f}  z={info['base_height']:.3f}  c={info['contacts']}")
        if done or trunc:
            break

    print("avg step reward:", rew_sum / max(1, t + 1))

    env.close()
