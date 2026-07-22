"""
DWL MuJoCo 环境 — XBot-L 人形机器人
基于 humanoid-gym/sim2sim.py 的 MuJoCo 推理流程重构
实现: reset/step/观测提取(47维)/特权观测(73维)/奖励计算(12项)/域随机化
"""
import os
import numpy as np
from collections import deque

import mujoco
from scipy.spatial.transform import Rotation as R


def quaternion_to_euler(quat):
    """四元数 [x,y,z,w] -> 欧拉角 [roll, pitch, yaw]"""
    x, y, z, w = quat
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return np.array([roll, pitch, yaw])


class XBotLMuJoCoEnv:
    """单个 MuJoCo XBot-L 环境实例"""
    def __init__(self, cfg, model=None, render=False):
        self.cfg = cfg
        self.render_mode = render

        # 共享模型(内存优化) 或 加载新模型
        if model is not None:
            self.model = model
        else:
            model_path = os.path.join(os.path.dirname(__file__), cfg.model_path)
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.model.opt.timestep = cfg.sim_dt
        self.data = mujoco.MjData(self.model)

        if render:
            import mujoco_viewer
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        else:
            self.viewer = None

        # Joint limits
        self.joint_qpos0 = np.zeros(cfg.num_actions)
        self.dof_lower = np.zeros(cfg.num_actions)
        self.dof_upper = np.zeros(cfg.num_actions)
        for i, name in enumerate(cfg.joint_names):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.joint_qpos0[i] = self.model.jnt_qposadr[jid]  # store qpos address
            self.dof_lower[i] = self.model.jnt_range[jid][0]
            self.dof_upper[i] = self.model.jnt_range[jid][1]

        # Actually store the joint qpos address indices
        self._joint_qpos_ids = []
        for name in cfg.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._joint_qpos_ids.append(self.model.jnt_qposadr[jid])

        # Find foot bodies
        self.left_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'left_ankle_roll_link')
        self.right_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'right_ankle_roll_link')
        self.left_knee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'left_knee_link')
        self.right_knee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'right_knee_link')

        # Sensor IDs
        self.ori_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'orientation')
        self.gyro_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'angular-velocity')
        self.acc_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'linear-acceleration')

        # 域随机化状态
        self.dr_friction = 0.6
        self.dr_motor_strength = 1.0
        self.dr_added_mass = 0.0
        self.dr_action_delay = 0.0

        # 内部状态
        self.episode_step = 0
        self.phase = 0.0
        self.commands = np.zeros(3)  # vx, vy, vyaw
        self.last_action = np.zeros(cfg.num_actions)
        self.last_last_action = np.zeros(cfg.num_actions)
        self.feet_air_time = np.zeros(2)
        self.last_contacts = np.zeros(2, dtype=bool)
        self.foot_swing_time = np.zeros(2)
        self.foot_liftoff_z = np.zeros(2)
        self.last_stance = np.ones(2)

    def reset(self):
        """重置环境到初始状态"""
        mujoco.mj_resetData(self.model, self.data)

        # 随机化初始关节位置 (小幅扰动)
        init_q = self.cfg.default_joint_angles.copy()
        init_q += np.random.uniform(-0.1, 0.1, self.cfg.num_actions)
        init_q = np.clip(init_q, self.dof_lower, self.dof_upper)

        for i, name in enumerate(self.cfg.joint_names):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.data.qpos[self.model.jnt_qposadr[jid]] = init_q[i]

        # 随机化基座初始位姿
        base_pos = self.cfg.base_init_pos.copy()
        base_pos[0] += np.random.uniform(-0.1, 0.1)
        base_pos[1] += np.random.uniform(-0.1, 0.1)
        base_pos[2] += np.random.uniform(-0.05, 0.05)
        self.data.qpos[0:3] = base_pos

        mujoco.mj_forward(self.model, self.data)

        # 采样新的指令
        self._resample_commands()

        # 采样域随机化参数
        self._sample_dr_params()

        # 随机推力
        push_xy = np.random.uniform(-self.cfg.max_push_vel_xy, self.cfg.max_push_vel_xy, 2)
        push_ang = np.random.uniform(-self.cfg.max_push_ang_vel, self.cfg.max_push_ang_vel, 3)
        self.data.qvel[0:2] += push_xy
        self.data.qvel[3:6] += push_ang

        # 重置内部状态
        self.episode_step = 0
        self.last_action = np.zeros(self.cfg.num_actions)
        self.last_last_action = np.zeros(self.cfg.num_actions)
        self.feet_air_time = np.zeros(2)
        self.last_contacts = np.zeros(2, dtype=bool)
        self.foot_swing_time = np.zeros(2)
        self.foot_liftoff_z = np.zeros(2)
        self.last_stance = np.ones(2)

        # 获取初始观测
        obs = self._get_obs()
        privileged = self._get_privileged_obs()
        return obs, privileged

    def step(self, action):
        """执行动作并推进仿真
        action: (12,) 关节目标位置 (scaled, 会被action_scale缩放)
        返回: obs, privileged, reward, terminated, info
        """
        # 裁剪并缩放动作
        action = np.clip(action, -self.cfg.clip_actions, self.cfg.clip_actions)
        target_q = self.cfg.default_joint_angles + action * self.cfg.action_scale

        # 动作延迟 (域随机化)
        delay = np.random.uniform(0, self.dr_action_delay)
        action_delayed = (1 - delay) * action + delay * self.last_action

        # 记录旧状态 (用于奖励计算)
        old_obs = self._get_obs()
        old_vel = self.data.qvel[0:3].copy()
        old_dof_vel = self._get_joint_velocities().copy()

        # 执行 PD 控制 + 物理步进 (decimation步)
        for _ in range(self.cfg.decimation):
            # PD 控制
            current_q = self._get_joint_positions()
            current_dq = self._get_joint_velocities()
            tau = (target_q - current_q) * self.cfg.kps - current_dq * self.cfg.kds
            tau = np.clip(tau, -self.cfg.tau_limit, self.cfg.tau_limit)

            # 电机强度随机化
            tau *= self.dr_motor_strength

            # 施加力矩
            for i, name in enumerate(self.cfg.joint_names):
                act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                self.data.ctrl[act_id] = tau[i]

            mujoco.mj_step(self.model, self.data)
            if self.viewer:
                self.viewer.render()

        # 更新状态
        self.last_last_action = self.last_action.copy()
        self.last_action = action.copy()
        self.episode_step += 1

        # 计算观测
        obs = self._get_obs()
        privileged = self._get_privileged_obs()

        # 计算奖励
        reward, reward_dict, behavior = self._compute_rewards(
            action, old_dof_vel, old_vel, target_q)

        # 判断终止 (记录原因)
        terminated = self._check_termination()
        term_reason = 'timeout'
        if self.data.qpos[2] < self.cfg.termination_height:
            term_reason = 'low_height'
        elif abs(self._get_base_euler()[0]) > self.cfg.termination_orientation or \
             abs(self._get_base_euler()[1]) > self.cfg.termination_orientation:
            term_reason = 'tilted'

        # Info
        info = {
            'reward_dict': reward_dict,
            'episode_step': self.episode_step,
            'behavior': behavior,
            'terminated': terminated,
            'term_reason': term_reason,
        }

        return obs, privileged, reward, terminated, info

    def _get_joint_positions(self):
        """获取12个腿部关节的当前位置"""
        q = np.zeros(self.cfg.num_actions)
        for i, name in enumerate(self.cfg.joint_names):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            q[i] = self.data.qpos[self.model.jnt_qposadr[jid]]
        return q

    def _get_joint_velocities(self):
        """获取12个腿部关节的当前速度"""
        dq = np.zeros(self.cfg.num_actions)
        for i, name in enumerate(self.cfg.joint_names):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            dq[i] = self.data.qvel[self.model.jnt_dofadr[jid]]
        return dq

    def _get_base_velocity(self):
        """获取基座在世界系下的线速度 (来自传感器或直接取)"""
        # 使用 body 的线速度 (世界坐标系)
        # base_link is body 1 (free joint root)
        return self.data.qvel[0:3].copy()

    def _read_sensor(self, sensor_id):
        """安全读取传感器数据"""
        adr = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        return self.data.sensordata[adr:adr + dim].copy()

    def _get_base_angular_velocity(self):
        """获取基座角速度 (机体坐标系, 来自陀螺仪传感器)"""
        return self._read_sensor(self.gyro_sensor_id)

    def _get_base_euler(self):
        """获取基座欧拉角"""
        quat = self._read_sensor(self.ori_sensor_id)
        # MJCF sensor framequat 返回 [w, x, y, z]
        w, x, y, z = quat
        return quaternion_to_euler([x, y, z, w])

    def _get_base_lin_vel_body(self):
        """基座线速度 (机体坐标系)"""
        vel_world = self._get_base_velocity()
        euler = self._get_base_euler()
        r = R.from_euler('xyz', euler)
        vel_body = r.apply(vel_world, inverse=True)
        return vel_body

    def _get_projected_gravity(self):
        """投影重力向量 (机体坐标系)"""
        euler = self._get_base_euler()
        r = R.from_euler('xyz', euler)
        gvec = r.apply(np.array([0., 0., -1.]), inverse=True)
        return gvec

    def _get_foot_positions(self):
        """获取双脚的世界位置"""
        left_pos = self.data.xpos[self.left_foot_id].copy()
        right_pos = self.data.xpos[self.right_foot_id].copy()
        return np.array([left_pos, right_pos])

    def _get_foot_velocities(self):
        """获取双脚的世界速度"""
        left_vel = np.zeros(6)
        right_vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.left_foot_id,
            left_vel, 0)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.right_foot_id,
            right_vel, 0)
        return left_vel[3:6], right_vel[3:6]  # 线速度部分

    def _get_contact_forces(self):
        """获取双脚的接触力 (世界坐标系, Z分量)"""
        left_force = np.zeros(6)
        right_force = np.zeros(6)
        # Contact forces from cfrc_ext (external forces)
        left_force[:] = self.data.cfrc_ext[self.left_foot_id, :]
        right_force[:] = self.data.cfrc_ext[self.right_foot_id, :]
        return np.array([left_force[2], right_force[2]])  # Z分量

    def _resample_commands(self):
        """采样新的速度指令"""
        self.commands[0] = np.random.uniform(*self.cfg.lin_vel_x_range)  # vx
        self.commands[1] = np.random.uniform(*self.cfg.lin_vel_y_range)  # vy
        self.commands[2] = np.random.uniform(*self.cfg.ang_vel_yaw_range)  # yaw rate

    def _sample_dr_params(self):
        """采样域随机化参数 (每episode调用一次)"""
        self.dr_friction = np.random.uniform(*self.cfg.friction_range)
        self.dr_motor_strength = np.random.uniform(*self.cfg.motor_strength_range)
        self.dr_added_mass = np.random.uniform(*self.cfg.added_mass_range)
        self.dr_action_delay = np.random.uniform(*self.cfg.action_delay_range) * 0.001

        # 修改 MuJoCo 模型的地面摩擦 (通过在geom上修改)
        # 注意: 实际摩擦修改需要通过修改 model.geom_friction
        # 这里简化：在奖励/接触中通过缩放接触力来近似
        # 完整实现应调用 mujoco.mj_setConst 或直接修改 model
        try:
            ground_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'ground')
            self.model.geom_friction[ground_id, 0] = self.dr_friction
        except:
            pass

    def _get_gait_phase(self):
        """获取步态相位: 返回 (sin_pos, cos_pos, stance_mask)
        stance_mask: (2,) [左脚, 右脚] 1=支撑相, 0=摆动相
        """
        phase = (self.episode_step * self.cfg.policy_dt) / self.cfg.cycle_time
        sin_pos = np.sin(2 * np.pi * phase)
        stance_left = sin_pos >= 0
        stance_right = sin_pos < 0
        # 双支撑相
        if abs(sin_pos) < 0.1:
            stance_left = stance_right = True
        return sin_pos, np.cos(2 * np.pi * phase), np.array([float(stance_left), float(stance_right)])

    # ============================================================
    # 五次多项式足部轨迹 (论文 Table IV + Section IV-B-2)
    # f(t) = 9.6t⁵ + 12.0t⁴ - 18.8t³ + 5.0t² + 0.1t  (t in [0, T_swing=0.5])
    # ============================================================
    def _quintic_traj(self, t):
        """返回参考高度(归一化, 0~1)和参考速度
        t: 归一化时间 (0=离地, 1=着地, 对应论文t_paper=t*0.5)"""
        t_paper = t * 0.5           # 映射到论文时间域 [0, 0.5]
        h = (9.6*t_paper**5 + 12.0*t_paper**4 - 18.8*t_paper**3 +
             5.0*t_paper**2 + 0.1*t_paper)          # 高度(m), max≈0.1
        v = (48.0*t_paper**4 + 48.0*t_paper**3 - 56.4*t_paper**2 +
             10.0*t_paper + 0.1)                     # 速度(m/s)
        h_norm = h / 0.1            # 归一化到 [0, 1], 0.1=论文h_max
        return h_norm, v / 0.1      # 速度也归一化

    # ============================================================
    # 观测提取
    # ============================================================
    def _get_obs(self):
        """提取47维本体感知观测 (论文Table I, XBot-L适配版)

        结构: clock(2) + commands(3) + joint_pos(12) + joint_vel(12)
               + last_actions(12) + ang_vel(3) + euler(3) = 47
        """
        sin_pos, cos_pos, stance = self._get_gait_phase()
        q = (self._get_joint_positions() - self.cfg.default_joint_angles) * self.cfg.obs_scales.dof_pos
        dq = self._get_joint_velocities() * self.cfg.obs_scales.dof_vel
        ang_vel = self._get_base_angular_velocity() * self.cfg.obs_scales.ang_vel
        euler = self._get_base_euler() * self.cfg.obs_scales.quat

        # 传感器噪声 (论文: 均匀分布, 直接加在已缩放观测上)
        if hasattr(self, 'dr_friction'):
            q += np.random.uniform(-self.cfg.dof_pos_noise, self.cfg.dof_pos_noise, self.cfg.num_actions)
            dq += np.random.uniform(-self.cfg.dof_vel_noise, self.cfg.dof_vel_noise, self.cfg.num_actions)
            ang_vel += np.random.uniform(-self.cfg.ang_vel_noise, self.cfg.ang_vel_noise, 3)
            euler += np.random.uniform(-self.cfg.euler_noise, self.cfg.euler_noise, 3)

        obs = np.concatenate([
            np.array([sin_pos, cos_pos]),                     # clock (2)
            self.commands * np.array([self.cfg.obs_scales.lin_vel,
                                       self.cfg.obs_scales.lin_vel,
                                       self.cfg.obs_scales.ang_vel]),  # commands (3)
            q,                                                # joint_pos (12)
            dq,                                               # joint_vel (12)
            self.last_action,                                 # last_actions (12)
            ang_vel,                                          # ang_vel (3)
            euler,                                            # euler (3)
        ])
        obs = np.clip(obs, -self.cfg.clip_observations, self.cfg.clip_observations)
        return obs.astype(np.float32)

    def _get_privileged_obs(self):
        """提取137维特权观测 (论文: state=obs(47)+priv(137)=184)

        Critic 独享信息: 基座速度、DR参数、足部真值、力矩等
        """
        base_lin_vel_body = self._get_base_lin_vel_body() * self.cfg.obs_scales.lin_vel
        base_lin_vel_world = self._get_base_velocity() * self.cfg.obs_scales.lin_vel
        base_ang_vel = self._get_base_angular_velocity() * self.cfg.obs_scales.ang_vel
        proj_gravity = self._get_projected_gravity()
        euler = self._get_base_euler() * self.cfg.obs_scales.quat
        base_height = self.data.qpos[2]
        sin_pos, cos_pos, stance = self._get_gait_phase()
        foot_positions = self._get_foot_positions()
        foot_vels = self._get_foot_velocities()
        contact_forces = self._get_contact_forces()
        joint_pos = self._get_joint_positions()
        joint_vel = self._get_joint_velocities()
        # 关节力矩近似 (PD 输出)
        torques = (self.cfg.default_joint_angles - joint_pos) * self.cfg.kps - joint_vel * self.cfg.kds
        torques = np.clip(torques, -self.cfg.tau_limit, self.cfg.tau_limit)

        privileged = np.concatenate([
            base_lin_vel_body,                          #  3  (机体线速度)
            base_lin_vel_world,                         #  3  (世界线速度)
            base_ang_vel,                               #  3  (角速度)
            proj_gravity,                               #  3  (投影重力)
            euler,                                      #  3  (欧拉角)
            np.array([base_height]),                    #  1  (基座高度)
            np.array([self.dr_friction]),               #  1  (地面摩擦)
            np.array([self.dr_motor_strength]),         #  1  (电机强度)
            np.array([self.dr_added_mass / 30.0]),      #  1  (附加质量)
            np.array([self.dr_action_delay]),           #  1  (动作延迟)
            stance,                                     #  2  (步态掩码)
            foot_positions.flatten(),                   #  6  (双脚世界位置)
            foot_vels[0],                               #  3  (左脚速度)
            foot_vels[1],                               #  3  (右脚速度)
            contact_forces / 400.0,                     #  2  (接触力归一化)
            (np.abs(contact_forces) > 5.0).astype(np.float32),  #  2  (接触布尔)
            torques / self.cfg.tau_limit,               # 12  (关节力矩归一化)
            np.zeros(12, dtype=np.float64),             # 12  (dof_acc—仿真器不支持, 留空)
            self.commands * np.array([self.cfg.obs_scales.lin_vel,
                                       self.cfg.obs_scales.lin_vel,
                                       self.cfg.obs_scales.ang_vel]),  #  3  (指令)
            np.array([sin_pos, cos_pos]),               #  2  (步态相位)
            self.foot_swing_time / max(self.cfg.swing_time, 0.01),  #  2  (摆动时间)
            (foot_positions[:, 2] - self.foot_liftoff_z).clip(-0.2, 0.2),  #  2  (抬脚高度)
            joint_pos,                                  # 12  (关节位置)
            joint_vel * self.cfg.obs_scales.dof_vel,    # 12  (关节速度)
            self.last_action,                           # 12  (上一动作)
            self.last_last_action,                      # 12  (上上动作)
            np.array([self.episode_step / max(self.cfg.episode_steps, 1)]),  #  1  (episode进度)
        ])
        # 精确填充到 privileged_dim
        if len(privileged) < self.cfg.privileged_dim:
            privileged = np.pad(privileged,
                              (0, self.cfg.privileged_dim - len(privileged)))
        return privileged[:self.cfg.privileged_dim].astype(np.float32)

    # ============================================================
    # 奖励函数 — DWL 论文 Table V, 11项奖励
    # ============================================================
    def _compute_rewards(self, action, old_dof_vel, old_vel, target_q):
        r = self.cfg.rewards
        rewards = {}

        current_q = self._get_joint_positions()
        current_dq = self._get_joint_velocities()
        base_vel = self._get_base_lin_vel_body()
        ang_vel = self._get_base_angular_velocity()
        euler = self._get_base_euler()
        projected_gravity = self._get_projected_gravity()
        contact_forces = self._get_contact_forces()
        sin_pos, cos_pos, stance = self._get_gait_phase()
        foot_positions = self._get_foot_positions()
        foot_vels = self._get_foot_velocities()

        # ===== 1. 线速度跟踪 (Table V: sigma=5.0, w=1.0) =====
        lin_vel_error = np.sum((self.commands[:2] - base_vel[:2]) ** 2)
        rewards['lin_vel'] = np.exp(-lin_vel_error * r.lin_vel_sigma)

        # ===== 2. 角速度跟踪 (Table V: sigma=7.0, w=1.0) =====
        ang_vel_error = (self.commands[2] - ang_vel[2]) ** 2
        rewards['ang_vel'] = np.exp(-ang_vel_error * r.ang_vel_sigma)

        # ===== 3. 姿态跟踪 (Table V: sigma=5.0, w=1.0) =====
        rewards['orientation'] = np.exp(-np.sum(euler[:2] ** 2) * r.orientation_sigma)

        # ===== 4. 身高跟踪 (Table V: sigma=10.0, w=0.5) =====
        base_height = self.data.qpos[2]
        h_error = base_height - self.cfg.base_height_target
        rewards['height'] = np.exp(-(h_error ** 2) * r.height_sigma)

        # ===== 5. 周期性接触力 (Table V: w=1.0) =====
        contact_norm = np.clip(np.abs(contact_forces) / self.cfg.max_contact_force, 0.0, 1.0)
        rewards['periodic_contact'] = float(np.sum(stance * contact_norm))

        # ===== 6. 周期性足部速度 (Table V: w=1.0) =====
        swing_foot_speed = np.zeros(2)
        for fi in range(2):
            swing_foot_speed[fi] = np.linalg.norm(foot_vels[fi][:2])
        rewards['periodic_vel'] = float(np.sum((1.0 - stance) * np.clip(swing_foot_speed, 0.0, 2.0) / 2.0))

        # ===== 足部摆动时间跟踪 =====
        for foot_idx in range(2):
            is_stance = stance[foot_idx] > 0.5
            if self.last_stance[foot_idx] > 0.5 and not is_stance:
                self.foot_swing_time[foot_idx] = 0.0
                self.foot_liftoff_z[foot_idx] = foot_positions[foot_idx][2]
            elif not is_stance:
                self.foot_swing_time[foot_idx] += self.cfg.policy_dt
            self.last_stance[foot_idx] = float(is_stance)

        # ===== 7. 足部高度跟踪 (Table V: sigma=5.0, w=1.0) =====
        # ===== 8. 足部速度跟踪 (Table V: sigma=3.0, w=0.5) =====
        foot_h_rw = 0.0
        foot_v_rw = 0.0
        for foot_idx in range(2):
            is_stance = stance[foot_idx] > 0.5
            if not is_stance and self.cfg.swing_time > 0:
                t_norm = np.clip(self.foot_swing_time[foot_idx] / self.cfg.swing_time, 0.0, 1.0)
                h_ref_norm, v_ref_norm = self._quintic_traj(t_norm)

                actual_dz = foot_positions[foot_idx][2] - self.foot_liftoff_z[foot_idx]
                ref_dz = h_ref_norm * self.cfg.h_max
                h_err = actual_dz - ref_dz
                foot_h_rw += np.exp(-h_err ** 2 * r.foot_height_sigma)

                actual_vz = foot_vels[foot_idx][2]
                ref_vz = v_ref_norm * self.cfg.h_max / self.cfg.swing_time
                v_err = actual_vz - ref_vz
                foot_v_rw += np.exp(-v_err ** 2 * r.foot_vel_sigma)

        rewards['foot_height'] = foot_h_rw / 2.0
        rewards['foot_vel'] = foot_v_rw / 2.0

        # ===== 9. 默认关节姿态 (Table V: sigma=2.0, w=0.2) =====
        joint_diff = current_q - self.cfg.default_joint_angles
        rewards['default_joint'] = np.exp(-np.sum(joint_diff ** 2) * r.default_joint_sigma)

        # ===== 10. 能耗惩罚 (Table V: w=-0.0001) =====
        torques = (target_q - current_q) * self.cfg.kps - current_dq * self.cfg.kds
        torques = np.clip(torques, -self.cfg.tau_limit, self.cfg.tau_limit)
        rewards['energy'] = np.sum(np.abs(torques * current_dq))

        # ===== 11. 动作平滑 (Table V: w=-0.01) =====
        action_diff2 = action - 2.0 * self.last_action + self.last_last_action
        rewards['action_smooth'] = np.sum(action_diff2 ** 2)

        # 用于日志
        foot_slip = np.sum(np.abs(np.array([
            np.linalg.norm(self.data.cvel[self.left_foot_id][3:6]),
            np.linalg.norm(self.data.cvel[self.right_foot_id][3:6])
        ]))) * 0.1

        # ===== 加权求和 (严格按论文 Table V) =====
        total = (
            r.lin_vel * rewards['lin_vel'] +
            r.ang_vel * rewards['ang_vel'] +
            r.orientation * rewards['orientation'] +
            r.height * rewards['height'] +
            r.periodic_contact * rewards['periodic_contact'] +
            r.periodic_vel * rewards['periodic_vel'] +
            r.foot_height * rewards['foot_height'] +
            r.foot_vel * rewards['foot_vel'] +
            r.default_joint * rewards['default_joint'] +
            r.energy * rewards['energy'] +
            r.action_smooth * rewards['action_smooth']
        )

        left_knee_q = current_q[3]
        right_knee_q = current_q[9]

        behavior = {
            'cmd_vx': float(self.commands[0]),
            'cmd_vy': float(self.commands[1]),
            'cmd_vyaw': float(self.commands[2]),
            'actual_vx': float(base_vel[0]),
            'actual_vy': float(base_vel[1]),
            'actual_vyaw': float(ang_vel[2]),
            'base_vel_x_world': float(self.data.qvel[0]),
            'base_vel_y_world': float(self.data.qvel[1]),
            'roll': float(euler[0]),
            'pitch': float(euler[1]),
            'yaw': float(euler[2]),
            'ang_vel_x': float(ang_vel[0]),
            'ang_vel_y': float(ang_vel[1]),
            'ang_vel_z': float(ang_vel[2]),
            'gravity_z': float(projected_gravity[2]),
            'base_height': float(base_height),
            'height_error': float(h_error),
            'base_x': float(self.data.qpos[0]),
            'base_y': float(self.data.qpos[1]),
            'left_foot_z': float(foot_positions[0][2]),
            'right_foot_z': float(foot_positions[1][2]),
            'left_foot_x': float(foot_positions[0][0]),
            'right_foot_x': float(foot_positions[1][0]),
            'contact_left': float(contact_forces[0]),
            'contact_right': float(contact_forces[1]),
            'foot_slip_raw': float(foot_slip),
            'gait_sin': float(sin_pos),
            'stance_left': float(stance[0]),
            'stance_right': float(stance[1]),
            'left_knee_q': float(left_knee_q),
            'right_knee_q': float(right_knee_q),
            'dof_vel_mean': float(np.mean(np.abs(current_dq))),
            'torque_mean': float(np.mean(np.abs(torques))),
            'torque_max': float(np.max(np.abs(torques))),
            'action_mean': float(np.mean(np.abs(action))),
            'action_max': float(np.max(np.abs(action))),
            # DWL 论文 11 项奖励
            'r_lin_vel': float(rewards['lin_vel']),
            'r_ang_vel': float(rewards['ang_vel']),
            'r_orientation': float(rewards['orientation']),
            'r_height': float(rewards['height']),
            'r_periodic_contact': float(rewards['periodic_contact']),
            'r_periodic_vel': float(rewards['periodic_vel']),
            'r_foot_height': float(rewards['foot_height']),
            'r_foot_vel': float(rewards['foot_vel']),
            'r_default_joint': float(rewards['default_joint']),
            'r_energy': float(rewards['energy']),
            'r_action_smooth': float(rewards['action_smooth']),
        }

        return float(total), rewards, behavior

    def _check_termination(self):
        """检查是否终止: 基座过低 或 姿态过大"""
        base_height = self.data.qpos[2]
        euler = self._get_base_euler()

        if base_height < self.cfg.termination_height:
            return True
        if abs(euler[0]) > self.cfg.termination_orientation or \
           abs(euler[1]) > self.cfg.termination_orientation:
            return True
        if self.episode_step >= self.cfg.episode_steps:
            return True
        return False


class VecEnv:
    """向量化环境包装器: 多线程并行管理多个独立 MuJoCo 环境 (共享模型以节省内存)

    MuJoCo的mj_step是C函数(释放GIL), numpy运算也释放GIL, 因此ThreadPoolExecutor可实现真实并行。
    """
    def __init__(self, cfg, num_envs=None, num_workers=None):
        if num_envs is None:
            num_envs = cfg.num_envs
        self.cfg = cfg
        self.num_envs = num_envs

        # 线程数: 默认取CPU核数和8的较小值
        if num_workers is None:
            num_workers = min(os.cpu_count() or 8, 8)
        self.num_workers = max(1, num_workers)

        # 只加载一次模型, 所有环境共享 (节省 ~40MB/env * num_envs)
        model_path = os.path.join(os.path.dirname(__file__), cfg.model_path)
        shared_model = mujoco.MjModel.from_xml_path(model_path)
        shared_model.opt.timestep = cfg.sim_dt
        print(f"[Env] Shared model loaded, creating {num_envs} data instances ({self.num_workers} workers)...")
        self.envs = [XBotLMuJoCoEnv(cfg, model=shared_model, render=False) for _ in range(num_envs)]

        # 观测历史缓冲 (用于GRU序列输入)
        self.seq_len = 10  # DWL论文 seq_len=10
        self.obs_history = [deque(maxlen=self.seq_len) for _ in range(num_envs)]

        # 预分配线程安全的result buffers
        self._obs_buf = np.zeros((num_envs, cfg.obs_dim), dtype=np.float32)
        self._priv_buf = np.zeros((num_envs, cfg.privileged_dim), dtype=np.float32)
        self._reward_buf = np.zeros(num_envs, dtype=np.float32)
        self._done_buf = np.zeros(num_envs, dtype=bool)

    def _step_range(self, start, end, actions):
        """Thread worker: step envs in [start, end), 写入共享buffer的不同索引(线程安全)"""
        for i in range(start, end):
            obs, priv, r, done, info = self.envs[i].step(actions[i])
            self._obs_buf[i] = obs
            self._priv_buf[i] = priv
            self._reward_buf[i] = r
            self._done_buf[i] = done
            self._info_list[i] = info

    def _step_range_with_behavior(self, start, end, actions):
        """Thread worker with behavior data collection"""
        for i in range(start, end):
            obs, priv, r, done, info = self.envs[i].step(actions[i])
            self._obs_buf[i] = obs
            self._priv_buf[i] = priv
            self._reward_buf[i] = r
            self._done_buf[i] = done
            self._info_list[i] = info
            self._behavior_list[i] = info.get('behavior', {})

    def _parallel_step(self, actions, collect_behavior=False):
        """多线程并行步进所有环境"""
        from concurrent.futures import ThreadPoolExecutor

        step_fn = self._step_range_with_behavior if collect_behavior else self._step_range

        if self.num_workers <= 1:
            step_fn(0, self.num_envs, actions)
        else:
            chunk_size = max(1, (self.num_envs + self.num_workers - 1) // self.num_workers)
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                for i in range(0, self.num_envs, chunk_size):
                    end = min(i + chunk_size, self.num_envs)
                    futures.append(executor.submit(step_fn, i, end, actions))
                for f in futures:
                    f.result()

        # 串行处理重置和历史更新 (快速, 无需并行)
        for i in range(self.num_envs):
            obs_i = self._obs_buf[i]
            self.obs_history[i].append(obs_i)

            if self._done_buf[i]:
                obs_i, priv_i = self.envs[i].reset()
                self._obs_buf[i] = obs_i
                self._priv_buf[i] = priv_i
                for _ in range(self.seq_len):
                    self.obs_history[i].append(obs_i)

    def reset(self):
        """重置所有环境"""
        all_obs = np.zeros((self.num_envs, self.cfg.obs_dim), dtype=np.float32)
        all_priv = np.zeros((self.num_envs, self.cfg.privileged_dim), dtype=np.float32)

        for i, env in enumerate(self.envs):
            obs, priv = env.reset()
            all_obs[i] = obs
            all_priv[i] = priv
            for _ in range(self.seq_len):
                self.obs_history[i].append(obs)

        return all_obs, all_priv

    def step(self, actions):
        """对所有环境执行一步 (多线程并行)
        actions: (num_envs, 12)
        返回: obs_batch, priv_batch, rewards, dones, infos
        """
        self._info_list = [None] * self.num_envs
        self._parallel_step(actions, collect_behavior=False)

        return (self._obs_buf.copy(), self._priv_buf.copy(),
                self._reward_buf.copy(), self._done_buf.copy(),
                list(self._info_list))

    def step_with_behavior(self, actions):
        """同 step(), 但额外收集所有env的behavior数据用于日志"""
        self._info_list = [None] * self.num_envs
        self._behavior_list = [{}] * self.num_envs
        self._parallel_step(actions, collect_behavior=True)

        return (self._obs_buf.copy(), self._priv_buf.copy(),
                self._reward_buf.copy(), self._done_buf.copy(),
                list(self._info_list), list(self._behavior_list))

    def get_obs_sequences(self):
        """获取所有环境的观测序列 (B, seq_len, obs_dim)"""
        seqs = np.zeros((self.num_envs, self.seq_len, self.cfg.obs_dim), dtype=np.float32)
        for i in range(self.num_envs):
            for t, obs in enumerate(self.obs_history[i]):
                seqs[i, t] = obs
        return seqs

    def close(self):
        for env in self.envs:
            if env.viewer:
                env.viewer.close()
