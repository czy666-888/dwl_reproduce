"""
DWL (Denoising World Model Learning) 超参数配置
严格对齐论文 Table II, IV, V, VI, VIII
机器人: XBot-L (原论文 XBot-S 未开源, 参数已适配)
"""
import os
import numpy as np

class DWLConfig:
    # ============ 机器人模型 ============
    model_path = os.path.join(os.path.dirname(__file__), "models", "XBot-L.xml")
    # 12个腿部关节名 (MJCF motor顺序)
    joint_names = [
        'left_leg_roll_joint', 'left_leg_yaw_joint', 'left_leg_pitch_joint',
        'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
        'right_leg_roll_joint', 'right_leg_yaw_joint', 'right_leg_pitch_joint',
        'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    ]
    num_actions = 12

    # ============ DWL 网络维度 (Table VI) ============
    obs_dim = 47                # 本体感知观测 (Table I)
    state_dim = 184             # 完整状态: obs(47) + privileged(137)
    latent_dim = 24             # 隐状态瓶颈
    gru_hidden = 256            # GRU隐藏维度
    privileged_dim = 137        # 特权观测维度 (state_dim - obs_dim)

    # ============ PPO 参数 (Table VIII) ============
    gamma = 0.995
    gae_lambda = 0.95
    clip_range = 0.2            # epsilon = 0.2
    entropy_coef = 0.005        # 论文原值
    learning_rate = 1e-5        # 论文原值
    max_grad_norm = 1.0

    # ============ DWL 损失权重 (Equation 5) ============
    lambda_pi = 5.0
    lambda_v = 5.0
    lambda_l1 = 0.002           # L1 稀疏正则

    # ============ 训练规模 ============
    num_envs = 256              # 原论文 12288, 适配本地 GPU (T600 4GB)
    num_steps_per_env = 96      # 原论文24, 增大到96补偿环境数不足 (256 vs 12288)
    num_epochs = 5              # 原论文2, 增加epoch提高样本利用率, 配合KL早停
    num_mini_batches = 8        # 原论文4, 更多mini-batch提高梯度步数
    kl_early_stop = True        # KL早停: 当KL(old||new)超过阈值时停止epoch
    kl_target = 0.015           # KL目标阈值

    # ============ 仿真参数 ============
    sim_dt = 0.001              # MuJoCo 物理步长 (1000Hz)
    decimation = 10             # 策略控制间隔 = 10 * 0.001 = 0.01s (100Hz)
    policy_dt = sim_dt * decimation  # 0.01s

    # Episode
    episode_length_s = 24.0
    episode_steps = int(episode_length_s / policy_dt)  # 2400步

    # ============ 步态参数 (Table IV, XBot-L适配) ============
    cycle_time = 0.64           # 步态周期 0.64s
    swing_time = 0.5            # 摆动相 T_swing = 0.5s (论文值)
    h_max = 0.06                # 最大抬脚高度 (XBot-L=0.06, 论文XBot-S=0.1)

    # ============ 五次多项式足部轨迹系数 (Section IV-B-2) ============
    # f(t) = 9.6t^5 + 12.0t^4 - 18.8t^3 + 5.0t^2 + 0.1t
    traj_coeffs = [9.6, 12.0, -18.8, 5.0, 0.1, 0.0]

    # ============ PD 控制参数 ============
    # stiffness/damping: leg_roll=200, leg_yaw=200, leg_pitch=350, knee=350, ankle=15
    kps = np.array([200., 200., 350., 350., 15., 15.,
                    200., 200., 350., 350., 15., 15.], dtype=np.float64)
    kds = np.array([10., 10., 10., 10., 10., 10.,
                    10., 10., 10., 10., 10., 10.], dtype=np.float64)
    tau_limit = 200.0           # 力矩限幅 (Nm)
    action_scale = 0.25         # action -> 目标关节角度的缩放因子

    # ============ 默认关节姿态 ============
    default_joint_angles = np.zeros(12, dtype=np.float64)

    # ============ 初始化 ============
    base_init_pos = [0.0, 0.0, 0.95]  # 基座初始高度

    # ============ 指令范围 ============
    lin_vel_x_range = [-0.3, 0.6]   # m/s
    lin_vel_y_range = [-0.3, 0.3]
    ang_vel_yaw_range = [-0.3, 0.3]

    # ============ 域随机化 (Table II) ============
    friction_range = [0.2, 2.0]
    motor_strength_range = [0.9, 1.1]
    added_mass_range = [-5.0, 20.0]   # kg (论文: -5~20)
    action_delay_range = [0, 0.5]     # 实际延迟范围 [0, 5]ms
    action_noise = 0.02
    # 传感器噪声 (论文: 均匀分布, 加在已缩放的观测上)
    dof_pos_noise = 0.3              # uniform(-0.3, 0.3) rad
    dof_vel_noise = 1.0              # uniform(-1.0, 1.0) rad/s
    ang_vel_noise = 0.1              # uniform(-0.1, 0.1)
    euler_noise = 0.1                # uniform(-0.1, 0.1)

    # 随机推力
    max_push_vel_xy = 0.2
    max_push_ang_vel = 0.4
    push_interval_s = 4.0

    # ============ 终止条件 ============
    termination_height = 0.65        # 基座低于此高度终止 (0.35→0.55→0.65, 逐步收紧防止深蹲策略)
    termination_orientation = 1.0    # roll/pitch 超过此值终止 (rad)

    # ============ 奖励权重 (Table V, XBot-L + MuJoCo 适配) ============
    class rewards:
        # 跟踪奖励 weights
        lin_vel = 1.0                 # 线速度跟踪
        ang_vel = 1.0                 # 角速度跟踪
        orientation = 2.0             # 姿态跟踪 (0.5→1.0→2.0, 加强: 惩罚蹲姿导致的倾斜)
        height = 6.0                  # 身高跟踪 (0.5→2.0→6.0, 核心修改: 强制机器人站立)
        periodic_contact = 1.0        # 周期性接触力
        periodic_vel = 1.0            # 周期性足部速度
        foot_height = 1.0             # 足部高度跟踪
        foot_vel = 0.5                # 足部速度跟踪
        default_joint = 0.2           # 默认关节姿态
        energy = -0.0005              # 能耗惩罚 (0→-0.0001→-0.0005, 蹲姿力矩大, 加重惩罚)
        action_smooth = -0.01         # 动作平滑二阶惩罚

        # 跟踪 sigma (Table V, XBot-L适配)
        lin_vel_sigma = 5.0
        ang_vel_sigma = 7.0
        orientation_sigma = 8.0       # (5.0→8.0, 对小幅倾斜更灵敏)
        height_sigma = 30.0           # (10.0→30.0, 身高误差惩罚急剧增加)
        foot_height_sigma = 5.0
        foot_vel_sigma = 3.0
        default_joint_sigma = 2.0

    base_height_target = 0.89       # XBot-L 目标身高 (论文XBot-S=0.7)
    max_contact_force = 700.0       # 最大接触力

    # ============ 归一化 ============
    class obs_scales:
        lin_vel = 2.0
        ang_vel = 1.0
        dof_pos = 1.0
        dof_vel = 0.05
        quat = 1.0
        height_measurements = 5.0

    clip_observations = 18.0
    clip_actions = 18.0

    # ============ 日志 / 保存 ============
    save_interval = 100
    log_dir = "logs"
    checkpoint_dir = "checkpoints"
    device = "cuda"  # will fallback to cpu if no cuda
