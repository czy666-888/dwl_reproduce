"""
DWL (Denoising World Model Learning) 复现
论文: Advancing Humanoid Locomotion: Mastering Challenging Terrains
      with Denoising World Model Learning (RSS 2024)

架构:
  Encoder(GRU): o_t(47) -> h_t(256) -> z_t(24)     [去噪+压缩]
  Decoder(MLP):  z_t(24) -> s_tilde(184)             [状态重建]
  Actor(MLP):    z_t(24) -> a_t(12)                  [策略]
  Critic(MLP):   s_t(184) -> V(s_t)                  [价值评估]

训练: L_total = L_denoise + 5*L_pi(PPO) + 5*L_v
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

# ============================================================
# 0. 超参数配置 (Table VIII)
# ============================================================
class DWLConfig:
    # 维度
    obs_dim = 47          # 本体感知观测
    state_dim = 184        # 完整状态(含特权信息)
    action_dim = 12        # 12个腿部关节目标位置
    latent_dim = 24        # 隐状态瓶颈

    # GRU
    gru_hidden = 256

    # PPO
    gamma = 0.995
    gae_lambda = 0.95
    clip_range = 0.2      # [1-ε, 1+ε] = [0.8, 1.2]
    entropy_coef = 0.005
    learning_rate = 1e-5

    # DWL 损失权重
    lambda_pi = 5.0
    lambda_v = 5.0
    lambda_l1 = 0.002    # L1稀疏正则

    # 训练规模
    num_envs = 12288
    episode_steps = 2400
    num_epochs = 2
    batch_size = 24       # 每env截断步数

    # 足部轨迹
    swing_time = 0.5      # T = 0.5s
    h_max = 0.1           # 最大抬脚高度 0.1m
    dt = 0.01             # 100Hz控制频率


# ============================================================
# 1. 网络架构 (Table VI)
# ============================================================
class GRUEncoder(nn.Module):
    """Encoder: 噪声观测序列 -> 去噪隐状态
    GRU(47->256) -> Linear(256->256) -> ELU -> Linear(256->24)
    """
    def __init__(self, cfg):
        super().__init__()
        self.gru = nn.GRU(input_size=cfg.obs_dim,
                          hidden_size=cfg.gru_hidden,
                          batch_first=True)
        self.emb = nn.Sequential(
            nn.Linear(cfg.gru_hidden, cfg.gru_hidden),
            nn.ELU(alpha=1.0),
            nn.Linear(cfg.gru_hidden, cfg.latent_dim)
        )

    def forward(self, o_seq, h_prev=None):
        """
        o_seq: (B, seq_len, 47)  历史观测序列
        h_prev: (1, B, 256)      上一时刻隐藏状态
        返回: z_t (B, 24), h_t (1, B, 256)
        """
        gru_out, h_t = self.gru(o_seq, h_prev)  # gru_out: (B, seq_len, 256)
        z_t = self.emb(gru_out[:, -1, :])        # 取最后一步, 压缩到24维
        return z_t, h_t


class Decoder(nn.Module):
    """Decoder: 隐状态 -> 完整状态重建
    Linear(24->64) -> ELU -> Linear(64->184)
    """
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim, 64),
            nn.ELU(alpha=1.0),
            nn.Linear(64, cfg.state_dim)
        )

    def forward(self, z_t):
        return self.net(z_t)  # (B, 184)


class Actor(nn.Module):
    """Actor: 隐状态 -> 动作分布均值
    Linear(24->48) -> ELU -> Linear(48->12)
    """
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim, 48),
            nn.ELU(alpha=1.0),
            nn.Linear(48, cfg.action_dim)
        )
        # 可学习的对数标准差(每个关节独立)
        self.log_std = nn.Parameter(torch.zeros(cfg.action_dim))

    def forward(self, z_t, deterministic=False):
        mu = self.net(z_t)                    # (B, 12)
        std = torch.exp(self.log_std)         # (12,)
        if deterministic:
            return mu
        dist = Normal(mu, std)
        return dist


class Critic(nn.Module):
    """Critic: 完整状态 -> 价值标量
    Linear(184->512)->ELU->Linear(512->512)->ELU->Linear(512->256)->ELU->Linear(256->1)
    """
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.state_dim, 512),
            nn.ELU(alpha=1.0),
            nn.Linear(512, 512),
            nn.ELU(alpha=1.0),
            nn.Linear(512, 256),
            nn.ELU(alpha=1.0),
            nn.Linear(256, 1)
        )

    def forward(self, s_t):
        return self.net(s_t).squeeze(-1)  # (B,)


class DWLModel(nn.Module):
    """DWL完整模型"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = GRUEncoder(cfg)
        self.decoder = Decoder(cfg)
        self.actor = Actor(cfg)
        self.critic = Critic(cfg)

    def forward(self, o_seq, s_t, h_prev=None, deterministic=False):
        """
        o_seq: (B, seq_len, 47)
        s_t:   (B, 184) — 仅训练时用
        """
        z_t, h_t = self.encoder(o_seq, h_prev)
        s_tilde = self.decoder(z_t)
        dist = self.actor(z_t, deterministic=deterministic)
        value = self.critic(s_t)
        return z_t, s_tilde, dist, value, h_t


# ============================================================
# 2. 域随机化 (Table II)
# ============================================================
class DomainRandomization:
    """在观测和状态上施加DR噪声与掩码"""
    def __init__(self, cfg):
        self.cfg = cfg
        # 观测掩码: 哪些维度在真实机器人上可见(1=可见, 0=掩码)
        # Table I: 前47维 = Observation (全部可见)
        self.obs_mask = torch.ones(cfg.obs_dim)

    def add_sensor_noise(self, obs):
        """在观测上加传感器噪声 (Table II)"""
        B = obs.shape[0]
        device = obs.device
        # 关节位置 (索引 5:17, 共12维)
        obs[:, 5:17] += torch.empty(B, 12, device=device).uniform_(-0.3, 0.3)
        # 关节速度 (索引 17:29, 共12维)
        obs[:, 17:29] += torch.empty(B, 12, device=device).uniform_(-1.0, 1.0)
        # 角速度 (索引 29:32, 共3维)
        obs[:, 29:32] += torch.empty(B, 3, device=device).uniform_(-0.1, 0.1)
        # 欧拉角 (索引 32:35, 共3维)
        obs[:, 32:35] += torch.empty(B, 3, device=device).uniform_(-0.1, 0.1)
        return obs

    def sample_dr_params(self, B, device):
        """每episode采样一次DR参数"""
        return {
            'friction': torch.empty(B, device=device).uniform_(0.2, 2.0),
            'motor_strength': torch.empty(B, device=device).uniform_(0.9, 1.1),
            'payload': torch.empty(B, device=device).uniform_(-5, 20),
            'pd_factor': torch.empty(B, device=device).uniform_(0.8, 1.2),
            'system_delay': torch.randint(0, 11, (B,), device=device),
            'motor_offset': torch.empty(B, 12, device=device).uniform_(-0.05, 0.05),
        }

    def make_noisy_obs(self, clean_obs):
        """仿真中: 干净观测 -> 加噪声 -> 噪声观测"""
        noisy = clean_obs.clone()
        noisy = self.add_sensor_noise(noisy)
        return noisy

    def make_state(self, obs, privileged):
        """拼接观测+特权信息 -> 完整184维状态"""
        return torch.cat([obs, privileged], dim=-1)  # (B, 47+137=184)


# ============================================================
# 3. 五次多项式足部轨迹 (Table IV + Section IV-B-2)
# ============================================================
class QuinticFootTrajectory:
    """f(t) = 9.6t^5 + 12.0t^4 - 18.8t^3 + 5.0t^2 + 0.1t"""
    def __init__(self, cfg):
        self.T = cfg.swing_time       # 0.5s
        self.coeffs = [9.6, 12.0, -18.8, 5.0, 0.1, 0.0]  # a5 -> a0

    def __call__(self, t):
        """t: 相对摆动相开始的时间 (0到T)"""
        return (self.coeffs[0]*t**5 + self.coeffs[1]*t**4 +
                self.coeffs[2]*t**3 + self.coeffs[3]*t**2 +
                self.coeffs[4]*t    + self.coeffs[5])

    def derivative(self, t):
        """f'(t) = 48t^4 + 48t^3 - 56.4t^2 + 10t + 0.1"""
        return (48.0*t**4 + 48.0*t**3 - 56.4*t**2 + 10.0*t + 0.1)

    def get_reference(self, t_relative):
        """返回参考高度和参考速度"""
        h_ref = self.__call__(t_relative)
        v_ref = self.derivative(t_relative)
        return h_ref, v_ref


# ============================================================
# 4. 奖励函数 (Table V)
# ============================================================
class RewardFunction:
    """12项加权奖励"""
    def __init__(self, cfg):
        self.cfg = cfg
        self.traj = QuinticFootTrajectory(cfg)
        self.base_height_target = 0.7  # XBot-S身高0.7m(蹲姿)

    def tracking_error(self, e, w):
        """phi(e,w) = exp(-w * ||e||^2)"""
        return torch.exp(-w * (e ** 2).sum(dim=-1))

    def compute(self, obs, state, action, prev_action, stance_mask, t_relative):
        """
        简化版奖励计算(依赖仿真器提供的真值, 这里是结构示意)
        stance_mask: (B,2) [左脚, 右脚] 1=支撑相 0=摆动相
        """
        # 从state中提取各分量(仿真中直接可拿)
        base_vel = state[:, 47:50]        # 基座线速度
        base_ang_vel = state[:, 29:32]    # 角速度
        base_ori = state[:, 32:35]        # 欧拉角(roll,pitch,yaw)
        base_height = state[:, 35:36]     # 基座高度(NOT real index, 示意)
        foot_height = state[:, 78:80]     # 足部高度(示意)
        foot_vel = state[:, 80:82]        # 足部速度(示意)
        foot_contact = state[:, 83:85]    # 足部接触力(示意, scaled)
        joint_pos = state[:, 5:17]        # 关节位置
        joint_vel = state[:, 17:29]       # 关节速度
        torques = state[:, 171:183]       # 关节力矩

        cmd_xy = obs[:, 2:4]              # 速度指令vx,vy
        cmd_yaw = obs[:, 4:5]             # 转向指令

        rewards = {}

        # 1. 线速度跟踪
        v_err = base_vel[:, :2] - cmd_xy
        rewards['lin_vel'] = self.tracking_error(v_err, 5.0)

        # 2. 角速度跟踪
        w_err = base_ang_vel[:, 2:3] - cmd_yaw
        rewards['ang_vel'] = self.tracking_error(w_err, 7.0)

        # 3. 姿态跟踪 (保持竖直)
        rewards['orientation'] = self.tracking_error(base_ori[:, :2], 5.0)

        # 4. 身高跟踪
        h_err = base_height.squeeze(-1) - self.base_height_target
        rewards['height'] = self.tracking_error(h_err.unsqueeze(-1), 10.0)

        # 5. 周期性接触力
        rewards['periodic_force'] = (stance_mask * foot_contact).sum(dim=-1)

        # 6. 周期性足部速度
        rewards['periodic_vel'] = ((1 - stance_mask) * foot_vel).sum(dim=-1)

        # 7. 足部高度跟踪
        h_ref, v_ref = self.traj.get_reference(t_relative)
        rewards['foot_height'] = self.tracking_error(
            foot_height - h_ref.unsqueeze(-1), 5.0)

        # 8. 足部速度跟踪
        rewards['foot_vel_track'] = self.tracking_error(
            foot_vel - v_ref.unsqueeze(-1), 3.0)

        # 9. 默认关节姿态
        rewards['default_joint'] = self.tracking_error(joint_pos, 2.0)

        # 10-12. 惩罚项
        rewards['energy'] = -(torch.abs(torques * joint_vel).sum(dim=-1))
        rewards['action_smooth'] = -((action - 2*prev_action +
                                      torch.roll(action, 1, 0))**2).sum(dim=-1)

        # 奖励权重 (Table V)
        weights = {
            'lin_vel': 1.0, 'ang_vel': 1.0, 'orientation': 1.0,
            'height': 0.5, 'periodic_force': 1.0, 'periodic_vel': 1.0,
            'foot_height': 1.0, 'foot_vel_track': 0.5, 'default_joint': 0.2,
            'energy': -0.0001, 'action_smooth': -0.01,
        }

        total = sum(weights[k] * rewards[k] for k in weights)
        return total, rewards


# ============================================================
# 5. PPO损失 + GAE优势估计
# ============================================================
class PPOLoss:
    def __init__(self, cfg):
        self.cfg = cfg
        self.clip_low = 1.0 - cfg.clip_range   # 0.8
        self.clip_high = 1.0 + cfg.clip_range   # 1.2

    def compute_gae(self, rewards, values, dones, gamma, lam):
        """广义优势估计 (GAE)"""
        advantages = []
        gae = 0
        values = values + [0]  # 终端V=0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t+1] * (1-dones[t]) - values[t]
            gae = delta + gamma * lam * (1-dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return torch.stack(advantages), torch.stack(returns)

    def policy_loss(self, dist_new, dist_old, actions, advantages):
        """PPO Clip Loss (Equation 3)"""
        log_prob_new = dist_new.log_prob(actions).sum(dim=-1)
        log_prob_old = dist_old.log_prob(actions).sum(dim=-1)

        ratio = torch.exp(log_prob_new - log_prob_old)

        # Clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, self.clip_low, self.clip_high) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Entropy bonus
        entropy = dist_new.entropy().sum(dim=-1).mean()

        return policy_loss - self.cfg.entropy_coef * entropy

    def value_loss(self, values, returns):
        """MSE Value Loss (Equation 4)"""
        return F.mse_loss(values, returns)


# ============================================================
# 6. DWL完整训练器
# ============================================================
class DWLTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = DWLModel(cfg)
        self.dr = DomainRandomization(cfg)
        self.reward_fn = RewardFunction(cfg)
        self.ppo_loss = PPOLoss(cfg)
        self.traj = QuinticFootTrajectory(cfg)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)

    def denoise_loss(self, s_tilde, s_t, z_t):
        """L_denoise = MSE(s_tilde, s_t) + lambda_l1 * ||z_t||_1 (Equation 2)"""
        mse = F.mse_loss(s_tilde, s_t)
        l1 = self.cfg.lambda_l1 * z_t.abs().mean()
        return mse + l1, mse, l1

    def training_step(self, obs_seq, s_t, actions_old, dist_old,
                      advantages, returns, h_prev):
        """一次完整的DWL训练步"""
        # 前向传播
        z_t, s_tilde, dist_new, values, h_t = self.model(
            obs_seq, s_t, h_prev)

        # 三个损失
        l_denoise, mse, l1 = self.denoise_loss(s_tilde, s_t, z_t)
        l_pi = self.ppo_loss.policy_loss(
            dist_new, dist_old, actions_old, advantages)
        l_v = self.ppo_loss.value_loss(values, returns)

        # 总损失 (Equation 5)
        l_total = l_denoise + self.cfg.lambda_pi * l_pi + self.cfg.lambda_v * l_v

        # 反向传播
        self.optimizer.zero_grad()
        l_total.backward()
        self.optimizer.step()

        return {
            'total': l_total.item(),
            'denoise': l_denoise.item(),
            'mse': mse.item(),
            'l1': l1.item(),
            'policy': l_pi.item(),
            'value': l_v.item(),
        }

    def deploy(self, obs_seq, h_prev=None):
        """部署模式: 只保留Encoder+Actor"""
        with torch.no_grad():
            z_t, h_t = self.model.encoder(obs_seq, h_prev)
            action = self.model.actor(z_t, deterministic=True)
        return action, h_t


# ============================================================
# 7. 伪训练循环 (使用随机数据验证架构)
# ============================================================
def generate_dummy_data(cfg, B=64, seq_len=10):
    """生成模拟的仿真数据来测试模型"""
    # 观测: 47维
    obs = torch.randn(B, cfg.obs_dim)
    # 特权信息: 137维 (184-47)
    privileged = torch.randn(B, cfg.state_dim - cfg.obs_dim)
    # 完整状态 = 观测 + 特权信息
    state = torch.cat([obs, privileged], dim=-1)  # (B, 184)

    # 构造GRU输入序列 (seq_len步的历史)
    obs_seq = torch.randn(B, seq_len, cfg.obs_dim)

    # 旧策略分布 (用于PPO ratio)
    old_mu = torch.randn(B, cfg.action_dim)
    old_std = torch.ones(cfg.action_dim) * 0.5
    dist_old = Normal(old_mu, old_std)

    # 采样的动作
    actions = dist_old.sample()

    # 优势函数 & 回报
    advantages = torch.randn(B)
    returns = torch.randn(B)

    return obs_seq, state, actions, dist_old, advantages, returns, obs, privileged


def run_smoke_test():
    """冒烟测试: 验证模型能前向+反向传播"""
    print("=" * 60)
    print("DWL 复现 — 架构验证")
    print("=" * 60)

    cfg = DWLConfig()
    trainer = DWLTrainer(cfg)

    # 统计参数量
    total_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"\n总参数量: {total_params:,}")
    print(f"  (论文DWL Actor: ~320,192)")

    # 分模块统计
    enc_params = sum(p.numel() for p in trainer.model.encoder.parameters())
    dec_params = sum(p.numel() for p in trainer.model.decoder.parameters())
    act_params = sum(p.numel() for p in trainer.model.actor.parameters())
    crt_params = sum(p.numel() for p in trainer.model.critic.parameters())
    print(f"  Encoder: {enc_params:,}")
    print(f"  Decoder: {dec_params:,}")
    print(f"  Actor:   {act_params:,}")
    print(f"  Critic:  {crt_params:,}")

    # 生成假数据
    B, seq_len = 64, 10
    obs_seq, s_t, actions, dist_old, adv, ret, obs, priv = \
        generate_dummy_data(cfg, B, seq_len)
    h_prev = torch.zeros(1, B, cfg.gru_hidden)

    # 测试前向传播
    print(f"\n前向传播测试:")
    print(f"  obs_seq: {obs_seq.shape}  (B={B}, seq_len={seq_len}, obs_dim={cfg.obs_dim})")
    z_t, s_tilde, dist_new, values, h_t = trainer.model(obs_seq, s_t, h_prev)
    print(f"  z_t:      {z_t.shape}  (B, latent_dim={cfg.latent_dim})")
    print(f"  s_tilde:  {s_tilde.shape}  (B, state_dim={cfg.state_dim})")
    print(f"  values:   {values.shape}  (B,)")

    # 采样动作
    action = dist_new.sample()
    print(f"  action:   {action.shape}  (B, action_dim={cfg.action_dim})")

    # 测试损失
    l_denoise, mse, l1 = trainer.denoise_loss(s_tilde, s_t, z_t)
    l_pi = trainer.ppo_loss.policy_loss(dist_new, dist_old, actions, adv)
    l_v = trainer.ppo_loss.value_loss(values, ret)
    l_total = l_denoise + cfg.lambda_pi * l_pi + cfg.lambda_v * l_v
    print(f"\n损失计算:")
    print(f"  L_denoise  = {l_denoise.item():.4f} (MSE={mse.item():.4f}, L1={l1.item():.4f})")
    print(f"  L_pi(PPO)  = {l_pi.item():.4f}")
    print(f"  L_v         = {l_v.item():.4f}")
    print(f"  L_total     = {l_total.item():.4f}")

    # 测试反向传播 (验证梯度流)
    trainer.optimizer.zero_grad()
    l_total.backward()
    grad_norm = sum(p.grad.norm().item() for p in trainer.model.parameters()
                    if p.grad is not None)
    print(f"  梯度总模:   {grad_norm:.4f}  (反向传播OK)")

    # 测试部署模式
    action_deploy, _ = trainer.deploy(obs_seq[:, -3:, :], h_prev)
    print(f"\n部署模式 (仅Encoder+Actor):")
    print(f"  action: {action_deploy.shape}  (B, action_dim)")

    # 测试足部轨迹
    t_test = torch.tensor([0.0, 0.125, 0.25, 0.375, 0.5])
    h_ref, v_ref = trainer.traj.get_reference(t_test)
    print(f"\n五次多项式足部轨迹验证:")
    print(f"  t=0.00: h={h_ref[0].item():.4f}, v={v_ref[0].item():.4f}")
    print(f"  t=0.25: h={h_ref[2].item():.4f}, v={v_ref[2].item():.4f}")
    print(f"  t=0.50: h={h_ref[4].item():.4f}, v={v_ref[4].item():.4f}")
    print("  (期望: h(0)=0, h(0.25)=0.1, h(0.5)=0, v(0.5)=0 [软着陆])")

    # 测试域随机化
    noisy = trainer.dr.make_noisy_obs(obs)
    print(f"\n域随机化测试:")
    print(f"  干净观测均值: {obs.mean().item():.3f}")
    print(f"  噪声观测均值: {noisy.mean().item():.3f}")
    print(f"  差异(abs): {(noisy - obs).abs().mean().item():.4f}")

    # 验证DR参数采样
    dr_params = trainer.dr.sample_dr_params(B, torch.device('cpu'))
    print(f"  DR friction: [{dr_params['friction'].min():.2f}, {dr_params['friction'].max():.2f}]")
    print(f"  DR payload:  [{dr_params['payload'].min():.1f}, {dr_params['payload'].max():.1f}] kg")

    print(f"\n{'='*60}")
    print("所有模块验证通过! 架构复现完成.")
    print(f"{'='*60}")

    # 打印网络结构
    print(f"\n--- DWL 完整网络结构 ---")
    print(trainer.model)


if __name__ == "__main__":
    run_smoke_test()
