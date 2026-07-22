"""
DWL + PPO 训练器
实现: GAE优势估计 + PPO Clipped Update + DWL联合损失
L_total = L_denoise + 5.0 * L_pi + 5.0 * L_v
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class RolloutBuffer:
    """PPO Rollout 缓冲区"""
    def __init__(self, num_envs, num_steps, obs_shape, privileged_shape, action_dim,
                 state_dim, seq_len, device):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.device = device

        self.obs = torch.zeros(num_steps, num_envs, *obs_shape, device=device)
        self.obs_seq = torch.zeros(num_steps, num_envs, seq_len, *obs_shape, device=device)
        self.privileged = torch.zeros(num_steps, num_envs, *privileged_shape, device=device)
        self.actions = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.states = torch.zeros(num_steps, num_envs, state_dim, device=device)

        self.step = 0

    def add(self, obs, obs_seq, privileged, actions, log_probs, rewards, dones, values, states):
        self.obs[self.step] = obs
        self.obs_seq[self.step] = obs_seq
        self.privileged[self.step] = privileged
        self.actions[self.step] = actions
        self.log_probs[self.step] = log_probs
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.values[self.step] = values
        self.states[self.step] = states
        self.step += 1

    def clear(self):
        self.step = 0

    def get_all(self):
        """展平为 (num_steps * num_envs, ...)"""
        return {
            'obs': self.obs.view(-1, *self.obs.shape[2:]),
            'obs_seq': self.obs_seq.view(-1, *self.obs_seq.shape[2:]),
            'privileged': self.privileged.view(-1, *self.privileged.shape[2:]),
            'actions': self.actions.view(-1, *self.actions.shape[2:]),
            'log_probs': self.log_probs.view(-1),
            'rewards': self.rewards.view(-1),
            'dones': self.dones.view(-1),
            'values': self.values.view(-1),
            'states': self.states.view(-1, *self.states.shape[2:]),
        }


# ============================================================
# 对称增强: 利用人形机器人左右对称性, 数据量翻倍
# ============================================================

def _build_obs_mirror_idx(num_actions=12):
    """构建观测(47维)的镜像索引和取反掩码"""
    obs_dim = 47
    swap_idx = np.arange(obs_dim)
    negate = np.zeros(obs_dim, dtype=bool)

    # 关节/动作 swap: left[0:6] <-> right[6:12]
    # 观测中3组12维: joint_pos, joint_vel, last_action
    for left_start in [5, 17, 29]:
        right_start = left_start + 6
        for k in range(6):
            swap_idx[left_start + k] = right_start + k
            swap_idx[right_start + k] = left_start + k

    # 需要取反的维度
    negate[[3, 4, 41, 43, 44, 46]] = True  # cmd_vy, cmd_vyaw, ang_vel_x, ang_vel_z, euler_roll, euler_yaw

    return swap_idx, negate


def _build_action_mirror_idx(num_actions=12):
    """构建动作(12维)的镜像索引"""
    swap_idx = np.arange(num_actions)
    for k in range(6):
        swap_idx[k] = 6 + k
        swap_idx[6 + k] = k
    return swap_idx


def _build_state_mirror_idx(state_dim=184, obs_dim=47, priv_dim=137):
    """构建完整状态(184维)的镜像索引和取反掩码"""
    swap_idx = np.arange(state_dim)
    negate = np.zeros(state_dim, dtype=bool)

    # obs 部分 [0:47]
    obs_swap, obs_neg = _build_obs_mirror_idx()
    swap_idx[:obs_dim] = obs_swap
    negate[:obs_dim] = obs_neg

    # privileged 部分 [47:184]
    off = obs_dim  # offset = 47

    # [off+0:3]  base_lin_vel_body        → negate y
    negate[off + 1] = True
    # [off+3:6]  base_lin_vel_world       → negate y
    negate[off + 4] = True
    # [off+6:9]  base_ang_vel              → negate x, z
    negate[off + 6] = True
    negate[off + 8] = True
    # [off+9:12] proj_gravity              → negate y
    negate[off + 10] = True
    # [off+12:15] euler                     → negate roll, yaw
    negate[off + 12] = True
    negate[off + 14] = True
    # [off+15:20] height, dr_*              → 不变 (标量)

    # [off+20:22] stance (left, right)      → swap
    swap_idx[off + 20] = off + 21
    swap_idx[off + 21] = off + 20

    # [off+22:28] foot_positions (l_xyz, r_xyz) → swap left↔right, negate x
    for k in range(3):
        swap_idx[off + 22 + k] = off + 25 + k
        swap_idx[off + 25 + k] = off + 22 + k
    negate[off + 22] = True   # left_foot_x
    negate[off + 25] = True   # right_foot_x

    # [off+28:31] left_foot_vel  ↔ [off+31:34] right_foot_vel
    for k in range(3):
        swap_idx[off + 28 + k] = off + 31 + k
        swap_idx[off + 31 + k] = off + 28 + k

    # [off+34:36] contact_forces (l, r)     → swap
    swap_idx[off + 34] = off + 35
    swap_idx[off + 35] = off + 34
    # [off+36:38] contact_bool (l, r)       → swap
    swap_idx[off + 36] = off + 37
    swap_idx[off + 37] = off + 36

    # [off+38:50] torques (12)              → swap left[0:6]↔right[6:12]
    for k in range(6):
        swap_idx[off + 38 + k] = off + 44 + k
        swap_idx[off + 44 + k] = off + 38 + k
    # [off+50:62] dof_acc (12, zeros)       → swap
    for k in range(6):
        swap_idx[off + 50 + k] = off + 56 + k
        swap_idx[off + 56 + k] = off + 50 + k

    # [off+62:65] commands                   → negate vy, vyaw
    negate[off + 63] = True
    negate[off + 64] = True
    # [off+65:67] sin_pos, cos_pos          → 不变

    # [off+67:69] foot_swing_time (l, r)    → swap
    swap_idx[off + 67] = off + 68
    swap_idx[off + 68] = off + 67
    # [off+69:71] foot_liftoff_height (l,r) → swap
    swap_idx[off + 69] = off + 70
    swap_idx[off + 70] = off + 69

    # [off+71:83]  joint_pos (12)           → swap
    for k in range(6):
        swap_idx[off + 71 + k] = off + 77 + k
        swap_idx[off + 77 + k] = off + 71 + k
    # [off+83:95]  joint_vel (12)           → swap
    for k in range(6):
        swap_idx[off + 83 + k] = off + 89 + k
        swap_idx[off + 89 + k] = off + 83 + k
    # [off+95:107] last_action (12)         → swap
    for k in range(6):
        swap_idx[off + 95 + k] = off + 101 + k
        swap_idx[off + 101 + k] = off + 95 + k
    # [off+107:119] last_last_action (12)   → swap
    for k in range(6):
        swap_idx[off + 107 + k] = off + 113 + k
        swap_idx[off + 113 + k] = off + 107 + k
    # [off+119] episode_progress            → 不变
    # [off+120:137] padding zeros           → 不变

    return swap_idx, negate


# 预构建镜像索引 (模块加载时计算一次)
_obs_swap_idx, _obs_negate = _build_obs_mirror_idx()
_action_swap_idx = _build_action_mirror_idx()
_state_swap_idx, _state_negate = _build_state_mirror_idx()


def mirror_tensor(x, swap_idx, negate_mask, dim=-1):
    """对张量最后一维执行镜像操作: mirrored[i] = x[swap_idx[i]] * (1 if not negate else -1)"""
    mirrored = x[..., swap_idx]
    sign = torch.where(torch.tensor(negate_mask, device=x.device), -1.0, 1.0)
    # Broadcast sign to match x shape
    while sign.dim() < mirrored.dim():
        sign = sign.unsqueeze(0)
    return mirrored * sign


def mirror_buffer_data(buffer_data, device):
    """对称增强: 生成镜像数据并拼接, 数据量翻倍
    buffer_data: dict with keys 'obs_seq', 'states', 'actions', 'log_probs', 'advantages', 'returns'
    返回: augmented dict
    """
    obs_seq = buffer_data['obs_seq']       # (N, seq_len, 47)
    states = buffer_data['states']          # (N, 184)
    actions = buffer_data['actions']        # (N, 12)
    old_log_probs = buffer_data['log_probs']  # (N,)
    advantages = buffer_data['advantages']  # (N,)
    returns = buffer_data['returns']        # (N,)

    # 创建镜像
    action_swap = torch.tensor(_action_swap_idx, device=device)
    mirror_obs_seq = mirror_tensor(obs_seq, _obs_swap_idx, _obs_negate)
    mirror_states = mirror_tensor(states, _state_swap_idx, _state_negate)
    mirror_actions = actions[..., action_swap]

    # 拼接原始 + 镜像
    return {
        'obs_seq': torch.cat([obs_seq, mirror_obs_seq], dim=0),
        'states': torch.cat([states, mirror_states], dim=0),
        'actions': torch.cat([actions, mirror_actions], dim=0),
        'log_probs': torch.cat([old_log_probs, old_log_probs], dim=0),
        'advantages': torch.cat([advantages, advantages], dim=0),
        'returns': torch.cat([returns, returns], dim=0),
    }


class DWLTrainer:
    """DWL + PPO 联合训练器"""
    def __init__(self, model, cfg, device='cuda'):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device

        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

        # 损失权重
        self.lambda_pi = cfg.lambda_pi
        self.lambda_v = cfg.lambda_v
        self.lambda_l1 = cfg.lambda_l1
        self.clip_range = cfg.clip_range
        self.entropy_coef = cfg.entropy_coef
        self.max_grad_norm = cfg.max_grad_norm

        # 日志
        self.writer = None
        self.global_step = 0

    def init_writer(self, log_dir='logs'):
        if SummaryWriter is None:
            self.writer = None
            return
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def compute_gae(self, rewards, values, dones, next_values, gamma, lam):
        """GAE 优势估计
        rewards:  (T, num_envs)
        values:   (T, num_envs)
        dones:    (T, num_envs)
        next_values: (num_envs,)
        返回: advantages (T, num_envs), returns (T, num_envs)
        """
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(self.cfg.num_envs, device=self.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_values
            else:
                next_val = values[t + 1]
            delta = rewards[t] + gamma * next_val * (1.0 - dones[t].float()) - values[t]
            gae = delta + gamma * lam * (1.0 - dones[t].float()) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def denoise_loss(self, s_tilde, s_t, z_t):
        """L_denoise = MSE(s_tilde, s_t) + lambda_l1 * ||z_t||_1"""
        mse = F.mse_loss(s_tilde, s_t)
        l1 = self.lambda_l1 * z_t.abs().mean()
        return mse + l1, mse, l1

    def update(self, buffer_data):
        """DWL + PPO 联合更新 (对称增强 + KL早停 + 优势归一化)"""
        # 对称增强: 数据量翻倍
        buffer_data = mirror_buffer_data(buffer_data, self.device)

        obs_seq = buffer_data['obs_seq']      # (2N, seq_len, obs_dim)
        s_t = buffer_data['states']            # (2N, state_dim)
        actions = buffer_data['actions']       # (2N, action_dim)
        old_log_probs = buffer_data['log_probs']  # (2N,)
        advantages = buffer_data['advantages']    # (2N,)
        returns = buffer_data['returns']          # (2N,)

        N = obs_seq.shape[0]
        batch_size = max(1, N // self.cfg.num_mini_batches)

        total_losses = []
        total_denoise = []
        total_pi = []
        total_v = []

        for epoch in range(self.cfg.num_epochs):
            # 随机打乱
            indices = torch.randperm(N, device=self.device)
            epoch_kls = []

            for mb in range(self.cfg.num_mini_batches):
                start = mb * batch_size
                end = min(start + batch_size, N)
                if end <= start:
                    continue
                mb_idx = indices[start:end]

                mb_obs_seq = obs_seq[mb_idx]
                mb_s_t = s_t[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                # 优势归一化 (per mini-batch)
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # 前向传播
                new_log_prob, value, s_tilde, z_t, entropy = self.model.evaluate(
                    mb_obs_seq, mb_s_t, mb_actions)

                # Approximate KL: 0.5 * (log_prob_diff)^2
                log_diff = new_log_prob - mb_old_log_probs
                approx_kl = 0.5 * (log_diff ** 2).mean()
                epoch_kls.append(approx_kl.item())

                # --- 1. 去噪损失 ---
                l_denoise, mse, l1 = self.denoise_loss(s_tilde, mb_s_t, z_t)

                # --- 2. PPO Policy Loss ---
                ratio = torch.exp(log_diff)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_advantages
                l_pi = -torch.min(surr1, surr2).mean()
                l_pi = l_pi - self.entropy_coef * entropy.mean()

                # --- 3. Value Loss ---
                l_v = F.mse_loss(value, mb_returns)

                # --- 总损失 ---
                l_total = l_denoise + self.lambda_pi * l_pi + self.lambda_v * l_v

                # 反向传播
                self.optimizer.zero_grad()
                l_total.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_losses.append(l_total.item())
                total_denoise.append(l_denoise.item())
                total_pi.append(l_pi.item())
                total_v.append(l_v.item())

            # KL早停: 如果均值KL超过目标, 提前结束epoch
            if self.cfg.kl_early_stop and epoch_kls:
                mean_kl = np.mean(epoch_kls)
                if mean_kl > self.cfg.kl_target:
                    break

        metrics = {
            'loss/total': np.mean(total_losses),
            'loss/denoise': np.mean(total_denoise),
            'loss/policy': np.mean(total_pi),
            'loss/value': np.mean(total_v),
        }
        return metrics

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        return checkpoint
