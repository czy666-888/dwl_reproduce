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
        """DWL + PPO 联合更新"""
        obs_seq = buffer_data['obs_seq']      # (N, seq_len, obs_dim)
        s_t = buffer_data['states']            # (N, state_dim)
        actions = buffer_data['actions']       # (N, action_dim)
        old_log_probs = buffer_data['log_probs']  # (N,)
        advantages = buffer_data['advantages']    # (N,)
        returns = buffer_data['returns']          # (N,)

        N = obs_seq.shape[0]
        batch_size = N // self.cfg.num_mini_batches

        total_losses = []
        total_denoise = []
        total_pi = []
        total_v = []

        for epoch in range(self.cfg.num_epochs):
            # 随机打乱
            indices = torch.randperm(N, device=self.device)

            for mb in range(self.cfg.num_mini_batches):
                start = mb * batch_size
                end = min(start + batch_size, N)
                mb_idx = indices[start:end]

                mb_obs_seq = obs_seq[mb_idx]
                mb_s_t = s_t[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                # 前向传播
                new_log_prob, value, s_tilde, z_t, entropy = self.model.evaluate(
                    mb_obs_seq, mb_s_t, mb_actions)

                # --- 1. 去噪损失 ---
                l_denoise, mse, l1 = self.denoise_loss(s_tilde, mb_s_t, z_t)

                # --- 2. PPO Policy Loss ---
                ratio = torch.exp(new_log_prob - mb_old_log_probs)
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
