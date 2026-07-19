"""
DWL 网络架构 (Denoising World Model Learning)
论文 Table VI — XBot-L 适配版

GRU Encoder: 观测序列 -> 去噪隐状态
Decoder:     隐状态 -> 完整状态重建
Actor:       隐状态 -> 动作分布
Critic:      完整状态 -> 价值标量
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class GRUEncoder(nn.Module):
    """去噪编码器: 含噪观测序列 -> 隐状态
    GRU(obs_dim -> gru_hidden) -> MLP(gru_hidden -> latent_dim)
    """
    def __init__(self, obs_dim, gru_hidden, latent_dim):
        super().__init__()
        self.gru_hidden = gru_hidden
        self.gru = nn.GRU(input_size=obs_dim, hidden_size=gru_hidden,
                          num_layers=1, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden),
            nn.ELU(alpha=1.0),
            nn.Linear(gru_hidden, latent_dim),
        )

    def forward(self, o_seq, h_prev=None):
        """o_seq: (B, seq_len, obs_dim) 历史观测序列
           h_prev: (1, B, gru_hidden) 可选上一隐藏状态
           返回: z_t (B, latent_dim), h_t (1, B, gru_hidden)"""
        gru_out, h_t = self.gru(o_seq, h_prev)
        z_t = self.proj(gru_out[:, -1, :])   # 取最后一帧
        return z_t, h_t


class Decoder(nn.Module):
    """状态重建器: 隐状态 -> 完整状态估计
    MLP(latent_dim -> 64 -> state_dim)
    """
    def __init__(self, latent_dim, state_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ELU(alpha=1.0),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, z_t):
        """z_t: (B, latent_dim)
           返回: s_tilde (B, state_dim)"""
        return self.net(z_t)


class Actor(nn.Module):
    """策略网络: 隐状态 -> 动作分布 (Gaussian)
    MLP(latent_dim -> 48 -> action_dim) + log_std
    """
    def __init__(self, latent_dim, action_dim, hidden=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ELU(alpha=1.0),
            nn.Linear(hidden, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, z_t, deterministic=False):
        mu = self.net(z_t)
        if deterministic:
            return mu
        std = torch.exp(self.log_std)
        return Normal(mu, std)


class Critic(nn.Module):
    """价值网络: 完整状态 -> 价值
    MLP(state_dim -> 512 -> 512 -> 256 -> 1)
    """
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ELU(alpha=1.0),
            nn.Linear(512, 512),
            nn.ELU(alpha=1.0),
            nn.Linear(512, 256),
            nn.ELU(alpha=1.0),
            nn.Linear(256, 1),
        )

    def forward(self, s_t):
        return self.net(s_t).squeeze(-1)


class DWLModel(nn.Module):
    """DWL 完整模型: Encoder + Decoder + Actor + Critic"""
    def __init__(self, cfg):
        super().__init__()
        self.encoder = GRUEncoder(cfg.obs_dim, cfg.gru_hidden, cfg.latent_dim)
        self.decoder = Decoder(cfg.latent_dim, cfg.state_dim)
        self.actor = Actor(cfg.latent_dim, cfg.num_actions)
        self.critic = Critic(cfg.state_dim)
        self.cfg = cfg

    def forward(self, obs_seq, s_t, h_prev=None, deterministic=False):
        """完整前向传播
        obs_seq: (B, seq_len, obs_dim)
        s_t:     (B, state_dim) — 仅训练时使用 (Critic输入)
        h_prev:  (1, B, gru_hidden) 可选的GRU隐藏状态
        返回: z_t, s_tilde, dist/value, h_t
        """
        z_t, h_t = self.encoder(obs_seq, h_prev)
        s_tilde = self.decoder(z_t)

        if deterministic:
            action_mean = self.actor(z_t, deterministic=True)
            value = self.critic(s_t)
            return z_t, s_tilde, action_mean, value, h_t
        else:
            dist = self.actor(z_t, deterministic=False)
            value = self.critic(s_t)
            return z_t, s_tilde, dist, value, h_t

    def act_inference(self, obs_seq, h_prev=None):
        """部署模式: 仅 Encoder + Actor"""
        with torch.no_grad():
            z_t, h_t = self.encoder(obs_seq, h_prev)
            action = self.actor(z_t, deterministic=True)
        return action, h_t

    def act(self, obs_seq, h_prev=None):
        """训练采样模式"""
        with torch.no_grad():
            z_t, h_t = self.encoder(obs_seq, h_prev)
            dist = self.actor(z_t, deterministic=False)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, h_t

    def evaluate(self, obs_seq, s_t, action, h_prev=None):
        """评估模式: 返回 log_prob, value, s_tilde, z_t"""
        z_t, h_t = self.encoder(obs_seq, h_prev)
        s_tilde = self.decoder(z_t)
        dist = self.actor(z_t)
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(s_t)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, value, s_tilde, z_t, entropy


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    import sys; sys.path.insert(0, '.')
    from dwl_config import DWLConfig
    cfg = DWLConfig()
    model = DWLModel(cfg)

    print("=" * 50)
    print("DWL 模型参数统计")
    print("=" * 50)
    print(f"Encoder: {count_parameters(model.encoder):,}")
    print(f"Decoder: {count_parameters(model.decoder):,}")
    print(f"Actor:   {count_parameters(model.actor):,}")
    print(f"Critic:  {count_parameters(model.critic):,}")
    print(f"Total:   {count_parameters(model):,}")

    # 冒烟测试
    B, seq_len = 4, 10
    obs_seq = torch.randn(B, seq_len, cfg.obs_dim)
    s_t = torch.randn(B, cfg.state_dim)
    z_t, s_tilde, dist, value, h_t = model(obs_seq, s_t)
    print(f"\nForward OK: z_t={z_t.shape}, s_tilde={s_tilde.shape}, value={value.shape}")

    # 损失测试
    mse = F.mse_loss(s_tilde, s_t)
    l1 = cfg.lambda_l1 * z_t.abs().mean()
    action = dist.sample()
    print(f"MSE={mse.item():.4f}, L1={l1.item():.4f}")
    print("Network architecture verified!")
