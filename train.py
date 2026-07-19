"""
DWL + PPO 主训练脚本
main() 函数实现完整训练循环:
  Rollout (采集数据) -> GAE (计算优势) -> DWL Update (联合训练)
"""
import os
import sys
import time
import csv
import argparse
import numpy as np
import torch

# 添加当前目录到 path
sys.path.insert(0, os.path.dirname(__file__))

from dwl_config import DWLConfig
from dwl_networks import DWLModel, count_parameters
from dwl_env import VecEnv
from dwl_ppo import DWLTrainer, RolloutBuffer


class BehaviorLogger:
    """记录每个iteration的机器人行为数据到CSV"""

    BEHAVIOR_KEYS = [
        # 速度
        'cmd_vx', 'cmd_vy', 'cmd_vyaw',
        'actual_vx', 'actual_vy', 'actual_vyaw',
        'base_vel_x_world', 'base_vel_y_world',
        # 姿态
        'roll', 'pitch', 'yaw',
        'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
        'gravity_z',
        # 基座
        'base_height', 'height_error',
        'base_x', 'base_y',
        # 足部
        'left_foot_z', 'right_foot_z',
        'left_foot_x', 'right_foot_x',
        'contact_left', 'contact_right',
        'foot_slip_raw',
        # 步态
        'gait_sin', 'stance_left', 'stance_right',
        # 关节
        'left_knee_q', 'right_knee_q', 'dof_vel_mean',
        # 力矩 / 动作
        'torque_mean', 'torque_max',
        'action_mean', 'action_max',
        # DWL 论文 11 项奖励 (Table V)
        'r_lin_vel', 'r_ang_vel', 'r_orientation', 'r_height',
        'r_periodic_contact', 'r_periodic_vel',
        'r_foot_height', 'r_foot_vel',
        'r_default_joint', 'r_energy', 'r_action_smooth',
    ]

    def __init__(self, save_dir, num_envs):
        self.save_dir = save_dir
        self.num_envs = num_envs
        self.filename = os.path.join(save_dir, 'behavior_log.csv')
        self.file = None
        self.writer = None
        self._open()

    def _open(self):
        os.makedirs(self.save_dir, exist_ok=True)
        # 追加模式，断点续跑不丢失旧数据
        file_exists = os.path.exists(self.filename)
        self.file = open(self.filename, 'a', newline='', encoding='utf-8-sig')
        self.writer = csv.writer(self.file)
        if not file_exists:
            header = ['iteration'] + self.BEHAVIOR_KEYS + ['num_falls', 'fall_pct', 'mean_reward']
            self.writer.writerow(header)
        self.file.flush()

    def log(self, iteration, all_behavior, all_rewards, all_dones, all_term_reasons):
        """all_behavior: list of dict (len=num_envs * num_steps_per_env)"""
        if not all_behavior:
            return

        # 合并所有env、所有step的behavior数据, 取均值
        agg = {}
        for key in self.BEHAVIOR_KEYS:
            vals = [b[key] for b in all_behavior if key in b]
            agg[key] = np.mean(vals) if vals else 0.0

        # 统计摔倒
        n_falls = sum(1 for r in all_term_reasons if r != 'timeout')
        n_total = len(all_term_reasons)
        fall_pct = n_falls / n_total * 100 if n_total > 0 else 0.0

        mean_reward = np.mean(all_rewards) if len(all_rewards) > 0 else 0.0

        row = [iteration] + [agg[k] for k in self.BEHAVIOR_KEYS] + [n_falls, fall_pct, mean_reward]
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


def _build_state(obs, priv, state_dim, device):
    """构建完整状态: cat(obs, priv) → pad/trim → (B, state_dim)"""
    s = torch.cat([obs, priv], dim=-1)
    if s.shape[1] < state_dim:
        s = torch.cat([s, torch.zeros(s.shape[0], state_dim - s.shape[1], device=device)], dim=-1)
    return s[:, :state_dim]


def train(args):
    cfg = DWLConfig()

    # 覆盖参数
    if args.num_envs:
        cfg.num_envs = args.num_envs
    if args.lr:
        cfg.learning_rate = args.lr
    if args.num_iterations:
        num_iterations = args.num_iterations
    else:
        num_iterations = args.num_iterations or 3000

    # 设备
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Using: {device}")
    if device.type == 'cuda':
        print(f"[GPU] {torch.cuda.get_device_name(0)} | Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    # 创建环境
    print(f"[Env] Creating {cfg.num_envs} MuJoCo environments...")
    vec_env = VecEnv(cfg, num_envs=cfg.num_envs)

    # 创建模型
    print("[Model] Initializing DWL architecture...")
    model = DWLModel(cfg)
    n_params = count_parameters(model)
    print(f"[Model] Total parameters: {n_params:,}")

    # 创建训练器
    trainer = DWLTrainer(model, cfg, device=device)
    trainer.init_writer(os.path.join(cfg.log_dir, args.run_name or 'dwl_run'))

    # 加载checkpoint (如果指定)
    resume_iter = 0
    if args.resume:
        print(f"[Checkpoint] Loading: {args.resume}")
        trainer.load(args.resume)
        resume_iter = trainer.global_step
        print(f"[Checkpoint] Resuming from iteration {resume_iter}")

    # Rollout Buffer
    seq_len = vec_env.seq_len
    buffer = RolloutBuffer(
        cfg.num_envs, cfg.num_steps_per_env,
        (cfg.obs_dim,), (cfg.privileged_dim,), cfg.num_actions,
        cfg.state_dim, seq_len, device)

    # 初始观测
    obs_batch, priv_batch = vec_env.reset()
    obs_batch = torch.from_numpy(obs_batch).to(device)
    priv_batch = torch.from_numpy(priv_batch).to(device)

    # GRU隐藏状态
    h_prev = torch.zeros(1, cfg.num_envs, cfg.gru_hidden, device=device)

    # ============================================================
    # 训练循环
    # ============================================================
    total_iterations = resume_iter + num_iterations
    print(f"\n[Training] Starting from iter {resume_iter + 1} to {total_iterations} ({num_iterations} new iterations)...")
    print("=" * 70)

    start_time = time.time()
    total_steps = 0
    best_mean_reward = -float('inf')

    # Behavior 日志 (每5个iteration记录一次)
    behavior_logger = BehaviorLogger(os.path.join(cfg.log_dir, args.run_name or 'dwl_run'), cfg.num_envs)
    log_interval = 5

    for iteration in range(resume_iter + 1, total_iterations + 1):
        iter_start = time.time()

        # ---- Phase 1: Rollout ----
        buffer.clear()
        ep_rewards = []
        # 收集本iteration的行为数据
        iter_behavior = []
        iter_term_reasons = []

        for step in range(cfg.num_steps_per_env):
            # 获取观测序列
            obs_seq = vec_env.get_obs_sequences()
            obs_seq_t = torch.from_numpy(obs_seq).to(device)

            # 采样动作
            with torch.no_grad():
                actions_t, log_probs_t, _ = model.act(obs_seq_t, h_prev)
                # 构建完整状态: obs + privileged → state_dim
                state_t = _build_state(obs_batch, priv_batch, cfg.state_dim, device)
                values_t = model.critic(state_t)

            actions_np = actions_t.cpu().numpy()

            # 环境步进 (记录行为数据)
            next_obs, next_priv, rewards, dones, infos, behaviors = vec_env.step_with_behavior(actions_np)
            iter_behavior.extend(behaviors)
            # 记录终止的环境
            for info in infos:
                if info.get('terminated', False):
                    iter_term_reasons.append(info.get('term_reason', 'timeout'))

            next_obs_t = torch.from_numpy(next_obs).to(device)
            next_priv_t = torch.from_numpy(next_priv).to(device)
            rewards_t = torch.from_numpy(rewards).to(device)
            dones_t = torch.from_numpy(dones.astype(np.float32)).to(device)

            # 构建完整状态: obs(47) + priv(137) → state_dim(184)
            state_t = _build_state(obs_batch, priv_batch, cfg.state_dim, device)

            # 存入buffer
            buffer.add(obs_batch, obs_seq_t, priv_batch, actions_t, log_probs_t,
                      rewards_t, dones_t, values_t, state_t)

            obs_batch = next_obs_t
            priv_batch = next_priv_t

            ep_rewards.append(rewards_t.mean().item())

            # 更新GRU隐藏状态 (终止的环境重置)
            h_prev[:, dones_t.bool(), :] = 0.0

            total_steps += cfg.num_envs

        mean_ep_reward = np.mean(ep_rewards)

        # ---- Phase 2: GAE ----
        # 计算最后一个状态的价值
        with torch.no_grad():
            last_state = _build_state(obs_batch, priv_batch, cfg.state_dim, device)
            last_values = model.critic(last_state)

        advantages, returns = trainer.compute_gae(
            buffer.rewards, buffer.values, buffer.dones,
            last_values, cfg.gamma, cfg.gae_lambda)

        # ---- Phase 3: DWL Update ----
        buffer_data = buffer.get_all()
        buffer_data['advantages'] = advantages.view(-1)
        buffer_data['returns'] = returns.view(-1)
        # obs_seq already flattened by get_all() to (T*N, seq_len, obs_dim)

        metrics = trainer.update(buffer_data)

        # ---- Phase 4: Logging ----
        iteration_time = time.time() - iter_start
        steps_per_sec = (cfg.num_envs * cfg.num_steps_per_env) / iteration_time

        if trainer.writer:
            for k, v in metrics.items():
                trainer.writer.add_scalar(k, v, iteration)
            trainer.writer.add_scalar('reward/mean', mean_ep_reward, iteration)
            trainer.writer.add_scalar('perf/steps_per_sec', steps_per_sec, iteration)

        trainer.global_step += 1

        # 记录行为数据 (每log_interval次)
        if iteration % log_interval == 0 and iter_behavior:
            behavior_logger.log(iteration, iter_behavior, ep_rewards, dones.cpu().numpy() if hasattr(dones, 'cpu') else dones, iter_term_reasons)

        # 打印
        if iteration % 1 == 0:
            elapsed = time.time() - start_time
            print(f"Iter {iteration:5d}/{total_iterations} | "
                  f"Reward: {mean_ep_reward:7.3f} | "
                  f"Loss: {metrics['loss/total']:7.4f} | "
                  f"Steps/s: {steps_per_sec:7.0f} | "
                  f"Time: {elapsed/60:.1f}min")

        # 保存最优模型
        if mean_ep_reward > best_mean_reward:
            best_mean_reward = mean_ep_reward
            trainer.save(os.path.join(cfg.checkpoint_dir, 'best_model.pt'))

        # 定期保存
        if iteration % cfg.save_interval == 0:
            trainer.save(os.path.join(cfg.checkpoint_dir, f'model_iter_{iteration}.pt'))
            print(f"  [Checkpoint] Saved at iter {iteration}")

    # 训练结束
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Training complete! Total time: {total_time/3600:.1f}h")
    print(f"Best mean reward: {best_mean_reward:.4f}")
    print(f"Final model saved to: {cfg.checkpoint_dir}/final_model.pt")

    trainer.save(os.path.join(cfg.checkpoint_dir, 'final_model.pt'))
    behavior_logger.close()
    vec_env.close()

    if trainer.writer:
        trainer.writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DWL + PPO Training for XBot-L Humanoid Locomotion')
    parser.add_argument('--num_envs', type=int, default=None,
                        help='Number of parallel environments (default: from config)')
    parser.add_argument('--num_iterations', type=int, default=3000,
                        help='Number of training iterations')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: from config)')
    parser.add_argument('--run_name', type=str, default='dwl_run',
                        help='Run name for logging')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--smoke_test', action='store_true',
                        help='Quick smoke test to verify all components work')

    args = parser.parse_args()

    if args.smoke_test:
        print("=" * 60)
        print("DWL Smoke Test — 验证所有组件")
        print("=" * 60)

        cfg = DWLConfig()
        cfg.num_envs = 4
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")

        # 1. 测试 MuJoCo 环境
        print("\n[1/5] Testing MuJoCo environment...")
        vec_env = VecEnv(cfg, num_envs=4)
        obs, priv = vec_env.reset()
        print(f"  obs: {obs.shape} (expected: (4, {cfg.obs_dim}))")
        print(f"  privileged: {priv.shape} (expected: (4, {cfg.privileged_dim}))")

        actions = np.zeros((4, cfg.num_actions))
        next_obs, next_priv, rewards, dones, infos = vec_env.step(actions)
        print(f"  next_obs: {next_obs.shape}")
        print(f"  rewards: {rewards}")
        print(f"  dones: {dones}")
        print("  MuJoCo environment: OK")

        # 2. 测试 DWL 模型
        print("\n[2/5] Testing DWL Model...")
        model = DWLModel(cfg).to(device)
        n_params = count_parameters(model)
        B = cfg.num_envs
        obs_seq = torch.randn(B, 5, cfg.obs_dim, device=device)
        s_t = torch.randn(B, cfg.state_dim, device=device)
        z_t, s_tilde, dist, value, h_t = model(obs_seq, s_t)
        print(f"  z_t: {z_t.shape} | s_tilde: {s_tilde.shape} | value: {value.shape}")
        print(f"  Parameters: {n_params:,}")
        print("  DWL Model: OK")

        # 3. 测试前向+反向
        print("\n[3/5] Testing Forward+Backward...")
        trainer = DWLTrainer(model, cfg, device=device)
        l_denoise, mse, l1 = trainer.denoise_loss(s_tilde, s_t, z_t)
        l_total = l_denoise
        l_total.backward()
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        print(f"  L_denoise: {l_denoise.item():.4f} | Grad norm: {grad_norm:.4f}")
        print("  Forward+Backward: OK")

        # 4. 测试 GAE
        print("\n[4/5] Testing GAE...")
        T = cfg.num_steps_per_env
        dummy_rewards = torch.randn(T, B, device=device)
        dummy_values = torch.randn(T, B, device=device)
        dummy_dones = torch.zeros(T, B, device=device)
        next_values = torch.zeros(B, device=device)
        adv, ret = trainer.compute_gae(dummy_rewards, dummy_values, dummy_dones, next_values, cfg.gamma, cfg.gae_lambda)
        print(f"  advantages: {adv.shape} | returns: {ret.shape}")
        print("  GAE: OK")

        # 5. 测试完整训练步
        print("\n[5/5] Testing 1 full training iteration...")
        cfg.num_epochs = 1
        buffer = RolloutBuffer(B, T, (cfg.obs_dim,), (cfg.privileged_dim,), cfg.num_actions, cfg.state_dim, 5, device)
        # 收集几步
        obs, priv = vec_env.reset()
        obs_t = torch.from_numpy(obs).to(device)
        priv_t = torch.from_numpy(priv).to(device)
        seqs = vec_env.get_obs_sequences()
        seqs_t = torch.from_numpy(seqs).to(device)

        for _ in range(T):
            with torch.no_grad():
                action, log_prob, _ = model.act(seqs_t)
                s_t_dummy = _build_state(obs_t, priv_t, cfg.state_dim, device)
                value = model.critic(s_t_dummy)
            next_obs, next_priv, rewards, dones, info = vec_env.step(action.cpu().numpy())
            next_obs_t = torch.from_numpy(next_obs).to(device)
            next_priv_t = torch.from_numpy(next_priv).to(device)
            rewards_t = torch.from_numpy(rewards).to(device)
            dones_t = torch.from_numpy(dones.astype(np.float32)).to(device)

            s_t_full = _build_state(obs_t, priv_t, cfg.state_dim, device)

            buffer.add(obs_t, seqs_t, priv_t, action, log_prob,
                      rewards_t, dones_t, value, s_t_full)
            obs_t = next_obs_t
            priv_t = next_priv_t
            seqs = vec_env.get_obs_sequences()
            seqs_t = torch.from_numpy(seqs).to(device)

        # GAE
        adv, ret = trainer.compute_gae(
            buffer.rewards, buffer.values, buffer.dones,
            torch.zeros(B, device=device), cfg.gamma, cfg.gae_lambda)

        # DWL update
        buf_data = buffer.get_all()
        buf_data['advantages'] = adv.view(-1)
        buf_data['returns'] = ret.view(-1)
        # obs_seq already flattened by get_all()

        metrics = trainer.update(buf_data)
        print(f"  Update metrics: {metrics}")
        print("  1 full training iteration: OK")

        vec_env.close()
        print(f"\n{'='*60}")
        print("ALL SMOKE TESTS PASSED!")
        print(f"{'='*60}")

    else:
        train(args)
