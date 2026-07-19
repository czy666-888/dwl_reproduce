"""Diagnostic: measure rollout speed"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import torch, numpy as np
from dwl_config import DWLConfig
from dwl_networks import DWLModel, count_parameters
from dwl_env import VecEnv
from dwl_ppo import DWLTrainer, RolloutBuffer

cfg = DWLConfig()
cfg.num_envs = 4
N = cfg.num_envs
device = torch.device('cuda')
print(f'Device: {device}')

# Env
t0 = time.time()
vec_env = VecEnv(cfg, num_envs=N)
print(f'Env creation: {time.time()-t0:.1f}s')

obs_batch, priv_batch = vec_env.reset()
obs_t = torch.from_numpy(obs_batch).to(device)
priv_t = torch.from_numpy(priv_batch).to(device)

# Model
model = DWLModel(cfg).to(device)
trainer = DWLTrainer(model, cfg, device=device)

seq_len = vec_env.seq_len
buffer = RolloutBuffer(N, cfg.num_steps_per_env, (cfg.obs_dim,), (cfg.privileged_dim,),
                       cfg.num_actions, cfg.state_dim, seq_len, device)

# Time rollout
print(f'Rollout: {cfg.num_steps_per_env} steps x {N} envs...')
t0 = time.time()
for step in range(cfg.num_steps_per_env):
    obs_seq = vec_env.get_obs_sequences()
    obs_seq_t = torch.from_numpy(obs_seq).to(device)

    with torch.no_grad():
        action, log_prob, _ = model.act(obs_seq_t)
        s_t = torch.cat([obs_t, torch.zeros(N, cfg.state_dim - cfg.obs_dim, device=device)], dim=-1)
        if s_t.shape[1] < cfg.state_dim:
            s_t = torch.nn.functional.pad(s_t, (0, cfg.state_dim - s_t.shape[1]))
        value = model.critic(s_t)

    actions_np = action.cpu().numpy()
    next_obs, next_priv, rewards, dones, infos = vec_env.step(actions_np)

    next_obs_t = torch.from_numpy(next_obs).to(device)
    rewards_t = torch.from_numpy(rewards).to(device)
    dones_t = torch.from_numpy(dones.astype(np.float32)).to(device)

    buffer.add(obs_t, obs_seq_t, priv_t, action, log_prob, rewards_t, dones_t, value, s_t)
    obs_t = next_obs_t
    priv_t = torch.from_numpy(next_priv).to(device)

t_rollout = time.time() - t0
per_env_step = t_rollout / N / cfg.num_steps_per_env * 1000
print(f'  Rollout: {t_rollout:.2f}s ({per_env_step:.1f}ms/env-step)')

# GAE + Update
advantages, returns = trainer.compute_gae(buffer.rewards, buffer.values, buffer.dones,
                                           torch.zeros(N, device=device), cfg.gamma, cfg.gae_lambda)
buf_data = buffer.get_all()
buf_data['advantages'] = advantages.view(-1)
buf_data['returns'] = returns.view(-1)

t0 = time.time()
metrics = trainer.update(buf_data)
t_update = time.time() - t0
print(f'  Update: {t_update:.2f}s')
print(f'  Total/iter (4 envs): {t_rollout + t_update:.1f}s')

# Extrapolate to 256 envs
t_total = t_rollout + t_update
est_256 = t_rollout * (256 / N) + t_update
print(f'\nEstimated for 256 envs: {est_256:.1f}s/iter')
print(f'Estimated for 10 iters (first log): {est_256*10:.0f}s = {est_256*10/60:.1f}min')

vec_env.close()
print('Done!')
