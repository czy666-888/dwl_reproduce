"""v4 训练分析 — 论文版足部轨迹跟踪"""
import csv, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(__file__)
CSV = os.path.join(BASE, 'logs', 'dwl_run3', 'behavior_log.csv')

with open(CSV, 'r', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))

bh_iters = np.array([int(r['iteration']) for r in rows])

def get(k):
    return np.array([float(r[k]) for r in rows])

reward = get('mean_reward')
fall_pct = get('fall_pct')
base_height = get('base_height')
r_survival = get('r_survival')
r_orientation = get('r_orientation')
r_base_height = get('r_base_height')
r_vel_gate = get('r_vel_gate')
r_gated_lin = get('r_gated_lin_vel')
r_gated_ang = get('r_gated_ang_vel')
r_foot_h = get('r_foot_height')
r_foot_v = get('r_foot_vel')
r_torques = get('r_torques')
r_dof_vel = get('r_dof_vel')
r_default_j = get('r_default_joint')
r_contact_f = get('r_contact_force')
r_foot_slip = get('r_foot_slip')
r_action_s = get('r_action_smooth')
torque_mean = get('torque_mean')
roll_abs = abs(get('roll'))
pitch_abs = abs(get('pitch'))
dof_vel_mean = get('dof_vel_mean')

def smooth(d, w=10):
    if len(d) < w: return d
    return np.convolve(d, np.ones(w)/w, mode='valid')

w = 30

# ================================================================
# 图1: 总览 — Reward + Fall + Survival + Foot Trajectory
# ================================================================
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
fig1.suptitle('v4 Training Summary — 五次多项式足部轨迹跟踪\nXBot-L, 256 envs, 3000 iters',
              fontsize=14, fontweight='bold')

ax = axes1[0, 0]
ax.plot(bh_iters, reward, 'o', alpha=0.2, color='#1f77b4', markersize=2)
if len(reward) >= w: ax.plot(bh_iters[w-1:], smooth(reward, w), color='#1f77b4', linewidth=2.5)
ax.axhline(y=np.mean(reward), color='darkblue', ls='--', alpha=0.5, label=f'Mean: {np.mean(reward):.3f}')
ax.set_title(f'Mean Reward ({reward[0]:.3f} -> {reward[-1]:.3f})')
ax.set_xlabel('Iteration'); ax.set_ylabel('Reward'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes1[0, 1]
ax.fill_between(bh_iters, 0, fall_pct, alpha=0.3, color='red')
ax.plot(bh_iters, fall_pct, 'r-', linewidth=1.5)
ax.set_title(f'Fall Rate (always {np.mean(fall_pct):.0f}%)')
ax.set_xlabel('Iteration'); ax.set_ylabel('%'); ax.set_ylim(0, 105); ax.grid(True, alpha=0.3)

ax = axes1[1, 0]
ax.plot(bh_iters, r_survival, 'o', alpha=0.2, color='#2ca02c', markersize=2)
if len(r_survival) >= 5: ax.plot(bh_iters[4:], smooth(r_survival, 5), color='#2ca02c', linewidth=2)
ax.set_title(f'Survival ({r_survival[0]:.3f} -> {r_survival[-1]:.3f})')
ax.set_xlabel('Iteration'); ax.set_ylabel('Score'); ax.grid(True, alpha=0.3)

ax = axes1[1, 1]
ax.plot(bh_iters, r_foot_h, 'o', alpha=0.2, color='#ff7f0e', markersize=2, label='Foot Height Track')
ax.plot(bh_iters, r_foot_v, 'o', alpha=0.2, color='#9467bd', markersize=2, label='Foot Vel Track')
if len(bh_iters) >= 5:
    ax.plot(bh_iters[4:], smooth(r_foot_h, 5), color='#ff7f0e', linewidth=2)
    ax.plot(bh_iters[4:], smooth(r_foot_v, 5), color='#9467bd', linewidth=2)
ax.set_title(f'Foot Trajectory Tracking (h:{r_foot_h[0]:.3f}->{r_foot_h[-1]:.3f} v:{r_foot_v[0]:.3f}->{r_foot_v[-1]:.3f})')
ax.set_xlabel('Iteration'); ax.set_ylabel('Score'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
p1 = os.path.join(BASE, 'v4_summary.png')
plt.savefig(p1, dpi=150, bbox_inches='tight')
print(f'Saved: {p1}')

# ================================================================
# 图2: 奖励分项详情
# ================================================================
items = [
    ('Survival', r_survival, '#2ca02c'),
    ('Orientation', r_orientation, '#1f77b4'),
    ('Base Height', r_base_height, '#d62728'),
    ('Vel Gate', r_vel_gate, '#17becf'),
    ('Gated Lin Vel', r_gated_lin, '#1f77b4'),
    ('Gated Ang Vel', r_gated_ang, '#ff7f0e'),
    ('Foot Height Track', r_foot_h, '#ff7f0e'),
    ('Foot Vel Track', r_foot_v, '#9467bd'),
    ('Torques (raw)', r_torques, '#8c564b'),
    ('DOF Vel (raw)', r_dof_vel, '#bcbd22'),
    ('Default Joint', r_default_j, '#7f7f7f'),
    ('Contact Force', r_contact_f, '#e377c2'),
]

fig2, axes2 = plt.subplots(4, 3, figsize=(20, 18))
fig2.suptitle('v4 Reward Items — 5-iteration Rolling Means', fontsize=14, fontweight='bold')

for i, (label, data, color) in enumerate(items):
    ax = axes2[i // 3, i % 3]
    ax.plot(bh_iters, data, 'o', alpha=0.2, color=color, markersize=2)
    if len(data) >= 5:
        ax.plot(bh_iters[4:], smooth(data, 5), color=color, linewidth=2)
    delta = data[-1] - data[0]
    pct = (delta / abs(data[0]) * 100) if abs(data[0]) > 0.001 else 0
    ax.set_title(f'{label}  ({data[0]:.3f}->{data[-1]:.3f}, {pct:+.0f}%)')
    ax.set_xlabel('Iteration'); ax.grid(True, alpha=0.3)

plt.tight_layout()
p2 = os.path.join(BASE, 'v4_reward_items.png')
plt.savefig(p2, dpi=150, bbox_inches='tight')
print(f'Saved: {p2}')

# ================================================================
# 图3: 行为指标
# ================================================================
fig3, axes3 = plt.subplots(2, 3, figsize=(18, 12))
fig3.suptitle('v4 Behavior Metrics', fontsize=14, fontweight='bold')

plots = [
    ('Base Height (m)', base_height, 0.89, 'Target 0.89m'),
    ('|Roll| (rad)', roll_abs, None, ''),
    ('|Pitch| (rad)', pitch_abs, None, ''),
    ('Mean Torque (Nm)', torque_mean, None, ''),
    ('DOF Vel Mean (rad/s)', dof_vel_mean, None, ''),
    ('Foot Slip', r_foot_slip, None, ''),
]
for i, (title, data, hline, hlabel) in enumerate(plots):
    ax = axes3[i // 3, i % 3]
    ax.plot(bh_iters, data, 'o', alpha=0.2, color='#1f77b4', markersize=2)
    if len(data) >= 5:
        ax.plot(bh_iters[4:], smooth(data, 5), color='#1f77b4', linewidth=2)
    if hline is not None:
        ax.axhline(y=hline, color='green', ls='--', alpha=0.5, label=hlabel)
        ax.legend(fontsize=8)
    ax.set_title(f'{title} ({data[0]:.3f} -> {data[-1]:.3f})')
    ax.set_xlabel('Iteration'); ax.grid(True, alpha=0.3)

plt.tight_layout()
p3 = os.path.join(BASE, 'v4_behavior.png')
plt.savefig(p3, dpi=150, bbox_inches='tight')
print(f'Saved: {p3}')

# ================================================================
# 终端摘要
# ================================================================
print(f'\n{"="*70}')
print(f'v4 训练分析 — 五次多项式足部轨迹跟踪')
print(f'{"="*70}')
print(f'Reward: {reward[0]:.3f} -> {reward[-1]:.3f}')
print(f'Fall Rate: 100% (全程)')
print(f'')
print(f'改善项:')
print(f'  Foot Height Track:  {r_foot_h[0]:.3f} -> {r_foot_h[-1]:.3f}  (+{(r_foot_h[-1]/r_foot_h[0]-1)*100:.0f}%)')
print(f'  Foot Vel Track:     {r_foot_v[0]:.3f} -> {r_foot_v[-1]:.3f}  (+{(r_foot_v[-1]/r_foot_v[0]-1)*100:.0f}%)')
print(f'  Torques:            {r_torques[0]:.0f} -> {r_torques[-1]:.0f}  ({(r_torques[-1]/r_torques[0]-1)*100:.0f}%)')
print(f'')
print(f'持平/恶化:')
print(f'  Survival:           {r_survival[0]:.3f} -> {r_survival[-1]:.3f}')
print(f'  Orientation:        {r_orientation[0]:.3f} -> {r_orientation[-1]:.3f}')
print(f'  Base Height:        {base_height[0]:.3f} -> {base_height[-1]:.3f} m')
print(f'')
print(f'诊断: 足部轨迹学习有进展(+119%), 但平衡从未改善。')
print(f'      固定正弦步态(0.64s周期)强迫机器人在站稳前抬脚, 导致永远摔倒。')
print(f'      建议: 用RL学习步态相位, 而非硬编码正弦波。')
