"""分图可视化：PPO训练趋势 / 12项奖励 / 行为指标"""
import csv, re, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(__file__)

# ---- 解析控制台日志 ----
output_path = r'C:\Users\czy66\AppData\Local\Temp\claude\C--Users-czy66\7febedf9-6b0f-4a2e-a213-fd5da8737fa8\tasks\bzu6zz3cz.output'
with open(output_path, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

pattern = r'Iter\s+(\d+)/3000\s+\|\s+Reward:\s+([\d.]+)\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+Steps/s:\s+([\d.]+)\s+\|\s+Time:\s+([\d.]+)'
matches = re.findall(pattern, content)
pp_iters = np.array([int(m[0]) for m in matches])
pp_rewards = np.array([float(m[1]) for m in matches])
pp_losses = np.array([float(m[2]) for m in matches])
pp_speed = np.array([float(m[3]) for m in matches])
pp_time = np.array([float(m[4]) for m in matches])

# ---- 读取 behavior_log ----
CSV = os.path.join(BASE, 'logs', 'dwl_run1', 'behavior_log.csv')
with open(CSV, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

bh_iters = np.array([int(r['iteration']) for r in rows])

r_names = ['r_track_lin_vel', 'r_track_ang_vel', 'r_orientation',
           'r_base_height', 'r_feet_air_time', 'r_contact_number',
           'r_contact_force', 'r_foot_slip', 'r_default_joint',
           'r_torques', 'r_dof_vel', 'r_action_smooth']
r_labels_short = [
    'Lin Vel Track', 'Ang Vel Track', 'Orientation', 'Base Height',
    'Feet Air Time', 'Contact Number', 'Contact Forces', 'Foot Slip',
    'Default Joint', 'Torques', 'DOF Vel', 'Action Smooth',
]
r_data = {n: np.array([float(r[n]) for r in rows]) for n in r_names}

# Behavior keys
base_height = np.array([float(r['base_height']) for r in rows])
roll_abs = np.array([abs(float(r['roll'])) for r in rows])
pitch_abs = np.array([abs(float(r['pitch'])) for r in rows])
torque_mean = np.array([float(r['torque_mean']) for r in rows])
dof_vel_mean = np.array([float(r['dof_vel_mean']) for r in rows])
actual_vx = np.array([abs(float(r['actual_vx'])) for r in rows])
actual_vyaw = np.array([abs(float(r['actual_vyaw'])) for r in rows])
foot_slip = np.array([float(r['foot_slip_raw']) for r in rows])
contact_left = np.array([float(r['contact_left']) for r in rows])
contact_right = np.array([float(r['contact_right']) for r in rows])
fall_pct = np.array([float(r['fall_pct']) for r in rows])
mean_reward = np.array([float(r['mean_reward']) for r in rows])

def smooth(d, w=30):
    if len(d) < w: return d
    return np.convolve(d, np.ones(w)/w, mode='valid')

C = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
     '#aec7e8', '#ffbb78']
w = 50

# ===================================================================
# 图 1: PPO 训练曲线 (Reward / Loss / Speed / Cumulative Time)
# ===================================================================
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 11))
fig1.suptitle('PPO Training Curves — XBot-L, 256 envs, 3000 iters, 12.4h',
              fontsize=14, fontweight='bold')

ax = axes1[0, 0]
ax.plot(pp_iters, pp_rewards, alpha=0.12, color='#1f77b4', linewidth=0.3)
if len(pp_rewards) >= w:
    ax.plot(pp_iters[w-1:], smooth(pp_rewards, w), color='#1f77b4', linewidth=2.5)
ax.axhline(y=np.mean(pp_rewards), color='darkblue', ls='--', alpha=0.5, lw=1.2,
           label=f'Mean: {np.mean(pp_rewards):.3f}')
ax.set_title(f'Episode Reward  ({pp_rewards[0]:.2f} → {pp_rewards[-1]:.2f},  Best: {max(pp_rewards):.3f})')
ax.set_xlabel('Iteration'); ax.set_ylabel('Reward')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes1[0, 1]
ax.plot(pp_iters, pp_losses, alpha=0.12, color='#d62728', linewidth=0.3)
if len(pp_losses) >= w:
    ax.plot(pp_iters[w-1:], smooth(pp_losses, w), color='#d62728', linewidth=2.5)
ax.set_title(f'DWL Total Loss  ({pp_losses[0]:.0f} → {pp_losses[-1]:.0f})')
ax.set_xlabel('Iteration'); ax.set_ylabel('L_total')
ax.set_ylim(0, 600)
ax.grid(True, alpha=0.3)

ax = axes1[1, 0]
valid = np.where(pp_speed < 600, pp_speed, np.nan)
ax.plot(pp_iters, valid, alpha=0.2, color='#2ca02c', linewidth=0.3)
if len(pp_speed) >= w:
    ax.plot(pp_iters[w-1:], smooth(pp_speed, w), color='#2ca02c', linewidth=2.5)
avg_sp = np.nanmean(valid)
ax.axhline(y=avg_sp, color='darkgreen', ls='--', alpha=0.5, lw=1.2,
           label=f'Mean active: {avg_sp:.0f} steps/s')
ax.set_title(f'Training Speed')
ax.set_xlabel('Iteration'); ax.set_ylabel('Steps / Second')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes1[1, 1]
hours = pp_time / 60
ax.plot(pp_iters, hours, color='purple', linewidth=2)
ax.fill_between(pp_iters, 0, hours, alpha=0.08, color='purple')
ax.set_title(f'Cumulative Time  ({hours[-1]:.1f}h total)')
ax.set_xlabel('Iteration'); ax.set_ylabel('Hours')
ax.grid(True, alpha=0.3)

plt.tight_layout()
p1 = os.path.join(BASE, '01_ppo_training_curves.png')
plt.savefig(p1, dpi=150, bbox_inches='tight')
print(f'Saved: {p1}')

# ===================================================================
# 图 2: 12 项奖励分项
# ===================================================================
fig2, axes2 = plt.subplots(4, 3, figsize=(20, 18))
fig2.suptitle('12 Reward Item Breakdown — 5-iteration Rolling Means',
              fontsize=14, fontweight='bold')

for i, (name, label) in enumerate(zip(r_names, r_labels_short)):
    ax = axes2[i // 3, i % 3]
    data = r_data[name]
    ax.plot(bh_iters, data, 'o', alpha=0.2, color=C[i], markersize=2)
    if len(data) >= 5:
        ax.plot(bh_iters[4:], smooth(data, 5), color=C[i], linewidth=2)
    ax.set_title(f'{i+1}. {label}', fontsize=12)
    ax.set_xlabel('Iteration'); ax.grid(True, alpha=0.3)
    delta = data[-1] - data[0]
    pct = (delta / abs(data[0]) * 100) if abs(data[0]) > 0.001 else 0
    ax.annotate(f'{data[0]:.3f} → {data[-1]:.3f}\n({pct:+.0f}%)', xy=(0.02, 0.96),
                xycoords='axes fraction', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
p2 = os.path.join(BASE, '02_reward_items.png')
plt.savefig(p2, dpi=150, bbox_inches='tight')
print(f'Saved: {p2}')

# ===================================================================
# 图 3: 行为指标 (基座 / 姿态 / 力矩 / 足部)
# ===================================================================
fig3, axes3 = plt.subplots(3, 3, figsize=(18, 15))
fig3.suptitle('Behavior Metrics — 5-iteration Rolling Means',
              fontsize=14, fontweight='bold')

# 3.1 Base height
ax = axes3[0, 0]
ax.plot(bh_iters, base_height, 'o', alpha=0.2, color='#2ca02c', markersize=2)
if len(base_height) >= 5:
    ax.plot(bh_iters[4:], smooth(base_height, 5), color='#2ca02c', linewidth=2)
ax.axhline(y=0.89, color='green', ls='--', alpha=0.5, label='Target 0.89m')
ax.set_title(f'Base Height ({base_height[0]:.3f} → {base_height[-1]:.3f}m)')
ax.set_xlabel('Iteration'); ax.set_ylabel('m')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 3.2 Fall rate
ax = axes3[0, 1]
ax.fill_between(bh_iters, 0, fall_pct, alpha=0.3, color='red')
ax.plot(bh_iters, fall_pct, 'r-', linewidth=1.5)
ax.set_title('Fall Rate (100% throughout)')
ax.set_xlabel('Iteration'); ax.set_ylabel('%')
ax.set_ylim(0, 105); ax.grid(True, alpha=0.3)

# 3.3 Mean reward (behavior)
ax = axes3[0, 2]
ax.plot(bh_iters, mean_reward, 'o', alpha=0.2, color='#1f77b4', markersize=2)
if len(mean_reward) >= 5:
    ax.plot(bh_iters[4:], smooth(mean_reward, 5), color='#1f77b4', linewidth=2)
ax.set_title(f'Mean Reward ({mean_reward[0]:.2f} → {mean_reward[-1]:.2f})')
ax.set_xlabel('Iteration'); ax.set_ylabel('Reward')
ax.grid(True, alpha=0.3)

# 3.4 Roll / Pitch
ax = axes3[1, 0]
ax.plot(bh_iters, roll_abs, 'o', alpha=0.2, color='#d62728', markersize=2, label='|Roll|')
ax.plot(bh_iters, pitch_abs, 'o', alpha=0.2, color='#9467bd', markersize=2, label='|Pitch|')
if len(bh_iters) >= 5:
    ax.plot(bh_iters[4:], smooth(roll_abs, 5), color='#d62728', linewidth=1.5)
    ax.plot(bh_iters[4:], smooth(pitch_abs, 5), color='#9467bd', linewidth=1.5)
ax.set_title('Body Orientation (|Roll|, |Pitch|)')
ax.set_xlabel('Iteration'); ax.set_ylabel('rad')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 3.5 Torque mean
ax = axes3[1, 1]
ax.plot(bh_iters, torque_mean, 'o', alpha=0.2, color='#8c564b', markersize=2)
if len(bh_iters) >= 5:
    ax.plot(bh_iters[4:], smooth(torque_mean, 5), color='#8c564b', linewidth=2)
ax.set_title(f'Mean Joint Torque ({torque_mean[0]:.1f} → {torque_mean[-1]:.1f} Nm)')
ax.set_xlabel('Iteration'); ax.set_ylabel('Nm')
ax.grid(True, alpha=0.3)

# 3.6 DOF vel mean
ax = axes3[1, 2]
ax.plot(bh_iters, dof_vel_mean, 'o', alpha=0.2, color='#17becf', markersize=2)
if len(bh_iters) >= 5:
    ax.plot(bh_iters[4:], smooth(dof_vel_mean, 5), color='#17becf', linewidth=2)
ax.set_title(f'DOF Velocity Mean ({dof_vel_mean[0]:.1f} → {dof_vel_mean[-1]:.1f} rad/s)')
ax.set_xlabel('Iteration'); ax.set_ylabel('rad/s')
ax.grid(True, alpha=0.3)

# 3.7 Velocity tracking
ax = axes3[2, 0]
ax.plot(bh_iters, actual_vx, 'o', alpha=0.2, color='#1f77b4', markersize=2, label='|Vx body|')
ax.plot(bh_iters, actual_vyaw, 'o', alpha=0.2, color='#ff7f0e', markersize=2, label='|Vyaw body|')
if len(bh_iters) >= 5:
    ax.plot(bh_iters[4:], smooth(actual_vx, 5), color='#1f77b4', linewidth=1.5)
    ax.plot(bh_iters[4:], smooth(actual_vyaw, 5), color='#ff7f0e', linewidth=1.5)
ax.set_title('Actual Body Velocity')
ax.set_xlabel('Iteration'); ax.set_ylabel('m/s or rad/s')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 3.8 Foot slip
ax = axes3[2, 1]
ax.plot(bh_iters, foot_slip, 'o', alpha=0.2, color='#e377c2', markersize=2)
if len(bh_iters) >= 5:
    ax.plot(bh_iters[4:], smooth(foot_slip, 5), color='#e377c2', linewidth=2)
ax.set_title(f'Foot Slip ({foot_slip[0]:.2f} → {foot_slip[-1]:.2f})')
ax.set_xlabel('Iteration'); ax.set_ylabel('m/s')
ax.grid(True, alpha=0.3)

# 3.9 Contact forces
ax = axes3[2, 2]
ax.plot(bh_iters, np.abs(contact_left), 'o', alpha=0.2, color='#ff7f0e', markersize=2, label='|Left|')
ax.plot(bh_iters, np.abs(contact_right), 'o', alpha=0.2, color='#1f77b4', markersize=2, label='|Right|')
if len(bh_iters) >= 5:
    ax.plot(bh_iters[4:], smooth(np.abs(contact_left), 5), color='#ff7f0e', linewidth=1.5)
    ax.plot(bh_iters[4:], smooth(np.abs(contact_right), 5), color='#1f77b4', linewidth=1.5)
ax.set_title('Foot Contact Forces')
ax.set_xlabel('Iteration'); ax.set_ylabel('N (Z-axis)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
p3 = os.path.join(BASE, '03_behavior_metrics.png')
plt.savefig(p3, dpi=150, bbox_inches='tight')
print(f'Saved: {p3}')

# =========== 终端摘要 ===========
print(f'\n{"="*80}')
print(f'Parsed: {len(pp_iters)} PPO iterations + {len(bh_iters)} behavior log entries')
print(f'Reward: {pp_rewards[0]:.3f} → {pp_rewards[-1]:.3f}  |  Best: {max(pp_rewards):.4f}')
print(f'Loss:   {pp_losses[0]:.0f} → {pp_losses[-1]:.0f}  |  Speed: {avg_sp:.0f} steps/s avg')
print(f'Time:   {pp_time[-1]/60:.1f}h  |  Fall rate: 100%')
print(f'\n12 Reward Items:')
for name, label in zip(r_names, r_labels_short):
    d = r_data[name]
    delta = d[-1] - d[0]
    pct = (delta / abs(d[0]) * 100) if abs(d[0]) > 0.001 else 0
    arrow = '↑' if delta > 0 else '↓'
    print(f'  {label:20s}  {d[0]:8.3f} → {d[-1]:8.3f}  ({pct:+.0f}%) {arrow}')
