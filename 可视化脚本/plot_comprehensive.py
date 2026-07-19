"""综合可视化: PPO训练曲线 + 12项奖励 + 行为指标"""
import csv, re, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(__file__)

# ---- 1. 解析控制台日志 (PPO reward/loss/speed) ----
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

# ---- 2. 读取 behavior_log.csv ----
CSV = os.path.join(BASE, 'logs', 'dwl_run1', 'behavior_log.csv')
with open(CSV, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

bh_iters = np.array([int(r['iteration']) for r in rows])
mean_reward = np.array([float(r['mean_reward']) for r in rows])
fall_pct = np.array([float(r['fall_pct']) for r in rows])
base_height = np.array([float(r['base_height']) for r in rows])

# 12 reward items
r_names = ['r_track_lin_vel', 'r_track_ang_vel', 'r_orientation',
           'r_base_height', 'r_feet_air_time', 'r_contact_number',
           'r_contact_force', 'r_foot_slip', 'r_default_joint',
           'r_torques', 'r_dof_vel', 'r_action_smooth']
r_labels = [
    '1. Lin Vel Track (↑ good)', '2. Ang Vel Track (↑ good)', '3. Orientation (↑ good)',
    '4. Base Height (↑ good)', '5. Feet Air Time (↑ good)', '6. Contact Number',
    '7. Contact Forces (↓ good)', '8. Foot Slip (↓ good)', '9. Default Joint (↑ good)',
    '10. Torques raw (↓ good)', '11. DOF Vel raw (↓ good)', '12. Action Smooth (↓ good)',
]
r_data = {n: np.array([float(r[n]) for r in rows]) for n in r_names}

# Behavior
actual_vx = np.array([float(r['actual_vx']) for r in rows])
actual_vyaw = np.array([float(r['actual_vyaw']) for r in rows])
roll = np.array([abs(float(r['roll'])) for r in rows])
pitch = np.array([abs(float(r['pitch'])) for r in rows])
torque_mean = np.array([float(r['torque_mean']) for r in rows])

# ---- Smooth ----
def smooth(d, w=10):
    if len(d) < w: return d
    return np.convolve(d, np.ones(w)/w, mode='valid')

C = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
     '#aec7e8', '#ffbb78']

# ============================================================
fig = plt.figure(figsize=(26, 30))
gs = fig.add_gridspec(7, 4, height_ratios=[1, 1, 1, 1, 1, 1, 0.6],
                      hspace=0.45, wspace=0.35)

fig.suptitle('DWL + PPO Training on XBot-L Humanoid — 3000 Iterations (12.4h)\n'
             '256 envs x 24 steps, NVIDIA T600 4GB, Fall Rate: 100% throughout',
             fontsize=17, fontweight='bold', y=0.995)

# ===== Row 1: PPO Reward | DWL Loss | Steps/s | Cumul Time =====
ax = fig.add_subplot(gs[0, 0])
ax.plot(pp_iters, pp_rewards, alpha=0.15, color='#1f77b4', linewidth=0.5)
w = 50
if len(pp_rewards) >= w:
    ax.plot(pp_iters[w-1:], smooth(pp_rewards, w), color='#1f77b4', linewidth=2)
ax.axhline(y=np.mean(pp_rewards), color='darkblue', ls='--', alpha=0.5, lw=1)
ax.set_title(f'PPO Mean Reward (start={pp_rewards[0]:.2f}, end={pp_rewards[-1]:.2f})', fontsize=11)
ax.set_ylabel('Reward'); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[0, 1])
ax.plot(pp_iters, pp_losses, alpha=0.15, color='#d62728', linewidth=0.5)
if len(pp_losses) >= w:
    ax.plot(pp_iters[w-1:], smooth(pp_losses, w), color='#d62728', linewidth=2)
ax.set_title(f'DWL Total Loss (start={pp_losses[0]:.0f}, end={pp_losses[-1]:.0f})', fontsize=11)
ax.set_ylabel('L_total'); ax.grid(True, alpha=0.3)
ax.set_ylim(0, pp_losses[0]*1.1)  # zoom to show convergence

ax = fig.add_subplot(gs[0, 2])
# Filter out sleep spikes (>500 steps/s is unrealistic for this hardware)
valid_sp = np.where(pp_speed < 600, pp_speed, np.nan)
ax.plot(pp_iters, valid_sp, alpha=0.3, color='#2ca02c', linewidth=0.5)
if len(pp_speed) >= w:
    ax.plot(pp_iters[w-1:], smooth(pp_speed, w), color='#2ca02c', linewidth=2)
avg_sp = np.nanmean(valid_sp)
ax.axhline(y=avg_sp, color='darkgreen', ls='--', alpha=0.5, lw=1,
           label=f'Mean: {avg_sp:.0f} steps/s')
ax.set_title(f'Training Speed', fontsize=11)
ax.set_ylabel('Steps / Second'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[0, 3])
hours = pp_time / 60
ax.plot(pp_iters, hours, color='purple', linewidth=1.5)
ax.fill_between(pp_iters, 0, hours, alpha=0.1, color='purple')
ax.set_title(f'Cumulative Time (total={hours[-1]:.1f}h)', fontsize=11)
ax.set_ylabel('Hours'); ax.grid(True, alpha=0.3)

# ===== Row 2: Fall Rate + Base Height + Mean Reward (behavior) =====
ax = fig.add_subplot(gs[1, 0])
ax.fill_between(bh_iters, 0, fall_pct, alpha=0.35, color='red')
ax.plot(bh_iters, fall_pct, 'r-', linewidth=1.5)
ax.set_title('Fall Rate (always 100%)', fontsize=11)
ax.set_ylabel('%'); ax.set_ylim(0, 105); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[1, 1])
ax.plot(bh_iters, base_height, 'o', alpha=0.3, color='#2ca02c', markersize=2)
if len(base_height) >= 10:
    ax.plot(bh_iters[9:], smooth(base_height, 10), color='#2ca02c', linewidth=1.5)
ax.axhline(y=0.89, color='green', ls='--', alpha=0.5, label='Target 0.89m')
ax.set_title(f'Base Height ({base_height[0]:.3f}→{base_height[-1]:.3f}m)', fontsize=11)
ax.set_ylabel('m'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[1, 2])
ax.plot(bh_iters, mean_reward, 'o', alpha=0.3, color='#1f77b4', markersize=2)
if len(mean_reward) >= 10:
    ax.plot(bh_iters[9:], smooth(mean_reward, 10), color='#1f77b4', linewidth=1.5)
ax.set_title(f'Mean Reward per 5 iters ({mean_reward[0]:.2f}→{mean_reward[-1]:.2f})', fontsize=11)
ax.set_ylabel('Reward'); ax.grid(True, alpha=0.3)

# Actual velocity
ax = fig.add_subplot(gs[1, 3])
ax.plot(bh_iters, np.abs(actual_vx), 'o', alpha=0.25, color='#1f77b4', markersize=2, label='|Vx body|')
ax.plot(bh_iters, np.abs(actual_vyaw), 'o', alpha=0.25, color='#ff7f0e', markersize=2, label='|Vyaw body|')
if len(bh_iters) >= 10:
    ax.plot(bh_iters[9:], smooth(np.abs(actual_vx), 10), color='#1f77b4', linewidth=1.2)
    ax.plot(bh_iters[9:], smooth(np.abs(actual_vyaw), 10), color='#ff7f0e', linewidth=1.2)
ax.set_title('Actual Body Velocity Magnitude', fontsize=11)
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# ===== Rows 3-5: 12 Reward Items =====
for i, name in enumerate(r_names):
    row = 2 + i // 4
    col = i % 4
    ax = fig.add_subplot(gs[row, col])
    data = r_data[name]
    label = r_labels[i]
    color = C[i]
    ax.plot(bh_iters, data, 'o', alpha=0.25, color=color, markersize=2)
    if len(data) >= 10:
        ax.plot(bh_iters[9:], smooth(data, 10), color=color, linewidth=1.8)
    ax.set_title(label, fontsize=10)
    ax.grid(True, alpha=0.3)
    delta = data[-1] - data[0]
    pct = (delta / abs(data[0]) * 100) if data[0] != 0 else 0
    color_tag = 'green' if ('↑ good' in label and delta > 0) or ('↓ good' in label and delta < 0) else 'red'
    ax.annotate(f'{data[0]:.3f} → {data[-1]:.3f} ({pct:+.0f}%)',
                xy=(0.98, 0.05), xycoords='axes fraction',
                ha='right', fontsize=8, color=color_tag,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.8))

# ===== Row 6: Behavior Metrics =====
ax = fig.add_subplot(gs[5, 0])
ax.plot(bh_iters, roll, 'o', alpha=0.25, color='#d62728', markersize=2, label='|Roll|')
ax.plot(bh_iters, pitch, 'o', alpha=0.25, color='#9467bd', markersize=2, label='|Pitch|')
if len(bh_iters) >= 10:
    ax.plot(bh_iters[9:], smooth(roll, 10), color='#d62728', linewidth=1.5)
    ax.plot(bh_iters[9:], smooth(pitch, 10), color='#9467bd', linewidth=1.5)
ax.set_title(f'Absolute Roll / Pitch'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[5, 1])
ax.plot(bh_iters, torque_mean, 'o', alpha=0.25, color='#8c564b', markersize=2)
if len(bh_iters) >= 10:
    ax.plot(bh_iters[9:], smooth(torque_mean, 10), color='#8c564b', linewidth=1.5)
ax.set_title(f'Mean Joint Torque ({torque_mean[0]:.1f}→{torque_mean[-1]:.1f} Nm)', fontsize=11)
ax.grid(True, alpha=0.3)

# Reward histogram: first vs second half
ax = fig.add_subplot(gs[5, 2])
mid = len(pp_rewards) // 2
bins = np.linspace(1.5, 2.6, 30)
ax.hist(pp_rewards[:mid], bins=bins, alpha=0.5, label=f'1st half (median={np.median(pp_rewards[:mid]):.3f})', color='#1f77b4')
ax.hist(pp_rewards[mid:], bins=bins, alpha=0.5, label=f'2nd half (median={np.median(pp_rewards[mid:]):.3f})', color='#ff7f0e')
ax.set_title('Reward Distribution Shift'); ax.set_xlabel('Reward')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Loss histogram
ax = fig.add_subplot(gs[5, 3])
ax.hist(pp_losses[:mid], bins=30, alpha=0.5, label=f'1st half', color='#d62728', range=(0, 500))
ax.hist(pp_losses[mid:], bins=30, alpha=0.5, label=f'2nd half', color='#ff7f0e', range=(0, 500))
ax.set_title('Loss Distribution Shift'); ax.set_xlabel('Loss')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ===== Row 7: Summary =====
ax = fig.add_subplot(gs[6, :])
ax.axis('off')

# Build summary text dynamically
lines = [
    f"SUMMARY  |  XBot-L Humanoid  |  256 envs x 24 steps  |  3000 iters  |  12.4h  |  Fall Rate: 100%  |  "
    f"Reward: {pp_rewards[0]:.3f} → {pp_rewards[-1]:.3f}  |  Loss: {pp_losses[0]:.0f} → {pp_losses[-1]:.0f}  |  "
    f"Avg Speed: {avg_sp:.0f} steps/s",
]
for i, name in enumerate(r_names):
    d = r_data[name]
    delta = d[-1] - d[0]
    pct = (delta / abs(d[0]) * 100) if d[0] != 0 else 0
    arrow = '↑' if pct > 0 else '↓'
    lines.append(f"  {r_labels[i]:40s} {d[0]:8.3f} → {d[-1]:8.3f}  ({pct:+.0f}%) {arrow}")

summary_text = '\n'.join(lines)
ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
        fontsize=9, fontfamily='monospace', ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.7))

plt.tight_layout(rect=[0, 0.02, 1, 0.98])
save_path = os.path.join(BASE, 'training_comprehensive.png')
plt.savefig(save_path, dpi=120, bbox_inches='tight')
print(f'Saved: {save_path}')
print(f'\nParsed {len(pp_iters)} PPO iterations + {len(bh_iters)} behavior log entries')
