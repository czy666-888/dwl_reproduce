"""全面可视化: 训练曲线 + 12项奖励分项 + 行为指标"""
import csv, re, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

BASE = os.path.dirname(__file__)
CSV = os.path.join(BASE, 'logs', 'dwl_run1', 'behavior_log.csv')

# ---- 1. 读取 behavior_log.csv ----
with open(CSV, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

iters = np.array([int(r['iteration']) for r in rows])
mean_reward = np.array([float(r['mean_reward']) for r in rows])
fall_pct = np.array([float(r['fall_pct']) for r in rows])
base_height = np.array([float(r['base_height']) for r in rows])
height_error = np.array([float(r['height_error']) for r in rows])

# 12 reward items
r_track_lin = np.array([float(r['r_track_lin_vel']) for r in rows])
r_track_ang = np.array([float(r['r_track_ang_vel']) for r in rows])
r_orient = np.array([float(r['r_orientation']) for r in rows])
r_base_h = np.array([float(r['r_base_height']) for r in rows])
r_air_time = np.array([float(r['r_feet_air_time']) for r in rows])
r_contact_num = np.array([float(r['r_contact_number']) for r in rows])
r_contact_f = np.array([float(r['r_contact_force']) for r in rows])
r_foot_slip = np.array([float(r['r_foot_slip']) for r in rows])
r_default_j = np.array([float(r['r_default_joint']) for r in rows])
r_torques = np.array([float(r['r_torques']) for r in rows])
r_dof_vel = np.array([float(r['r_dof_vel']) for r in rows])
r_action_s = np.array([float(r['r_action_smooth']) for r in rows])

# Behavior metrics
actual_vx = np.array([float(r['actual_vx']) for r in rows])
actual_vyaw = np.array([float(r['actual_vyaw']) for r in rows])
roll = np.array([abs(float(r['roll'])) for r in rows])
pitch = np.array([abs(float(r['pitch'])) for r in rows])
torque_mean = np.array([float(r['torque_mean']) for r in rows])
dof_vel_mean = np.array([float(r['dof_vel_mean']) for r in rows])
foot_slip_raw = np.array([float(r['foot_slip_raw']) for r in rows])

# ---- Smooth ----
def smooth(d, w=10):
    if len(d) < w: return d
    return np.convolve(d, np.ones(w)/w, mode='valid')

# ---- Color palette ----
C = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
     '#aec7e8', '#ffbb78']

fig = plt.figure(figsize=(22, 26))
gs = fig.add_gridspec(6, 3, height_ratios=[1, 1, 1, 1, 1, 0.8])
fig.suptitle('DWL + PPO Training Analysis — XBot-L, 256 envs, 3000 iters (12.4h)\n'
             'Fall Rate: 100% throughout — robot never learned to stand',
             fontsize=16, fontweight='bold', y=0.99)

# ===== Row 1: Reward + Fall Rate + Base Height =====
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(iters, mean_reward, 'o', alpha=0.3, color='#1f77b4', markersize=2)
if len(mean_reward) >= 10:
    ax1.plot(iters[9:], smooth(mean_reward, 10), color='#1f77b4', linewidth=2)
ax1.set_title(f'Mean Reward (start={mean_reward[0]:.2f}, end={mean_reward[-1]:.2f})')
ax1.set_ylabel('Reward'); ax1.grid(True, alpha=0.3)
ax1.axhline(y=np.mean(mean_reward), color='darkblue', ls='--', alpha=0.4)

# Parse console log for loss curve
log_files = [f for f in os.listdir(os.path.join(BASE, 'logs', 'dwl_run1'))
             if f.endswith('.output') or 'train' in f.lower()]
try:
    from glob import glob
    import subprocess
    # Try reading from claude task output
    pass
except: pass

# Fall rate subplot
ax2 = fig.add_subplot(gs[0, 1])
ax2.fill_between(iters, 0, fall_pct, alpha=0.3, color='red')
ax2.plot(iters, fall_pct, 'r-', linewidth=1.5)
ax2.set_title('Fall Rate (always 100%)')
ax2.set_ylabel('%'); ax2.set_ylim(0, 105); ax2.grid(True, alpha=0.3)

# Base height
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(iters, base_height, 'o', alpha=0.3, color='#2ca02c', markersize=2)
if len(base_height) >= 10:
    ax3.plot(iters[9:], smooth(base_height, 10), color='#2ca02c', linewidth=2)
ax3.axhline(y=0.89, color='green', ls='--', alpha=0.5, label='Target 0.89m')
ax3.set_title(f'Base Height (start={base_height[0]:.3f}, end={base_height[-1]:.3f})')
ax3.set_ylabel('m'); ax3.legend(fontsize=7); ax3.grid(True, alpha=0.3)

# ===== Row 2-3: 12 Reward Items =====
reward_items = [
    (r_track_lin, '1. Lin Vel Track', C[0], True),
    (r_track_ang, '2. Ang Vel Track', C[1], True),
    (r_orient, '3. Orientation', C[2], True),
    (r_base_h, '4. Base Height', C[3], True),
    (r_air_time, '5. Feet Air Time', C[4], False),
    (r_contact_num, '6. Contact Number', C[5], False),
    (r_contact_f, '7. Contact Forces', C[6], False),
    (r_foot_slip, '8. Foot Slip', C[7], False),
    (r_default_j, '9. Default Joint', C[8], False),
    (r_torques, '10. Torques (raw)', C[9], False),
    (r_dof_vel, '11. DOF Vel (raw)', C[10], False),
    (r_action_s, '12. Action Smooth', C[11], False),
]

for i, (data, label, color, improved) in enumerate(reward_items):
    row = 1 + i // 3
    col = i % 3
    ax = fig.add_subplot(gs[row, col])
    ax.plot(iters, data, 'o', alpha=0.25, color=color, markersize=2)
    if len(data) >= 10:
        ax.plot(iters[9:], smooth(data, 10), color=color, linewidth=1.8)
    ax.set_title(label, fontsize=10, color='darkgreen' if improved else 'black')
    ax.grid(True, alpha=0.3)
    delta = data[-1] - data[0]
    ax.annotate(f'{delta:+.3f}', xy=(0.98, 0.05), xycoords='axes fraction',
                ha='right', fontsize=9, color='green' if delta > 0 else 'red',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

# ===== Row 4: Behavior Metrics =====
# Velocity tracking
ax_v = fig.add_subplot(gs[4, 0])
ax_v.plot(iters, np.abs(actual_vx), 'o', alpha=0.25, color='#1f77b4', markersize=2, label='|Vx body|')
ax_v.plot(iters, np.abs(actual_vyaw), 'o', alpha=0.25, color='#ff7f0e', markersize=2, label='|Vyaw body|')
if len(iters) >= 10:
    ax_v.plot(iters[9:], smooth(np.abs(actual_vx), 10), color='#1f77b4', linewidth=1.5)
    ax_v.plot(iters[9:], smooth(np.abs(actual_vyaw), 10), color='#ff7f0e', linewidth=1.5)
ax_v.set_title('Actual Velocity Magnitude'); ax_v.legend(fontsize=7); ax_v.grid(True, alpha=0.3)

# Orientation
ax_o = fig.add_subplot(gs[4, 1])
ax_o.plot(iters, roll, 'o', alpha=0.25, color='#d62728', markersize=2, label='|Roll|')
ax_o.plot(iters, pitch, 'o', alpha=0.25, color='#9467bd', markersize=2, label='|Pitch|')
if len(iters) >= 10:
    ax_o.plot(iters[9:], smooth(roll, 10), color='#d62728', linewidth=1.5)
    ax_o.plot(iters[9:], smooth(pitch, 10), color='#9467bd', linewidth=1.5)
ax_o.set_title('Absolute Roll / Pitch'); ax_o.legend(fontsize=7); ax_o.grid(True, alpha=0.3)

# Torques + DOF vel
ax_t = fig.add_subplot(gs[4, 2])
ax_t.plot(iters, torque_mean, 'o', alpha=0.25, color='#8c564b', markersize=2, label='Torque Mean')
if len(iters) >= 10:
    ax_t.plot(iters[9:], smooth(torque_mean, 10), color='#8c564b', linewidth=1.5)
ax_t.set_title('Mean Joint Torque'); ax_t.grid(True, alpha=0.3)

# ===== Row 5: Summary Stats =====
ax_sum = fig.add_subplot(gs[5, :])
ax_sum.axis('off')
# Compute deltas
d_track_lin = ((r_track_lin[-1] - r_track_lin[0]) / abs(r_track_lin[0]) * 100) if r_track_lin[0] != 0 else 0
d_track_ang = ((r_track_ang[-1] - r_track_ang[0]) / abs(r_track_ang[0]) * 100) if r_track_ang[0] != 0 else 0
d_orient = ((r_orient[-1] - r_orient[0]) / abs(r_orient[0]) * 100) if r_orient[0] != 0 else 0
d_torques = ((r_torques[-1] - r_torques[0]) / abs(r_torques[0]) * 100) if r_torques[0] != 0 else 0

summary = (
    f"Training Summary  |  XBot-L Humanoid  |  256 envs x 24 steps  |  3000 iterations  |  12.4 hours\n"
    f"{'='*110}\n"
    f"  Mean Reward: {mean_reward[0]:.3f} -> {mean_reward[-1]:.3f} (best: {max(mean_reward):.4f})   |   "
    f"Fall Rate: 100% throughout\n"
    f"{'='*110}\n"
    f"  12 Reward Item Trends:\n"
    f"    Lin Vel Track:    {r_track_lin[0]:.3f} -> {r_track_lin[-1]:.3f}  ({d_track_lin:+.0f}%)  |  "
    f"Ang Vel Track:    {r_track_ang[0]:.3f} -> {r_track_ang[-1]:.3f}  ({d_track_ang:+.0f}%)\n"
    f"    Orientation:      {r_orient[0]:.3f} -> {r_orient[-1]:.3f}  ({d_orient:+.0f}%)  |  "
    f"Base Height:      {r_base_h[0]:.3f} -> {r_base_h[-1]:.3f}\n"
    f"    Air Time:         {r_air_time[0]:.3f} -> {r_air_time[-1]:.3f}  |  "
    f"Contact Number:   {r_contact_num[0]:.3f} -> {r_contact_num[-1]:.3f}\n"
    f"    Contact Force:    {r_contact_f[0]:.4f} -> {r_contact_f[-1]:.4f}  |  "
    f"Foot Slip:        {r_foot_slip[0]:.3f} -> {r_foot_slip[-1]:.3f}\n"
    f"    Default Joint:    {r_default_j[0]:.3f} -> {r_default_j[-1]:.3f}  |  "
    f"Torques (raw):    {r_torques[0]:.0f} -> {r_torques[-1]:.0f}  ({d_torques:+.0f}%)\n"
    f"    DOF Vel (raw):    {r_dof_vel[0]:.1f} -> {r_dof_vel[-1]:.1f}  |  "
    f"Action Smooth:    {r_action_s[0]:.3f} -> {r_action_s[-1]:.3f}\n"
    f"{'='*110}\n"
    f"  DIAGNOSIS: Torques +{d_torques:.0f}% — policy learned to thrash legs harder, not to balance.\n"
    f"  Orientation & height rewards flat → upright posture never emerged.\n"
    f"  Suggested: raise orientation weight 1.0->3.0, LR 1e-5->5e-5, lower term_height 0.45->0.35."
)
ax_sum.text(0.5, 0.5, summary, transform=ax_sum.transAxes,
            fontsize=10, fontfamily='monospace', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout(rect=[0, 0.04, 1, 0.97])
save_path = os.path.join(BASE, 'training_analysis.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f'Saved: {save_path}')
print(summary)
