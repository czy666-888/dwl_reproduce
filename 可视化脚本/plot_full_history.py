import csv, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

csv_path = os.path.join(os.path.dirname(__file__), 'training_data_all_phases.csv')

iters, rewards, losses, speeds = [], [], [], []
with open(csv_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        iters.append(int(row['Iteration']))
        rewards.append(float(row['Reward']))
        losses.append(float(row['Loss']))
        speeds.append(float(row['Steps_per_sec']))

iters = np.array(iters)
rewards = np.array(rewards)
losses = np.array(losses)
speeds = np.array(speeds)

def smooth(data, window=50):
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')

fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
fig.suptitle('DWL + PPO Training — Full History (iter 1 ~ 3000)\nXBot-L, 256 envs, NVIDIA T600 4GB',
             fontsize=14, fontweight='bold')

# --- Reward ---
ax = axes[0]
ax.plot(iters, rewards, alpha=0.2, color='#1f77b4', linewidth=0.5)
sw = 50
if len(rewards) >= sw:
    ax.plot(iters[sw-1:], smooth(rewards, sw), color='#1f77b4', linewidth=2.0, label=f'Smoothed (w={sw})')
ax.axhline(y=np.mean(rewards), color='darkblue', linestyle='--', alpha=0.6, label=f'Mean = {np.mean(rewards):.3f}')
# Annotate segments
for x_start, x_end, label, y_pos in [(1, 367, 'From Scratch', 2.30), (551, 1100, 'Resume 1', 2.30), (1101, 3000, 'Resume 2', 2.30)]:
    ax.axvspan(x_start, x_end, alpha=0.06, color='gray')
    ax.text((x_start + x_end) / 2, y_pos, label, ha='center', fontsize=9, color='gray')
ax.set_ylabel('Mean Episode Reward')
ax.set_title(f'Reward  (min={min(rewards):.3f},  max={max(rewards):.3f},  start≈{rewards[0]:.3f},  end≈{rewards[-1]:.3f})')
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3)

# --- Loss ---
ax = axes[1]
ax.plot(iters, losses, alpha=0.2, color='#d62728', linewidth=0.5)
if len(losses) >= sw:
    ax.plot(iters[sw-1:], smooth(losses, sw), color='#d62728', linewidth=2.0)
ax.axhline(y=np.mean(losses), color='darkred', linestyle='--', alpha=0.6, label=f'Mean = {np.mean(losses):.0f}')
for x_start, x_end, _, _ in [(1, 367, '', 0), (551, 1100, '', 0), (1101, 3000, '', 0)]:
    ax.axvspan(x_start, x_end, alpha=0.06, color='gray')
# Highlight gap
ax.axvspan(367, 551, alpha=0.12, color='orange')
ax.text(459, ax.get_ylim()[1] * 0.95, 'Gap\n(~184 iters\nlog lost)', ha='center', fontsize=8, color='orange', fontstyle='italic')
ax.set_ylabel('L_total')
ax.set_title(f'Loss  (min={min(losses):.0f},  max={max(losses):.0f},  start={losses[0]:.0f},  end≈{losses[-1]:.0f})')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Steps/s ---
ax = axes[2]
active = np.where(speeds > 50, speeds, np.nan)
ax.plot(iters, speeds, alpha=0.2, color='#2ca02c', linewidth=0.5)
if len(speeds) >= sw:
    ax.plot(iters[sw-1:], smooth(speeds, sw), color='#2ca02c', linewidth=2.0)
avg_active = np.nanmean(active)
ax.axhline(y=avg_active, color='darkgreen', linestyle='--', alpha=0.6, label=f'Mean (active) = {avg_active:.0f} steps/s')
for x_start, x_end, _, _ in [(1, 367, '', 0), (551, 1100, '', 0), (1101, 3000, '', 0)]:
    ax.axvspan(x_start, x_end, alpha=0.06, color='gray')
ax.axvspan(367, 551, alpha=0.12, color='orange')
ax.set_xlabel('Iteration')
ax.set_ylabel('Steps / Second')
ax.set_title(f'Training Speed  (min={min(speeds):.0f},  max={max(speeds):.0f},  mean active={avg_active:.0f})')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
save_path = os.path.join(os.path.dirname(__file__), 'training_full_history.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f'Saved: {save_path}')

# Also a combined summary chart (2x2)
fig2, axes2 = plt.subplots(2, 2, figsize=(15, 11))
fig2.suptitle('DWL + PPO Training Summary — Full History (iter 1 ~ 3000)',
              fontsize=14, fontweight='bold')

# Reward
ax = axes2[0, 0]
ax.plot(iters, rewards, alpha=0.2, color='#1f77b4', linewidth=0.5)
if len(rewards) >= sw:
    ax.plot(iters[sw-1:], smooth(rewards, sw), color='#1f77b4', linewidth=2)
ax.axhline(y=np.mean(rewards), color='darkblue', linestyle='--', alpha=0.6)
ax.set_xlabel('Iteration'); ax.set_ylabel('Reward')
ax.set_title(f'Reward (mean={np.mean(rewards):.3f})')
ax.grid(True, alpha=0.3)

# Loss
ax = axes2[0, 1]
ax.plot(iters, losses, alpha=0.2, color='#d62728', linewidth=0.5)
if len(losses) >= sw:
    ax.plot(iters[sw-1:], smooth(losses, sw), color='#d62728', linewidth=2)
ax.set_xlabel('Iteration'); ax.set_ylabel('Loss')
ax.set_title(f'Loss (mean={np.mean(losses):.0f})')
ax.grid(True, alpha=0.3)

# Speed
ax = axes2[1, 0]
ax.plot(iters, speeds, alpha=0.2, color='#2ca02c', linewidth=0.5)
if len(speeds) >= sw:
    ax.plot(iters[sw-1:], smooth(speeds, sw), color='#2ca02c', linewidth=2)
ax.set_xlabel('Iteration'); ax.set_ylabel('Steps/s')
ax.set_title(f'Training Speed (mean active={avg_active:.0f})')
ax.grid(True, alpha=0.3)

# Histogram: Reward first half vs second half
ax = axes2[1, 1]
mid = len(rewards) // 2
ax.hist(rewards[:mid], bins=25, alpha=0.5, label=f'First half (median={np.median(rewards[:mid]):.3f})', color='#1f77b4')
ax.hist(rewards[mid:], bins=25, alpha=0.5, label=f'Second half (median={np.median(rewards[mid:]):.3f})', color='#ff7f0e')
ax.set_xlabel('Reward'); ax.set_ylabel('Count')
ax.set_title('Reward Distribution Shift')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
save_path2 = os.path.join(os.path.dirname(__file__), 'training_summary_chart.png')
plt.savefig(save_path2, dpi=150, bbox_inches='tight')
print(f'Saved: {save_path2}')
