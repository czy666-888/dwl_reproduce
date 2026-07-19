"""解析训练日志并生成可视化图表"""
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 最新的训练日志
log_path = r'C:\Users\czy66\AppData\Local\Temp\claude\C--Users-czy66\d5d8e2e0-fbfb-4495-8199-f1b8bd5acd45\tasks\bm7x81tmm.output'

with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

pattern = r'Iter\s+(\d+)/\d+\s+\|\s+Reward:\s+([\d.]+)\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+Steps/s:\s+([\d.]+)\s+\|\s+Time:\s+([\d.]+)'
matches = re.findall(pattern, content)

iters = np.array([int(m[0]) for m in matches])
rewards = np.array([float(m[1]) for m in matches])
losses = np.array([float(m[2]) for m in matches])
steps_per_sec = np.array([float(m[3]) for m in matches])
times = np.array([float(m[4]) for m in matches])

print(f'Parsed {len(matches)} iterations (iter {iters[0]} to {iters[-1]})')

def smooth(data, window=10):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle(f'DWL + PPO Training on XBot-L ({len(matches)} iterations, 256 envs, T600 4GB)',
             fontsize=14, fontweight='bold')

# 1. Reward
ax = axes[0, 0]
ax.plot(iters, rewards, alpha=0.25, color='blue', linewidth=0.6)
if len(rewards) >= 10:
    sr = smooth(rewards, 20)
    ax.plot(iters[19:], sr, 'blue', linewidth=2, label=f'Smoothed (w=20)')
ax.axhline(y=np.mean(rewards), color='darkblue', linestyle='--', alpha=0.5,
           label=f'Mean: {np.mean(rewards):.3f}')
ax.set_xlabel('Iteration'); ax.set_ylabel('Mean Episode Reward')
ax.set_title(f'Reward (Best: {np.max(rewards):.4f}, Final: {rewards[-1]:.4f})')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 2. Total Loss
ax = axes[0, 1]
ax.plot(iters, losses, alpha=0.25, color='red', linewidth=0.6)
if len(losses) >= 10:
    sl = smooth(losses, 20)
    ax.plot(iters[19:], sl, 'red', linewidth=2)
ax.set_xlabel('Iteration'); ax.set_ylabel('L_total')
ax.set_title(f'DWL Total Loss (Start: {losses[0]:.0f}, End: {losses[-1]:.0f})')
ax.grid(True, alpha=0.3)

# 3. Speed
ax = axes[1, 0]
ax.plot(iters, steps_per_sec, alpha=0.4, color='green', linewidth=0.6)
valid_sps = steps_per_sec[steps_per_sec > 50]
avg_sp = np.mean(valid_sps) if len(valid_sps) > 0 else np.mean(steps_per_sec)
ax.axhline(y=avg_sp, color='darkgreen', linestyle='--',
           label=f'Avg (active): {avg_sp:.0f} steps/s')
if len(valid_sps) >= 10:
    ss = smooth(steps_per_sec, 20)
    ax.plot(iters[19:], ss, 'green', linewidth=2)
ax.set_xlabel('Iteration'); ax.set_ylabel('Steps / Second')
ax.set_title('Training Speed')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 4. Cumulative Time
ax = axes[1, 1]
ax.plot(iters, times / 60, color='purple', linewidth=2)
ax.set_xlabel('Iteration'); ax.set_ylabel('Hours')
ax.set_title(f'Cumulative Time (Total: {times[-1]/60:.1f}h)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = r'C:\Users\czy66\Desktop\dwl_reproduce\training_results.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f'Chart saved to: {save_path}')

print(f'\n=== Training Summary ===')
print(f'Iterations: {iters[0]} -> {iters[-1]} ({len(matches)} total)')
print(f'Total time: {times[-1]:.1f} min ({times[-1]/60:.2f}h)')
print(f'Best Reward: {np.max(rewards):.4f} (iter {iters[np.argmax(rewards)]})')
print(f'Final Reward: {rewards[-1]:.4f}')
print(f'Mean Reward: {np.mean(rewards):.4f}')
print(f'Initial Loss: {losses[0]:.0f} -> Final Loss: {losses[-1]:.0f}')
print(f'Avg Speed (active): {avg_sp:.0f} steps/s')
