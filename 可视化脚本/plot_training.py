"""解析训练日志并生成可视化图表"""
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

log_path = r'C:\Users\czy66\AppData\Local\Temp\claude\C--Users-czy66\745bae5a-f6b2-44f2-ae4a-4568bdb4672d\tasks\bkiik2owd.output'

with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# 解析每一行
pattern = r'Iter\s+(\d+)/150\s+\|\s+Reward:\s+([\d.]+)\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+Steps/s:\s+([\d.]+)\s+\|\s+Time:\s+([\d.]+)'
matches = re.findall(pattern, content)

iters = [int(m[0]) for m in matches]
rewards = [float(m[1]) for m in matches]
losses = [float(m[2]) for m in matches]
steps_per_sec = [float(m[3]) for m in matches]
times = [float(m[4]) for m in matches]

print(f'Parsed {len(matches)} iterations')

# 平滑函数
def smooth(data, window=5):
    return np.convolve(data, np.ones(window)/window, mode='valid')

# ===== 图表 =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('DWL + PPO Training on XBot-L (150 iterations, 256 envs, GPU: T600 4GB)', fontsize=14, fontweight='bold')

# 1. Total Loss
ax = axes[0, 0]
ax.plot(iters, losses, alpha=0.3, color='red', linewidth=0.8, label='Raw Loss')
if len(losses) >= 5:
    smoothed_l = smooth(losses, 10)
    ax.plot(iters[9:], smoothed_l, color='red', linewidth=2, label='Smoothed (window=10)')
ax.set_xlabel('Iteration')
ax.set_ylabel('L_total')
ax.set_title('DWL Total Loss (L_denoise + 5*L_pi + 5*L_v)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.annotate(f'Start: {losses[0]:.0f}', xy=(1, losses[0]), fontsize=9, color='darkred')
ax.annotate(f'End: {losses[-1]:.0f}', xy=(150, losses[-1]), fontsize=9, color='darkred')

# 2. Episode Reward
ax = axes[0, 1]
ax.plot(iters, rewards, alpha=0.3, color='blue', linewidth=0.8, label='Raw Reward')
if len(rewards) >= 5:
    smoothed_r = smooth(rewards, 10)
    ax.plot(iters[9:], smoothed_r, color='blue', linewidth=2, label='Smoothed (window=10)')
ax.set_xlabel('Iteration')
ax.set_ylabel('Mean Episode Reward')
ax.set_title('Training Reward')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(1.8, 2.2)

# 3. Steps per second
ax = axes[1, 0]
ax.plot(iters, steps_per_sec, alpha=0.5, color='green', linewidth=0.8)
if len(steps_per_sec) >= 5:
    smoothed_sp = smooth(steps_per_sec, 10)
    ax.plot(iters[9:], smoothed_sp, color='green', linewidth=2)
avg_sp = np.mean(steps_per_sec)
ax.axhline(y=avg_sp, color='darkgreen', linestyle='--', label=f'Average: {avg_sp:.0f} steps/s')
ax.set_xlabel('Iteration')
ax.set_ylabel('Steps / Second')
ax.set_title('Training Speed')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Cumulative time
ax = axes[1, 1]
cum_time = np.cumsum([0] + [24.5]*len(iters))[:len(iters)]  # approximate
ax.plot(iters, times, color='purple', linewidth=2)
ax.set_xlabel('Iteration')
ax.set_ylabel('Cumulative Time (min)')
ax.set_title('Training Progress')
ax.grid(True, alpha=0.3)
ax.annotate(f'Total: {times[-1]:.1f} min ({times[-1]/60:.1f}h)',
            xy=(150, times[-1]), fontsize=10, color='purple')

plt.tight_layout()
save_path = r'C:\Users\czy66\Desktop\dwl_reproduce\training_results.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f'Chart saved to: {save_path}')

# 打印关键统计
print(f'\n=== Training Summary ===')
print(f'Iterations: {len(iters)}/150 completed')
print(f'Total time: {times[-1]:.1f} min ({times[-1]/60:.2f}h)')
print(f'Initial Loss: {losses[0]:.1f} -> Final Loss: {losses[-1]:.1f} ({(1-losses[-1]/losses[0])*100:.1f}% reduction)')
print(f'Best Reward: {max(rewards):.3f}')
print(f'Final Reward: {rewards[-1]:.3f}')
print(f'Average Speed: {avg_sp:.0f} steps/s')
print(f'Average Time/Iter: {times[-1]/len(iters)*60:.1f}s')
