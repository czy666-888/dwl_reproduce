import re, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

# 合并所有训练数据
logs = [
    r'C:\Users\czy66\AppData\Local\Temp\claude\C--Users-czy66\745bae5a-f6b2-44f2-ae4a-4568bdb4672d\tasks\bkiik2owd.output',   # 150 iter
    r'C:\Users\czy66\AppData\Local\Temp\claude\C--Users-czy66\745bae5a-f6b2-44f2-ae4a-4568bdb4672d\tasks\bqzq0u1o9.output',   # 100 iter
]

all_iters, all_rewards, all_losses, all_sps = [], [], [], []
base_iter = 0

for log in logs:
    try:
        with open(log, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except:
        continue

    pattern = r'Iter\s+(\d+)/\d+\s+\|\s+Reward:\s+([\d.]+)\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+Steps/s:\s+([\d.]+)\s+\|\s+Time:\s+([\d.]+)'
    matches = re.findall(pattern, content)

    for m in matches:
        all_iters.append(base_iter + int(m[0]))
        all_rewards.append(float(m[1]))
        all_losses.append(float(m[2]))
        all_sps.append(float(m[3]))

    if matches:
        base_iter = all_iters[-1]

# Remove duplicates (overlapping iterations)
seen = set()
clean_iters, clean_rewards, clean_losses, clean_sps = [], [], [], []
for i, r, l, s in zip(all_iters, all_rewards, all_losses, all_sps):
    if i not in seen:
        seen.add(i)
        clean_iters.append(i)
        clean_rewards.append(r)
        clean_losses.append(l)
        clean_sps.append(s)

iters = np.array(clean_iters); rewards = np.array(clean_rewards)
losses = np.array(clean_losses); sps = np.array(clean_sps)

def smooth(data, w=10):
    return np.convolve(data, np.ones(w)/w, mode='valid')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'DWL + PPO Training (XBot-L, 256 envs, T600 4GB)\n{len(iters)} total iterations', fontsize=14, fontweight='bold')

# Loss
ax = axes[0, 0]
ax.plot(iters, losses, alpha=0.3, color='red', lw=0.8)
if len(losses)>=10:
    sl = smooth(losses, 10); ax.plot(iters[9:], sl, 'red', lw=2, label=f'Final: {losses[-1]:.1f}')
ax.set_xlabel('Iteration'); ax.set_ylabel('L_total'); ax.set_title('DWL Total Loss')
ax.legend(); ax.grid(True, alpha=0.3)

# Reward
ax = axes[0, 1]
ax.plot(iters, rewards, alpha=0.3, color='blue', lw=0.8)
if len(rewards)>=10:
    sr = smooth(rewards, 10); ax.plot(iters[9:], sr, 'blue', lw=2, label=f'Final: {rewards[-1]:.3f}')
ax.set_xlabel('Iteration'); ax.set_ylabel('Reward'); ax.set_title('Episode Reward')
ax.legend(); ax.grid(True, alpha=0.3)

# Steps/s
ax = axes[1, 0]
ax.plot(iters, sps, alpha=0.5, color='green', lw=0.8)
avg_sp = np.mean(sps[sps > 50])  # exclude sleep spikes
ax.axhline(y=avg_sp, color='darkgreen', linestyle='--', label=f'Avg: {avg_sp:.0f} steps/s')
ax.set_xlabel('Iteration'); ax.set_ylabel('Steps/s'); ax.set_title('Training Speed')
ax.legend(); ax.grid(True, alpha=0.3)

# Histogram of Loss
ax = axes[1, 1]
first_half = losses[:len(losses)//2]; second_half = losses[len(losses)//2:]
ax.hist(first_half, bins=20, alpha=0.5, label=f'First half (median={np.median(first_half):.0f})', color='red')
ax.hist(second_half, bins=20, alpha=0.5, label=f'Second half (median={np.median(second_half):.0f})', color='blue')
ax.set_xlabel('Loss'); ax.set_title('Loss Distribution Shift')
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = r'C:\Users\czy66\Desktop\dwl_reproduce\training_results.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f'Chart saved: {save_path}')

# Stats
print(f'\n=== Training Summary ===')
print(f'Total iterations: {len(iters)} (150 old + {len(iters)-150} new)')
print(f'Initial Loss: {losses[0]:.1f} -> Final Loss: {losses[-1]:.1f}')
print(f'Best Reward: {max(rewards):.4f} (iter {iters[np.argmax(rewards)]})')
print(f'Final Reward: {rewards[-1]:.4f}')
print(f'Avg Speed: {avg_sp:.0f} steps/s')
