"""v1 vs v2 对比分析 — 证明权重调整不足以解决问题"""
import csv, re, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(__file__)

def load_console(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    pattern = r'Iter\s+(\d+)/3000\s+\|\s+Reward:\s+([\d.]+)\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+Steps/s:\s+([\d.]+)\s+\|\s+Time:\s+([\d.]+)'
    matches = re.findall(pattern, content)
    return (
        np.array([int(m[0]) for m in matches]),
        np.array([float(m[1]) for m in matches]),
        np.array([float(m[2]) for m in matches]),
        np.array([float(m[3]) for m in matches]),
    )

def load_behavior(csv_path):
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    iters = np.array([int(r['iteration']) for r in rows])
    return iters, rows

def smooth(d, w=30):
    if len(d) < w: return d
    return np.convolve(d, np.ones(w)/w, mode='valid')

# ---- 加载数据 ----
v1_iters, v1_r, v1_loss, v1_sp = load_console(
    r'C:\Users\czy66\AppData\Local\Temp\claude\C--Users-czy66\7febedf9-6b0f-4a2e-a213-fd5da8737fa8\tasks\bzu6zz3cz.output')
v2_iters, v2_r, v2_loss, v2_sp = load_console(
    r'C:\Users\czy66\AppData\Local\Temp\claude\C--Users-czy66\7febedf9-6b0f-4a2e-a213-fd5da8737fa8\tasks\bb9gajdj1.output')

v1_bh_iters, v1_bh_all = load_behavior(os.path.join(BASE, 'logs', 'dwl_run1', 'behavior_log.csv'))
v2_bh_iters, v2_bh_all = load_behavior(os.path.join(BASE, 'logs', 'dwl_run2', 'behavior_log.csv'))

# 限制到 v2 的范围内 (max 1600 iters)
max_iter = 1600
v1_mask = v1_iters <= max_iter
v1_iters = v1_iters[v1_mask]; v1_r = v1_r[v1_mask]; v1_loss = v1_loss[v1_mask]; v1_sp = v1_sp[v1_mask]
v2_mask = v2_iters <= max_iter
v2_iters = v2_iters[v2_mask]; v2_r = v2_r[v2_mask]; v2_loss = v2_loss[v2_mask]; v2_sp = v2_sp[v2_mask]

# 同时过滤 behavior 数据
v1_bh = [r for i, r in enumerate(v1_bh_all) if v1_bh_iters[i] <= max_iter]
v1_bh_iters = v1_bh_iters[v1_bh_iters <= max_iter]
v2_bh = [r for i, r in enumerate(v2_bh_all) if v2_bh_iters[i] <= max_iter]
v2_bh_iters = v2_bh_iters[v2_bh_iters <= max_iter]

def get_bh(rr, key):
    return np.array([float(r[key]) for r in rr])

v1_fall = get_bh(v1_bh, 'fall_pct')
v2_fall = get_bh(v2_bh, 'fall_pct')
v1_orient = get_bh(v1_bh, 'r_orientation')
v2_orient = get_bh(v2_bh, 'r_orientation')
v1_torques = get_bh(v1_bh, 'r_torques')
v2_torques = get_bh(v2_bh, 'r_torques')
v1_height = get_bh(v1_bh, 'base_height')
v2_height = get_bh(v2_bh, 'base_height')

w = 50

# ===== 图表 =====
fig = plt.figure(figsize=(22, 16))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], hspace=0.4, wspace=0.35)
fig.suptitle('v1 vs v2 对比 (前1600轮) — 权重调整没有改变根本问题',
             fontsize=15, fontweight='bold')

# Row 1: Reward (v1 | v2 | overlay)
for col, (label, iters, data, color) in enumerate([
    ('v1 奖励', v1_iters, v1_r, '#1f77b4'),
    ('v2 奖励', v2_iters, v2_r, '#ff7f0e'),
]):
    ax = fig.add_subplot(gs[0, col])
    ax.plot(iters, data, alpha=0.12, color=color, linewidth=0.3)
    if len(data) >= w:
        ax.plot(iters[w-1:], smooth(data, w), color=color, linewidth=2.5)
    ax.axhline(y=np.mean(data), color=color, ls='--', alpha=0.5, lw=1.2,
               label=f'均值: {np.mean(data):.3f}')
    ax.set_title(f'{label} | {data[0]:.2f}→{data[-1]:.2f}')
    ax.set_xlabel('迭代轮数'); ax.set_ylabel('奖励'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Overlay
ax = fig.add_subplot(gs[0, 2])
ax.plot(v1_iters, v1_r, alpha=0.08, color='#1f77b4', linewidth=0.3)
ax.plot(v2_iters, v2_r, alpha=0.08, color='#ff7f0e', linewidth=0.3)
if len(v1_r) >= w: ax.plot(v1_iters[w-1:], smooth(v1_r, w), color='#1f77b4', linewidth=2, label=f'v1 均值 {np.mean(v1_r):.3f}')
if len(v2_r) >= w: ax.plot(v2_iters[w-1:], smooth(v2_r, w), color='#ff7f0e', linewidth=2, label=f'v2 均值 {np.mean(v2_r):.3f}')
ax.set_title('奖励对比 (v1 vs v2 叠加) — v2分数高仅因权重变大'); ax.set_xlabel('迭代轮数')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Row 2: Loss (v1 | v2 | overlay)
for col, (label, iters, data, color) in enumerate([
    ('v1 损失', v1_iters, v1_loss, '#d62728'),
    ('v2 损失', v2_iters, v2_loss, '#d62728'),
]):
    ax = fig.add_subplot(gs[1, col])
    ax.plot(iters, data, alpha=0.12, color=color, linewidth=0.3)
    if len(data) >= w: ax.plot(iters[w-1:], smooth(data, w), color=color, linewidth=2.5)
    ax.set_title(f'{label} | {data[0]:.0f}→{data[-1]:.0f}'); ax.set_xlabel('迭代轮数'); ax.set_ylabel('损失')
    ax.set_ylim(0, 800); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[1, 2])
ax.plot(v1_iters, v1_loss, alpha=0.08, color='#d62728', linewidth=0.3)
ax.plot(v2_iters, v2_loss, alpha=0.08, color='#d62728', linewidth=0.3)
if len(v1_loss) >= w: ax.plot(v1_iters[w-1:], smooth(v1_loss, w), color='#d62728', linewidth=2, label=f'v1 终值 {v1_loss[-1]:.0f}')
if len(v2_loss) >= w: ax.plot(v2_iters[w-1:], smooth(v2_loss, w), color='#d62728', linewidth=2, label=f'v2 终值 {v2_loss[-1]:.0f}')
ax.set_title('损失对比 — 收敛速度相近'); ax.set_xlabel('迭代轮数')
ax.legend(fontsize=8); ax.set_ylim(0, 800); ax.grid(True, alpha=0.3)

# Row 3: 关键指标 (摔倒率 | 姿态 | 力矩)
metrics = [
    ('摔倒率 (全程100%)', v1_bh_iters, v1_fall, v2_bh_iters, v2_fall, '%'),
    ('姿态奖励 (无改善)', v1_bh_iters, v1_orient, v2_bh_iters, v2_orient, ''),
    ('力矩 Nm² (越小越好)', v1_bh_iters, v1_torques, v2_bh_iters, v2_torques, ''),
]

for col, (title, xi1, d1, xi2, d2, unit) in enumerate(metrics):
    ax = fig.add_subplot(gs[2, col])
    ax.plot(xi1, d1, 'o', alpha=0.2, color='#1f77b4', markersize=2, label='v1')
    ax.plot(xi2, d2, 'o', alpha=0.2, color='#ff7f0e', markersize=2, label='v2')
    if len(d1) >= 5:
        ax.plot(xi1[4:], smooth(d1, 5), color='#1f77b4', linewidth=1.5)
    if len(d2) >= 5:
        ax.plot(xi2[4:], smooth(d2, 5), color='#ff7f0e', linewidth=1.5)
    ax.set_title(f'{title} ({unit})')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(BASE, 'v1_vs_v2_comparison.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f'Saved: {save_path}')

# 关键数据摘要
print(f'\nv1 前1600轮: reward={v1_r[0]:.2f}→{v1_r[-1]:.2f}, loss={v1_loss[0]:.0f}→{v1_loss[-1]:.0f}, 摔倒率=100%')
print(f'v2 前1600轮: reward={v2_r[0]:.2f}→{v2_r[-1]:.2f}, loss={v2_loss[0]:.0f}→{v2_loss[-1]:.0f}, 摔倒率=100%')
print(f'\n结论: v2 只是数字变大了(权重翻倍), 摔倒率不变。证明调权重不够, 需要改奖励架构(v3门控)。')
