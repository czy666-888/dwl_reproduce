"""dwl_reproduce2 综合分析 — 训练曲线 + 各Run行为数据 + 跨Run对比"""
import csv, os, re, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(__file__)
LOGS = os.path.join(BASE, 'logs')
TRAIN_LOG = os.path.join(BASE, 'train_log.txt')

RUNS = ['dwl_run1', 'dwl_run2', 'dwl_run3', 'dwl_run5', 'dwl_run6', 'dwl_run7']

def smooth(d, w=10):
    if len(d) < w:
        return d
    return np.convolve(d, np.ones(w) / w, mode='valid')

# ================================================================
# PART 1: 解析 train_log.txt
# ================================================================
print("=" * 60)
print("Part 1: Parsing train_log.txt...")

with open(TRAIN_LOG, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

pattern = r'Iter\s+(\d+)/\d+\s+\|\s+Reward:\s+([\d.]+)\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+Steps/s:\s+([\d.]+)\s+\|\s+Time:\s+([\d.]+)'
matches = re.findall(pattern, content)

t_iters = np.array([int(m[0]) for m in matches])
t_rewards = np.array([float(m[1]) for m in matches])
t_losses = np.array([float(m[2]) for m in matches])
t_sps = np.array([float(m[3]) for m in matches])
t_times = np.array([float(m[4]) for m in matches])

# Remove duplicate iterations
seen = set()
clean_i, clean_r, clean_l, clean_s, clean_t = [], [], [], [], []
for i, r, l, s, t in zip(t_iters, t_rewards, t_losses, t_sps, t_times):
    if i not in seen:
        seen.add(i)
        clean_i.append(i); clean_r.append(r); clean_l.append(l); clean_s.append(s); clean_t.append(t)

t_iters = np.array(clean_i); t_rewards = np.array(clean_r); t_losses = np.array(clean_l)
t_sps = np.array(clean_s); t_times = np.array(clean_t)

print(f"  Parsed {len(t_iters)} iterations (iter {t_iters[0]} to {t_iters[-1]})")
print(f"  Reward: {t_rewards[0]:.3f} -> {t_rewards[-1]:.3f}")
print(f"  Loss: {t_losses[0]:.0f} -> {t_losses[-1]:.1f}")
print(f"  Total time: {t_times[-1]/60:.1f} hours")

# ================================================================
# PART 2: 加载所有Run的Behavior Log
# ================================================================
print("\n" + "=" * 60)
print("Part 2: Loading behavior logs...")

# Column name mappings: different runs use different reward column names
# run1/2: r_track_lin_vel, r_track_ang_vel, r_torques, r_dof_vel, r_feet_air_time, r_contact_number
# run3: r_survival, r_track_lin_vel, r_track_ang_vel, r_torques, r_dof_vel, etc.
# run5/6/7: r_lin_vel, r_ang_vel, r_energy, r_periodic_contact, r_periodic_vel
COLUMN_MAP = {
    'r_lin_vel': ['r_lin_vel', 'r_track_lin_vel'],
    'r_ang_vel': ['r_ang_vel', 'r_track_ang_vel'],
    'r_orientation': ['r_orientation'],
    'r_height': ['r_height', 'r_base_height'],
    'r_periodic_contact': ['r_periodic_contact', 'r_contact_number'],
    'r_periodic_vel': ['r_periodic_vel', 'r_contact_force'],
    'r_foot_height': ['r_foot_height', 'r_feet_air_time'],
    'r_foot_vel': ['r_foot_vel', 'r_foot_slip'],
    'r_default_joint': ['r_default_joint'],
    'r_energy': ['r_energy', 'r_torques'],
    'r_action_smooth': ['r_action_smooth'],
}

def _col(columns, key):
    """Find which column name exists in the CSV for a given logical key."""
    if key in COLUMN_MAP:
        for candidate in COLUMN_MAP[key]:
            if candidate in columns:
                return candidate
    return key  # fallback

def _get(rows, columns, key):
    return np.array([float(r[_col(columns, key)]) for r in rows])

run_data = {}
for run in RUNS:
    csv_path = os.path.join(LOGS, run, 'behavior_log.csv')
    if not os.path.exists(csv_path):
        print(f"  SKIP {run}: no behavior_log.csv")
        continue
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames if reader.fieldnames else list(rows[0].keys())

    bh_iters = np.array([int(r['iteration']) for r in rows])
    d = {
        'iters': bh_iters,
        'n_rows': len(rows),
        'columns': columns,
        'mean_reward': np.array([float(r['mean_reward']) for r in rows]),
        'fall_pct': np.array([float(r['fall_pct']) for r in rows]),
        'base_height': np.array([float(r['base_height']) for r in rows]),
        'roll': np.abs(np.array([float(r['roll']) for r in rows])),
        'pitch': np.abs(np.array([float(r['pitch']) for r in rows])),
        'torque_mean': np.array([float(r['torque_mean']) for r in rows]),
        'dof_vel_mean': np.array([float(r['dof_vel_mean']) for r in rows]),
        'foot_slip': np.array([float(r['foot_slip_raw']) for r in rows]),
    }
    # Load reward items via column mapping
    for logical_key in ['r_lin_vel', 'r_ang_vel', 'r_orientation', 'r_height',
                         'r_periodic_contact', 'r_periodic_vel', 'r_foot_height',
                         'r_foot_vel', 'r_default_joint', 'r_energy', 'r_action_smooth']:
        try:
            d[logical_key] = _get(rows, columns, logical_key)
        except (KeyError, ValueError):
            d[logical_key] = np.zeros(len(rows))  # fill with zeros if not available

    # Contacts
    try:
        d['contact_left'] = np.array([float(r['contact_left']) for r in rows])
        d['contact_right'] = np.array([float(r['contact_right']) for r in rows])
    except KeyError:
        d['contact_left'] = np.zeros(len(rows))
        d['contact_right'] = np.zeros(len(rows))

    run_data[run] = d
    r0 = d['mean_reward'][0]
    r1 = d['mean_reward'][-1]
    print(f"  {run}: {len(rows)} rows, iter {bh_iters[0]}-{bh_iters[-1]}, "
          f"reward {r0:.3f}->{r1:.3f}")

# ================================================================
# CHART 1: 训练总览 (train_log.txt)
# ================================================================
print("\n" + "=" * 60)
print("Generating charts...")

w = 30
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
fig1.suptitle('DWL+PPO Training Overview — dwl_reproduce2 (train_log.txt)\nXBot-L, 256 envs, GPU: T600 4GB',
              fontsize=14, fontweight='bold')

ax = axes1[0, 0]
ax.plot(t_iters, t_rewards, 'o', alpha=0.15, color='#1f77b4', markersize=1.5)
if len(t_rewards) >= w:
    ax.plot(t_iters[w-1:], smooth(t_rewards, w), color='#1f77b4', linewidth=2.5)
ax.axhline(y=np.mean(t_rewards), color='darkblue', ls='--', alpha=0.5, label=f'Mean: {np.mean(t_rewards):.3f}')
ax.set_title(f'Mean Episode Reward ({t_rewards[0]:.3f} -> {t_rewards[-1]:.3f})')
ax.set_xlabel('Iteration'); ax.set_ylabel('Reward'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes1[0, 1]
ax.plot(t_iters, t_losses, 'o', alpha=0.15, color='#d62728', markersize=1.5)
if len(t_losses) >= w:
    ax.plot(t_iters[w-1:], smooth(t_losses, w), color='#d62728', linewidth=2)
ax.axhline(y=np.mean(t_losses), color='darkred', ls='--', alpha=0.5, label=f'Mean: {np.mean(t_losses):.0f}')
ax.set_title(f'DWL Total Loss ({t_losses[0]:.0f} -> {t_losses[-1]:.1f})')
ax.set_xlabel('Iteration'); ax.set_ylabel('Loss'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes1[1, 0]
active_sps = t_sps[t_sps > 50]
ax.plot(t_iters, t_sps, alpha=0.3, color='#2ca02c', linewidth=0.6)
avg_sp = np.mean(active_sps) if len(active_sps) > 0 else np.mean(t_sps)
ax.axhline(y=avg_sp, color='darkgreen', ls='--', label=f'Avg (active): {avg_sp:.0f} steps/s')
ax.set_title(f'Training Speed (avg active={avg_sp:.0f} steps/s)')
ax.set_xlabel('Iteration'); ax.set_ylabel('Steps/s'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes1[1, 1]
ax.plot(t_iters, t_times / 60, color='#9467bd', linewidth=2)
ax.set_title(f'Cumulative Time (Total: {t_times[-1]/60:.1f}h = {t_times[-1]/3600:.1f}d)')
ax.set_xlabel('Iteration'); ax.set_ylabel('Hours'); ax.grid(True, alpha=0.3)

plt.tight_layout()
p1 = os.path.join(LOGS, '01_training_overview.png')
fig1.savefig(p1, dpi=150, bbox_inches='tight')
plt.close(fig1)
print(f"  Saved: {p1}")

# ================================================================
# CHART 2: 训练指标分阶段分析
# ================================================================
fig2, axes2 = plt.subplots(3, 1, figsize=(18, 14), sharex=True)
fig2.suptitle('DWL+PPO Training — Full History with Phase Analysis\nXBot-L, 256 envs, T600 4GB',
              fontsize=14, fontweight='bold')

sw = 50
# Reward
ax = axes2[0]
ax.plot(t_iters, t_rewards, alpha=0.15, color='#1f77b4', linewidth=0.5)
if len(t_rewards) >= sw:
    ax.plot(t_iters[sw-1:], smooth(t_rewards, sw), color='#1f77b4', linewidth=2.5)
ax.axhline(y=np.mean(t_rewards), color='darkblue', ls='--', alpha=0.5, label=f'Mean: {np.mean(t_rewards):.3f}')
# Segment phases
mid = len(t_rewards) // 2
ax.axvspan(t_iters[0], t_iters[mid], alpha=0.05, color='gray')
ax.axvspan(t_iters[mid], t_iters[-1], alpha=0.05, color='orange')
ax.text((t_iters[0] + t_iters[mid]) / 2, ax.get_ylim()[1] * 0.92, f'Early (mean={np.mean(t_rewards[:mid]):.3f})',
        ha='center', fontsize=10, color='gray')
ax.text((t_iters[mid] + t_iters[-1]) / 2, ax.get_ylim()[1] * 0.92, f'Late (mean={np.mean(t_rewards[mid:]):.3f})',
        ha='center', fontsize=10, color='orange')
ax.set_ylabel('Mean Reward'); ax.set_title(f'Episode Reward (best={np.max(t_rewards):.4f} @ iter {t_iters[np.argmax(t_rewards)]})')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Loss
ax = axes2[1]
ax.plot(t_iters, t_losses, alpha=0.15, color='#d62728', linewidth=0.5)
if len(t_losses) >= sw:
    ax.plot(t_iters[sw-1:], smooth(t_losses, sw), color='#d62728', linewidth=2)
ax.axhline(y=np.mean(t_losses), color='darkred', ls='--', alpha=0.5, label=f'Mean: {np.mean(t_losses):.0f}')
ax.axvspan(t_iters[0], t_iters[mid], alpha=0.05, color='gray')
ax.axvspan(t_iters[mid], t_iters[-1], alpha=0.05, color='orange')
ax.set_ylabel('L_total'); ax.set_title(f'DWL Total Loss'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Speed
ax = axes2[2]
ax.plot(t_iters, t_sps, alpha=0.15, color='#2ca02c', linewidth=0.5)
if len(t_sps) >= sw:
    ax.plot(t_iters[sw-1:], smooth(t_sps, sw), color='#2ca02c', linewidth=2)
ax.axhline(y=avg_sp, color='darkgreen', ls='--', label=f'Mean active: {avg_sp:.0f} steps/s')
ax.axvspan(t_iters[0], t_iters[mid], alpha=0.05, color='gray')
ax.axvspan(t_iters[mid], t_iters[-1], alpha=0.05, color='orange')
ax.set_xlabel('Iteration'); ax.set_ylabel('Steps/s')
ax.set_title(f'Training Speed'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
p2 = os.path.join(LOGS, '02_training_phases.png')
fig2.savefig(p2, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"  Saved: {p2}")

# ================================================================
# CHART 3: 每个Run的Reward + Fall Rate 对比
# ================================================================
fig3, axes3 = plt.subplots(2, 3, figsize=(20, 13))
fig3.suptitle('All Runs — Reward & Fall Rate Comparison\n(dwl_run1~7, XBot-L, 256 envs)',
              fontsize=14, fontweight='bold')
axes3 = axes3.flatten()

colors_run = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
for idx, run in enumerate(RUNS):
    if run not in run_data:
        continue
    d = run_data[run]
    ax = axes3[idx]
    # Reward (left y)
    ax.plot(d['iters'], d['mean_reward'], 'o', alpha=0.2, color=colors_run[idx], markersize=2)
    if len(d['mean_reward']) >= 5:
        ax.plot(d['iters'][4:], smooth(d['mean_reward'], 5), color=colors_run[idx], linewidth=2)
    ax.set_title(f'{run} (n={d["n_rows"]})', fontweight='bold')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Mean Reward', color=colors_run[idx])
    ax.tick_params(axis='y', labelcolor=colors_run[idx])
    ax.grid(True, alpha=0.3)
    # Fall rate (right y)
    ax2 = ax.twinx()
    ax2.fill_between(d['iters'], 0, d['fall_pct'], alpha=0.15, color='red')
    ax2.plot(d['iters'], d['fall_pct'], 'r-', linewidth=1, alpha=0.6)
    ax2.set_ylabel('Fall %', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 105)

plt.tight_layout()
p3 = os.path.join(LOGS, '03_all_runs_reward_fall.png')
fig3.savefig(p3, dpi=150, bbox_inches='tight')
plt.close(fig3)
print(f"  Saved: {p3}")

# ================================================================
# CHART 4: 最新Run (dwl_run7) 详细分析
# ================================================================
latest = 'dwl_run7'
if latest in run_data:
    d = run_data[latest]
    bi = d['iters']

    # 4A: Reward items breakdown
    reward_items = [
        ('Lin Vel', d['r_lin_vel'], '#1f77b4'),
        ('Ang Vel', d['r_ang_vel'], '#ff7f0e'),
        ('Orientation', d['r_orientation'], '#2ca02c'),
        ('Height', d['r_height'], '#d62728'),
        ('Periodic Contact', d['r_periodic_contact'], '#9467bd'),
        ('Periodic Vel', d['r_periodic_vel'], '#8c564b'),
        ('Foot Height', d['r_foot_height'], '#e377c2'),
        ('Foot Vel', d['r_foot_vel'], '#7f7f7f'),
        ('Default Joint', d['r_default_joint'], '#bcbd22'),
        ('Energy', d['r_energy'], '#17becf'),
        ('Action Smooth', d['r_action_smooth'], '#ff9896'),
    ]

    fig4a, axes4a = plt.subplots(4, 3, figsize=(20, 18))
    fig4a.suptitle(f'{latest} — Reward Items Breakdown', fontsize=14, fontweight='bold')

    for i, (label, data, color) in enumerate(reward_items):
        ax = axes4a[i // 3, i % 3]
        ax.plot(bi, data, 'o', alpha=0.2, color=color, markersize=2)
        if len(data) >= 5:
            ax.plot(bi[4:], smooth(data, 5), color=color, linewidth=2)
        delta = data[-1] - data[0]
        pct = (delta / abs(data[0]) * 100) if abs(data[0]) > 0.001 else 0
        ax.set_title(f'{label}  ({data[0]:.3f} -> {data[-1]:.3f}, {pct:+.0f}%)')
        ax.set_xlabel('Iteration'); ax.grid(True, alpha=0.3)

    # Hide extra subplot
    if len(reward_items) < 12:
        for j in range(len(reward_items), 12):
            axes4a[j // 3, j % 3].set_visible(False)

    plt.tight_layout()
    p4a = os.path.join(LOGS, '04a_run7_reward_items.png')
    fig4a.savefig(p4a, dpi=150, bbox_inches='tight')
    plt.close(fig4a)
    print(f"  Saved: {p4a}")

    # 4B: Behavior metrics
    fig4b, axes4b = plt.subplots(2, 3, figsize=(18, 12))
    fig4b.suptitle(f'{latest} — Behavior Metrics', fontsize=14, fontweight='bold')

    behavior_plots = [
        ('Base Height (m)', d['base_height'], 0.89),
        ('|Roll| (rad)', d['roll'], None),
        ('|Pitch| (rad)', d['pitch'], None),
        ('Mean Torque (Nm)', d['torque_mean'], None),
        ('DOF Vel Mean (rad/s)', d['dof_vel_mean'], None),
        ('Foot Slip', d['foot_slip'], None),
    ]
    for i, (title, data, target) in enumerate(behavior_plots):
        ax = axes4b[i // 3, i % 3]
        ax.plot(bi, data, 'o', alpha=0.2, color='#1f77b4', markersize=2)
        if len(data) >= 5:
            ax.plot(bi[4:], smooth(data, 5), color='#1f77b4', linewidth=2)
        if target is not None:
            ax.axhline(y=target, color='green', ls='--', alpha=0.5, label=f'Target {target}m')
            ax.legend(fontsize=8)
        ax.set_title(f'{title} ({data[0]:.3f} -> {data[-1]:.3f})')
        ax.set_xlabel('Iteration'); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p4b = os.path.join(LOGS, '04b_run7_behavior.png')
    fig4b.savefig(p4b, dpi=150, bbox_inches='tight')
    plt.close(fig4b)
    print(f"  Saved: {p4b}")

    # 4C: Summary 2x2
    fig4c, axes4c = plt.subplots(2, 2, figsize=(16, 12))
    fig4c.suptitle(f'{latest} — Training Summary', fontsize=14, fontweight='bold')

    ax = axes4c[0, 0]
    ax.plot(bi, d['mean_reward'], 'o', alpha=0.2, color='#1f77b4', markersize=2)
    if len(d['mean_reward']) >= 10:
        ax.plot(bi[9:], smooth(d['mean_reward'], 10), color='#1f77b4', linewidth=2.5)
    ax.axhline(y=np.mean(d['mean_reward']), color='darkblue', ls='--', alpha=0.5,
               label=f'Mean: {np.mean(d["mean_reward"]):.3f}')
    ax.set_title(f'Mean Reward ({d["mean_reward"][0]:.3f} -> {d["mean_reward"][-1]:.3f})')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Reward'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes4c[0, 1]
    ax.fill_between(bi, 0, d['fall_pct'], alpha=0.3, color='red')
    ax.plot(bi, d['fall_pct'], 'r-', linewidth=1.5)
    ax.set_title(f'Fall Rate (mean={np.mean(d["fall_pct"]):.0f}%)')
    ax.set_xlabel('Iteration'); ax.set_ylabel('%'); ax.set_ylim(0, 105); ax.grid(True, alpha=0.3)

    ax = axes4c[1, 0]
    ax.plot(bi, d['base_height'], 'o', alpha=0.2, color='#2ca02c', markersize=2)
    if len(d['base_height']) >= 5:
        ax.plot(bi[4:], smooth(d['base_height'], 5), color='#2ca02c', linewidth=2)
    ax.axhline(y=0.89, color='green', ls='--', alpha=0.5, label='Target 0.89m')
    ax.set_title(f'Base Height ({d["base_height"][0]:.3f} -> {d["base_height"][-1]:.3f} m)')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Height (m)'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Foot trajectory tracking
    ax = axes4c[1, 1]
    ax.plot(bi, d['r_foot_height'], 'o', alpha=0.2, color='#ff7f0e', markersize=2, label='Foot Height')
    ax.plot(bi, d['r_foot_vel'], 'o', alpha=0.2, color='#9467bd', markersize=2, label='Foot Vel')
    if len(bi) >= 5:
        ax.plot(bi[4:], smooth(d['r_foot_height'], 5), color='#ff7f0e', linewidth=2)
        ax.plot(bi[4:], smooth(d['r_foot_vel'], 5), color='#9467bd', linewidth=2)
    ax.set_title(f'Foot Trajectory Tracking')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Score'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p4c = os.path.join(LOGS, '04c_run7_summary.png')
    fig4c.savefig(p4c, dpi=150, bbox_inches='tight')
    plt.close(fig4c)
    print(f"  Saved: {p4c}")

# ================================================================
# CHART 5: 跨Run对比 — 关键指标随时间变化
# ================================================================
fig5, axes5 = plt.subplots(3, 3, figsize=(20, 18))
fig5.suptitle('Cross-Run Comparison — Key Metrics Evolution\n(dwl_reproduce2, XBot-L 256 envs)',
              fontsize=14, fontweight='bold')

compare_metrics = [
    ('Mean Reward', 'mean_reward', 'Reward'),
    ('Fall Rate (%)', 'fall_pct', '%'),
    ('Base Height (m)', 'base_height', 'm'),
    ('|Roll| (rad)', 'roll', 'rad'),
    ('|Pitch| (rad)', 'pitch', 'rad'),
    ('Torque Mean (Nm)', 'torque_mean', 'Nm'),
    ('Foot Height Reward', 'r_foot_height', 'score'),
    ('Foot Vel Reward', 'r_foot_vel', 'score'),
    ('Orientation Reward', 'r_orientation', 'score'),
]

for i, (title, key, unit) in enumerate(compare_metrics):
    ax = axes5[i // 3, i % 3]
    for idx, run in enumerate(RUNS):
        if run not in run_data:
            continue
        d = run_data[run]
        data = d[key]
        if len(data) >= 5:
            ax.plot(d['iters'][4:], smooth(data, 5), color=colors_run[idx], linewidth=1.5, alpha=0.8, label=run)
        else:
            ax.plot(d['iters'], data, color=colors_run[idx], linewidth=1, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel('Iteration'); ax.set_ylabel(unit)
    ax.legend(fontsize=7, loc='best'); ax.grid(True, alpha=0.3)

plt.tight_layout()
p5 = os.path.join(LOGS, '05_cross_run_comparison.png')
fig5.savefig(p5, dpi=150, bbox_inches='tight')
plt.close(fig5)
print(f"  Saved: {p5}")

# ================================================================
# CHART 6: 跨Run — 起始/结束柱状图对比
# ================================================================
fig6, axes6 = plt.subplots(2, 3, figsize=(20, 13))
fig6.suptitle('Cross-Run Comparison — Start vs End Values\n(All runs, dwl_reproduce2)',
              fontsize=14, fontweight='bold')

bar_metrics = [
    ('Mean Reward', 'mean_reward'),
    ('Fall Rate (%)', 'fall_pct'),
    ('Base Height (m)', 'base_height'),
    ('|Roll| (rad)', 'roll'),
    ('Torque Mean (Nm)', 'torque_mean'),
    ('Foot Height Reward', 'r_foot_height'),
]

for i, (title, key) in enumerate(bar_metrics):
    ax = axes6[i // 3, i % 3]
    x = np.arange(len(RUNS))
    starts = []
    ends = []
    for run in RUNS:
        if run in run_data:
            d = run_data[run]
            starts.append(float(np.mean(d[key][:5])))  # avg of first 5
            ends.append(float(np.mean(d[key][-5:])))    # avg of last 5
        else:
            starts.append(0)
            ends.append(0)

    width = 0.35
    ax.bar(x - width/2, starts, width, label='Start', color='#1f77b4', alpha=0.7)
    ax.bar(x + width/2, ends, width, label='End', color='#ff7f0e', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(RUNS, rotation=30, ha='right', fontsize=8)
    ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
p6 = os.path.join(LOGS, '06_start_vs_end_bars.png')
fig6.savefig(p6, dpi=150, bbox_inches='tight')
plt.close(fig6)
print(f"  Saved: {p6}")

# ================================================================
# CHART 7: 奖励分布Histogram (train_log)
# ================================================================
fig7, axes7 = plt.subplots(1, 2, figsize=(14, 5))
fig7.suptitle('Reward & Loss Distribution — First Half vs Second Half',
              fontsize=14, fontweight='bold')

mid = len(t_rewards) // 2
ax = axes7[0]
ax.hist(t_rewards[:mid], bins=30, alpha=0.5, color='#1f77b4', label=f'First half (median={np.median(t_rewards[:mid]):.3f})')
ax.hist(t_rewards[mid:], bins=30, alpha=0.5, color='#ff7f0e', label=f'Second half (median={np.median(t_rewards[mid:]):.3f})')
ax.set_xlabel('Reward'); ax.set_ylabel('Count'); ax.set_title('Reward Distribution Shift')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes7[1]
ax.hist(t_losses[:mid], bins=30, alpha=0.5, color='#d62728', label=f'First half (median={np.median(t_losses[:mid]):.0f})')
ax.hist(t_losses[mid:], bins=30, alpha=0.5, color='#1f77b4', label=f'Second half (median={np.median(t_losses[mid:]):.0f})')
ax.set_xlabel('Loss'); ax.set_ylabel('Count'); ax.set_title('Loss Distribution Shift')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
p7 = os.path.join(LOGS, '07_distribution_shift.png')
fig7.savefig(p7, dpi=150, bbox_inches='tight')
plt.close(fig7)
print(f"  Saved: {p7}")

# ================================================================
# CHART 8: Run7 速度跟踪 + 接触力
# ================================================================
if latest in run_data:
    d = run_data[latest]
    bi = d['iters']

    fig8, axes8 = plt.subplots(2, 2, figsize=(16, 12))
    fig8.suptitle(f'{latest} — Velocity Tracking & Contacts', fontsize=14, fontweight='bold')

    ax = axes8[0, 0]
    ax.plot(bi, d['r_lin_vel'], 'o', alpha=0.2, color='#1f77b4', markersize=2)
    if len(bi) >= 5:
        ax.plot(bi[4:], smooth(d['r_lin_vel'], 5), color='#1f77b4', linewidth=2)
    ax.set_title(f'Linear Velocity Tracking ({d["r_lin_vel"][0]:.3f} -> {d["r_lin_vel"][-1]:.3f})')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Score'); ax.grid(True, alpha=0.3)

    ax = axes8[0, 1]
    ax.plot(bi, d['r_ang_vel'], 'o', alpha=0.2, color='#ff7f0e', markersize=2)
    if len(bi) >= 5:
        ax.plot(bi[4:], smooth(d['r_ang_vel'], 5), color='#ff7f0e', linewidth=2)
    ax.set_title(f'Angular Velocity Tracking ({d["r_ang_vel"][0]:.3f} -> {d["r_ang_vel"][-1]:.3f})')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Score'); ax.grid(True, alpha=0.3)

    ax = axes8[1, 0]
    ax.plot(bi, d['r_periodic_contact'], 'o', alpha=0.2, color='#2ca02c', markersize=2)
    if len(bi) >= 5:
        ax.plot(bi[4:], smooth(d['r_periodic_contact'], 5), color='#2ca02c', linewidth=2)
    ax.set_title(f'Periodic Contact ({d["r_periodic_contact"][0]:.3f} -> {d["r_periodic_contact"][-1]:.3f})')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Score'); ax.grid(True, alpha=0.3)

    ax = axes8[1, 1]
    ax.plot(bi, d['r_periodic_vel'], 'o', alpha=0.2, color='#9467bd', markersize=2)
    if len(bi) >= 5:
        ax.plot(bi[4:], smooth(d['r_periodic_vel'], 5), color='#9467bd', linewidth=2)
    ax.set_title(f'Periodic Velocity ({d["r_periodic_vel"][0]:.3f} -> {d["r_periodic_vel"][-1]:.3f})')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Score'); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p8 = os.path.join(LOGS, '08_run7_velocity_contacts.png')
    fig8.savefig(p8, dpi=150, bbox_inches='tight')
    plt.close(fig8)
    print(f"  Saved: {p8}")

# ================================================================
# 终端报告
# ================================================================
print("\n" + "=" * 70)
print("DWL_REPRODUCE2 综合分析报告")
print("=" * 70)

print(f"\n[Training Log] train_log.txt:")
print(f"  Iterations: {t_iters[0]} -> {t_iters[-1]} ({len(t_iters)} total)")
print(f"  Reward: {t_rewards[0]:.4f} -> {t_rewards[-1]:.4f}  (best: {np.max(t_rewards):.4f} @ iter {t_iters[np.argmax(t_rewards)]})")
print(f"  Loss: {t_losses[0]:.0f} -> {t_losses[-1]:.1f}")
print(f"  Speed: avg {avg_sp:.0f} steps/s")
print(f"  Total time: {t_times[-1]:.1f} min ({t_times[-1]/60:.1f} h)")

print(f"\n[Per-Run Summary]:")
for run in RUNS:
    if run not in run_data:
        print(f"  {run}: NO DATA")
        continue
    d = run_data[run]
    r_start = np.mean(d['mean_reward'][:5])
    r_end = np.mean(d['mean_reward'][-5:])
    f_start = np.mean(d['fall_pct'][:5])
    f_end = np.mean(d['fall_pct'][-5:])
    bh_start = np.mean(d['base_height'][:5])
    bh_end = np.mean(d['base_height'][-5:])
    print(f"  {run}: {d['n_rows']:>4d} rows | "
          f"Reward {r_start:.3f}->{r_end:.3f} | "
          f"Fall {f_start:.0f}%->{f_end:.0f}% | "
          f"Height {bh_start:.3f}->{bh_end:.3f}m")

print(f"\n[Charts saved to]: {LOGS}")
for f in sorted(os.listdir(LOGS)):
    if f.endswith('.png'):
        print(f"  {f}")

print("\nDone!")
