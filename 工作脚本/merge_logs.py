import re, os, statistics

temp_base = os.path.expandvars(r'%APPDATA%\..\Local\Temp\claude\C--Users-czy66')

log_files = [
    ('Phase1_from_scratch',       os.path.join(temp_base, '745bae5a-f6b2-44f2-ae4a-4568bdb4672d/tasks/bkiik2owd.output')),
    ('Phase2_from_scratch_2',     os.path.join(temp_base, '745bae5a-f6b2-44f2-ae4a-4568bdb4672d/tasks/bqzq0u1o9.output')),
    ('Phase3_from_scratch_3',     os.path.join(temp_base, '745bae5a-f6b2-44f2-ae4a-4568bdb4672d/tasks/b9mbg0qv0.output')),
    ('Phase4_resume_from_550',     os.path.join(temp_base, 'd5d8e2e0-fbfb-4495-8199-f1b8bd5acd45/tasks/borfoixou.output')),
    ('Phase5_resume_from_1100',    os.path.join(temp_base, 'd5d8e2e0-fbfb-4495-8199-f1b8bd5acd45/tasks/bm7x81tmm.output')),
]

all_rows = []
for label, path in log_files:
    if not os.path.exists(path):
        print(f'{label}: FILE NOT FOUND')
        continue
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    pattern = r'Iter\s+(\d+)/(\d+)\s+\|\s+Reward:\s+([\d.]+)\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+Steps/s:\s+([\d.]+)'
    matches = re.findall(pattern, content)
    if matches:
        iters = [int(m[0]) for m in matches]
        rewards = [float(m[2]) for m in matches]
        losses = [float(m[3]) for m in matches]
        speeds = [float(m[4]) for m in matches]
        print(f'{label}: {len(matches):4d} iters, iter {iters[0]:5d} -> {iters[-1]:5d}  |  '
              f'Reward {rewards[0]:.3f} -> {rewards[-1]:.3f}  |  '
              f'Loss {losses[0]:.0f} -> {losses[-1]:.0f}  |  '
              f'Speed {statistics.mean(speeds):.0f} avg')
        for i, r, l, s in zip(iters, rewards, losses, speeds):
            all_rows.append((label, i, r, l, s))

# Deduplicate by iteration number (keep last occurrence)
seen = {}
for row in all_rows:
    seen[row[1]] = row  # iter -> row
merged = sorted(seen.values(), key=lambda x: x[1])

# Write merged CSV
csv_path = os.path.expanduser(r'~\Desktop\dwl_reproduce\training_data_all_phases.csv')
with open(csv_path, 'w', encoding='utf-8-sig') as f:
    f.write('Phase,Iteration,Reward,Loss,Steps_per_sec\n')
    for phase, it, r, l, s in merged:
        f.write(f'{phase},{it},{r:.4f},{l:.1f},{s:.0f}\n')

print(f'\nMerged CSV: {csv_path}')
print(f'Total unique iterations: {len(merged)} (iter {merged[0][1]} -> {merged[-1][1]})')

# Summary by phase
print(f'\n=== Phase Summary ===')
for label, _ in log_files:
    rows = [r for r in merged if r[0] == label]
    if not rows:
        continue
    r_vals = [r[2] for r in rows]
    l_vals = [r[3] for r in rows]
    s_vals = [r[4] for r in rows]
    print(f'{label}:')
    print(f'  {len(rows)} iters, iter {rows[0][1]} -> {rows[-1][1]}')
    print(f'  Reward: {min(r_vals):.3f} ~ {max(r_vals):.3f}, mean={statistics.mean(r_vals):.4f}')
    print(f'  Loss:   {min(l_vals):.0f} ~ {max(l_vals):.0f}, mean={statistics.mean(l_vals):.0f}')
    print(f'  Speed:  {min(s_vals):.0f} ~ {max(s_vals):.0f}, mean={statistics.mean(s_vals):.0f}')
    print()

# Check gaps
print('=== Gaps in iteration sequence ===')
gaps = []
for i in range(1, len(merged)):
    gap = merged[i][1] - merged[i-1][1]
    if gap > 1:
        print(f'  Gap: iter {merged[i-1][1]} -> {merged[i][1]} ({gap} iters missing)')
        gaps.append(gap)
if not gaps:
    print('  No gaps!')
