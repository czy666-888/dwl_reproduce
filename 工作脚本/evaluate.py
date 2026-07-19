"""
评测脚本: 加载训练好的模型, 在MuJoCo中运行, 记录机器人实际行为数据
输出: 速度跟踪 / 基座高度 / 姿态(是否摔倒) / 行走轨迹
"""
import os, sys, time, csv
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from dwl_config import DWLConfig
from dwl_networks import DWLModel
from dwl_env import XBotLMuJoCoEnv

def evaluate(cmd_vx, cmd_vy, cmd_vyaw, model, cfg, device, render=False, max_steps=1200):
    """Run one episode with fixed velocity command, record everything."""
    env = XBotLMuJoCoEnv(cfg, model=None, render=render)
    obs, _ = env.reset()

    # Override command to a fixed value (instead of random)
    env.commands = np.array([cmd_vx, cmd_vy, cmd_vyaw])

    # GRU history
    seq_len = 5
    obs_history = [obs.copy() for _ in range(seq_len)]

    h_prev = torch.zeros(1, 1, cfg.gru_hidden, device=device)

    records = []
    for step in range(max_steps):
        # Build obs sequence
        obs_seq = np.stack(obs_history, axis=0)  # (seq_len, obs_dim)
        obs_seq_t = torch.from_numpy(obs_seq).unsqueeze(0).float().to(device)  # (1, seq_len, obs_dim)

        with torch.no_grad():
            action, h_prev = model.act_inference(obs_seq_t, h_prev)
        action_np = action.squeeze(0).cpu().numpy()

        obs, _, reward, terminated, _ = env.step(action_np)

        # Update history
        obs_history.pop(0)
        obs_history.append(obs.copy())

        # Record
        base_pos = env.data.qpos[0:3].copy()         # [x, y, z]
        base_quat = env.data.qpos[3:7].copy()         # quaternion
        base_vel_world = env.data.qvel[0:3].copy()     # world-frame velocity
        base_ang_vel = env._get_base_angular_velocity()
        euler = env._get_base_euler()
        joint_pos = env._get_joint_positions()
        foot_pos = env._get_foot_positions()
        contact_forces = env._get_contact_forces()

        records.append({
            'step': step,
            'time': step * cfg.policy_dt,
            'base_x': base_pos[0], 'base_y': base_pos[1], 'base_z': base_pos[2],
            'roll': euler[0], 'pitch': euler[1], 'yaw': euler[2],
            'vel_x_world': base_vel_world[0],
            'vel_y_world': base_vel_world[1],
            'vel_z_world': base_vel_world[2],
            'ang_vel_yaw': base_ang_vel[2],
            'cmd_vx': env.commands[0],
            'cmd_vy': env.commands[1],
            'cmd_vyaw': env.commands[2],
            'reward': reward,
            'left_foot_z': foot_pos[0][2],
            'right_foot_z': foot_pos[1][2],
            'contact_left': contact_forces[0],
            'contact_right': contact_forces[1],
        })

        if terminated:
            break

    if env.viewer:
        env.viewer.close()
    return records


def plot_results(all_records, cfg, save_dir):
    """Plot walking behavior for all test commands."""
    n_cmds = len(all_records)
    fig, axes = plt.subplots(n_cmds, 5, figsize=(22, 3.5 * n_cmds))
    if n_cmds == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle('XBot-L Walking Evaluation — Trained DWL Model\n', fontsize=14, fontweight='bold')

    for i, (label, records) in enumerate(all_records.items()):
        t = np.array([r['time'] for r in records])
        base_z = np.array([r['base_z'] for r in records])
        roll = np.array([r['roll'] for r in records])
        pitch = np.array([r['pitch'] for r in records])
        vel_x = np.array([r['vel_x_world'] for r in records])
        vel_y = np.array([r['vel_y_world'] for r in records])
        ang_yaw = np.array([r['ang_vel_yaw'] for r in records])
        cmd_vx = records[0]['cmd_vx']
        cmd_vy = records[0]['cmd_vy']
        cmd_vyaw = records[0]['cmd_vyaw']
        base_x = np.array([r['base_x'] for r in records])
        base_y = np.array([r['base_y'] for r in records])
        lf_z = np.array([r['left_foot_z'] for r in records])
        rf_z = np.array([r['right_foot_z'] for r in records])

        duration = t[-1]
        fell = duration < 23.0  # didn't survive full 24s

        # Col 1: Base height
        ax = axes[i, 0]
        ax.plot(t, base_z, 'b', linewidth=1.5)
        ax.axhline(y=cfg.base_height_target, color='green', linestyle='--', label=f'Target ({cfg.base_height_target}m)')
        ax.axhline(y=cfg.termination_height, color='red', linestyle='--', alpha=0.5, label='Termination')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Height (m)')
        ax.set_title(f'Base Height ({"FELL" if fell else "Survived"} {duration:.1f}s)')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # Col 2: Roll / Pitch (stability)
        ax = axes[i, 1]
        ax.plot(t, np.abs(roll), 'r', linewidth=1.2, label='|Roll|')
        ax.plot(t, np.abs(pitch), 'orange', linewidth=1.2, label='|Pitch|')
        ax.axhline(y=cfg.termination_orientation, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Angle (rad)')
        ax.set_title('Orientation (stability)')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.2)

        # Col 3: Linear velocity tracking
        ax = axes[i, 2]
        ax.plot(t, vel_x, 'b', linewidth=1.5, label='Actual vx')
        ax.plot(t, vel_y, 'r', linewidth=1.5, label='Actual vy')
        ax.axhline(y=cmd_vx, color='blue', linestyle='--', alpha=0.4, label=f'Cmd vx={cmd_vx}')
        ax.axhline(y=cmd_vy, color='red', linestyle='--', alpha=0.4, label=f'Cmd vy={cmd_vy}')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Velocity (m/s)')
        ax.set_title(f'Velocity Tracking (cmd: vx={cmd_vx}, vy={cmd_vy})')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # Col 4: Yaw rate tracking
        ax = axes[i, 3]
        ax.plot(t, ang_yaw, 'purple', linewidth=1.5, label='Actual')
        ax.axhline(y=cmd_vyaw, color='purple', linestyle='--', alpha=0.4, label=f'Cmd={cmd_vyaw}')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Yaw rate (rad/s)')
        ax.set_title('Yaw Rate Tracking')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # Col 5: Walking trajectory (top-down)
        ax = axes[i, 4]
        ax.plot(base_x, base_y, 'b', linewidth=1.5)
        ax.plot(base_x[0], base_y[0], 'go', markersize=8, label='Start')
        ax.plot(base_x[-1], base_y[-1], 'ro', markersize=8, label='End')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.set_title(f'Trajectory ({"Fell early!" if fell else f"{duration:.1f}s"})')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        ax.axis('equal')

        # Row label
        axes[i, 0].set_ylabel(f'[{label}]\nHeight (m)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(save_dir, 'evaluation_behavior.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Chart saved: {save_path}')
    return save_path


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    cfg = DWLConfig()
    cfg.num_envs = 1

    # Load best model
    model = DWLModel(cfg).to(device)
    checkpoint_path = os.path.join(cfg.checkpoint_dir, 'best_model.pt')
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(cfg.checkpoint_dir, 'final_model.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f'Loaded: {checkpoint_path} (trained to iter {checkpoint.get("global_step", "?")})')

    # Test commands: (vx, vy, vyaw)
    test_commands = [
        ('Forward 0.3 m/s',  0.3, 0.0, 0.0),
        ('Forward 0.5 m/s',  0.5, 0.0, 0.0),
        ('Turn right 0.2 rad/s',  0.1, 0.0, -0.2),
        ('Diagonal 0.3+0.2 m/s', 0.3, 0.2, 0.0),
    ]

    all_records = {}
    all_csv_rows = []

    for label, vx, vy, vyaw in test_commands:
        print(f'\nTesting: {label} (vx={vx}, vy={vy}, vyaw={vyaw})')
        records = evaluate(vx, vy, vyaw, model, cfg, device, render=False, max_steps=2400)
        all_records[label] = records

        duration = records[-1]['time']
        fell = duration < 23.0
        actual_vx_mean = np.mean([r['vel_x_world'] for r in records[100:]])  # skip startup
        actual_vy_mean = np.mean([r['vel_y_world'] for r in records[100:]])
        actual_vyaw_mean = np.mean([r['ang_vel_yaw'] for r in records[100:]])
        base_z_mean = np.mean([r['base_z'] for r in records])

        print(f'  Duration: {duration:.1f}s  Fell: {fell}  Steps: {len(records)}')
        print(f'  Base height: mean={base_z_mean:.3f}m (target={cfg.base_height_target})')
        print(f'  Velocity:  actual vx={actual_vx_mean:.3f} (cmd={vx}),  vy={actual_vy_mean:.3f} (cmd={vy}),  vyaw={actual_vyaw_mean:.3f} (cmd={vyaw})')
        print(f'  Reward: mean={np.mean([r["reward"] for r in records]):.3f}')

        # Append to CSV
        for r in records:
            all_csv_rows.append({
                'test_case': label,
                **r,
            })

    # Save CSV
    csv_path = os.path.join(os.path.dirname(__file__), 'evaluation_data.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=all_csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_csv_rows)
    print(f'\nCSV saved: {csv_path} ({len(all_csv_rows)} rows)')

    # Plot
    plot_results(all_records, cfg, os.path.dirname(__file__))


if __name__ == '__main__':
    main()
