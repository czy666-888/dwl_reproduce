"""Quick test: MuJoCo model loading + basic simulation"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import mujoco
import numpy as np

model_path = os.path.join(os.path.dirname(__file__), 'models', 'XBot-L.xml')
print(f'Loading: {model_path}')
print(f'Exists: {os.path.exists(model_path)}')

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

print(f'Model loaded OK!')
print(f'  nq (position DOFs): {model.nq}')
print(f'  nv (velocity DOFs): {model.nv}')
print(f'  nu (actuators):     {model.nu}')

# List actuators
print(f'\nActuators ({model.nu}):')
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f'  [{i}] {name}')

# Test stepping
print('\nStepping 100 times...')
for i in range(100):
    mujoco.mj_step(model, data)

base_pos = data.qpos[0:3]
base_vel = data.qvel[0:3]
print(f'Base position: {base_pos}')
print(f'Base velocity: {base_vel}')
print('MuJoCo simulation: OK!')
