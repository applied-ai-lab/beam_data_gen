import time

import numpy as np
import mujoco
from mujoco import MjModel, MjData
import mujoco.viewer


# # Load the MuJoCo model
# model = MjModel.from_xml_path("scene.xml")
# data = MjData(model)

# Function to check for collisions
def check_collisions(data):
    """
    Checks for collisions in the MuJoCo simulation.
    Returns True if any pair of geoms are in contact.
    """
    for i in range(data.ncon):  # Iterate through contacts
        contact = data.contact[i]
        geom1 = contact.geom1
        geom2 = contact.geom2
        print(f"Collision detected between geom {geom1} and geom {geom2}")
        return True  # Collision detected
    return False  # No collision

m = mujoco.MjModel.from_xml_path('resources/configs/three_beams.xml')
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
  start = time.time()
  while viewer.is_running():
    step_start = time.time()
    
    # d.qpos[7] += 0.5 * np.random.randn(1)   
    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)
    
    import pdb
    pdb.set_trace()

    # Check for collisions
    if check_collisions(d):
        print("Cuboids are in collision.")

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
