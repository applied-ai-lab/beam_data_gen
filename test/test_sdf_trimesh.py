import trimesh
import pysdf
import numpy as np
import mujoco
import matplotlib.pyplot as plt

from beam_data_gen.simulator.trimesh import MujocoToTrimeshScene
from beam_data_gen.simulator.sdf_scene import SDFScene


# --- Plot ---
def plot_potential_and_gradients(X, Y, U, dU_dx, dU_dy):
    plt.figure(figsize=(7, 6))
    plt.contourf(X, Y, U, levels=50, cmap='inferno')
    plt.colorbar(label="Potential")
    plt.quiver(X, Y, dU_dx, dU_dy, color='white', scale=10)
    plt.title("Potential Field and Repulsive Force Vectors")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def main():

    # --- Load or create mesh ---
    model = mujoco.MjModel.from_xml_path('resources/configs/robot_and_square.xml')
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    node_names = ["square_pin_A",
                    "square_beam_2",
                    "square_pin_B",
                    "square_beam_3",
                    "square_pin_C",
                    "square_beam_4",
                    "square_pin_D"]
    
    scene_builder = MujocoToTrimeshScene(model, data)
    scene_builder.update_scene(node_names)
    sdf_scene = SDFScene(scene_builder.scene)

    bounds = [-0.5, 0.5, -0.15, 0.75]  # [xmin, xmax, ymin, ymax]
    X, Y, U, dU_dx, dU_dy = sdf_scene.compute_potential_and_gradient(bounds)
    U = 1 - U
    plot_potential_and_gradients(X, Y, U, dU_dx, dU_dy)
    
    # World points to query
    grad = sdf_scene.query_geom(model, data, "square_beam_1")
    
    print(grad)
        
    return 0

if __name__ == "__main__":
    main()
