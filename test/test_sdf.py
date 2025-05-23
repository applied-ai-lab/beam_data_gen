import numpy as np
import mujoco
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from beam_data_gen.simulator.trimesh import MujocoToTrimeshScene
from beam_data_gen.simulator.sdf_scene import SDFScene


def plot_mesh(mesh):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get vertices and faces
    for face in mesh.faces:
        tri = mesh.vertices[face]
        ax.plot_trisurf(
            tri[:, 0], tri[:, 1], tri[:, 2],
            triangles=[[0, 1, 2]], color='lightgrey', alpha=0.5
        )

    ax.set_box_aspect([1, 1, 1])
    plt.title("Trimesh Mesh (simplified view)")
    plt.show()


def plot_potential_slice(potential_field, axis='z', index=25):
    if axis == 'z':
        slice = potential_field[:, :, index]
    elif axis == 'y':
        slice = potential_field[:, index, :]
    elif axis == 'x':
        slice = potential_field[index, :, :]
    else:
        raise ValueError("Invalid axis")

    plt.imshow(slice.T, origin='lower', cmap='inferno')
    plt.colorbar(label="Potential")
    plt.title(f"Potential field slice at {axis}={index}")
    plt.axis('equal')
    plt.show()

    
def potential_from_sdf(sdf_val, epsilon=0.05):
    if sdf_val < epsilon:
        return 0.5 * (epsilon - sdf_val)**2
    else:
        return 0.0
    
    
def compute_potential_field(sdf, bounds, resolution=50, epsilon=0.05):
    """
    Computes a 3D potential field from a Trimesh-based SDF.
    
    Returns:
        grid_vals: 3D array of potential values
        grid_points: meshgrid of X, Y, Z coordinates
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    x = np.linspace(xmin, xmax, resolution)
    y = np.linspace(ymin, ymax, resolution)
    z = np.linspace(zmin, zmax, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    sdf_vals = np.array([sdf(p) for p in points])
    potentials = np.array([potential_from_sdf(s, epsilon) for s in sdf_vals])
    potential_field = potentials.reshape((resolution, resolution, resolution))

    return potential_field, (X, Y, Z)


def plot_potential_and_gradients(X, Y, U, dU_dx, dU_dy):
    plt.figure(figsize=(7, 6))
    plt.contourf(X, Y, U, levels=50, cmap='inferno')
    plt.colorbar(label="Potential")
    plt.quiver(X, Y, -dU_dx, -dU_dy, color='white', scale=10)
    plt.title("Potential Field and Repulsive Force Vectors")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()
    

def main():

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

    # Convert to trimesh scene
    scene_builder = MujocoToTrimeshScene(model, data)
    scene = scene_builder.get_scene()
    # scene.show(None)

    # Build SDF from scene
    sdf_scene = SDFScene(scene)
    results = sdf_scene.query_geom(model, data, geom_name="square_beam_1")
    
    # Find a potential field
    potential_field, (X, Y, Z) = compute_potential_field(sdf_scene.sdf, 
                                                bounds=[0, 0.316, -0.04, 0.02, 0., 0.065])

    gx, gy, gz = np.gradient(potential_field)
    
    plot_potential_and_gradients(X, Y, Z, gx, gy)
    
    import pdb
    pdb.set_trace()
    
    plot_potential_slice(potential_field, axis="z", index=0)
    
    # plot_mesh(scene)

    print("Vertex SDFs:", results["sdf"])
    print("Vertex gradients:", results["gradients"])
    
    # Plot the gradients
    verts = results["vertices"]
    grads = results["gradients"]
    
    print("delta pos:", results["position_grad"])
    print("delta rot (axis-angle):", results["rotation_grad"])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(verts[:, 0], verts[:, 1], verts[:, 2],
                grads[:, 0], grads[:, 1], grads[:, 2], length=0.01)
    plt.show()
    
    return 0


if __name__ == "__main__":
    main()
