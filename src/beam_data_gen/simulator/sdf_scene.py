import trimesh
import mujoco
import pysdf
import numpy as np
from scipy.ndimage import zoom


class SDFScene:
    def __init__(self, scene: trimesh.Scene):  
        self.mesh = scene # Merge to a single mesh
        self.sdf = pysdf.SDF(self.mesh.vertices, self.mesh.faces)

    def potential_from_sdf(self, sdf_val, epsilon=0.05):
        # --- Potential function (quadratic repulsive) ---
        if sdf_val < epsilon:
            return 0.5 * (epsilon - sdf_val)**2
        else:
            return 0.0

    def compute_potential_and_gradient(self, bounds, res=50, epsilon=0.0001, z_slice=0.0):
        # --- Compute potential and gradient on a grid ---
        xmin, xmax, ymin, ymax = bounds
        x = np.linspace(xmin, xmax, res)
        y = np.linspace(ymin, ymax, res)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = np.full_like(X, z_slice)

        points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        sdf_vals = np.array([self.sdf(p) for p in points]).squeeze()
        
        pot_lst = [self.potential_from_sdf(s, epsilon) for s in sdf_vals]
        
        potentials = np.array(pot_lst)
        U = potentials.reshape((res, res))

        # Compute gradient of potential
        dU_dx, dU_dy = np.gradient(U, x[1] - x[0], y[1] - y[0])

        return X, Y, U, dU_dx, dU_dy
    
    def gradients(self, point, epsilon=0.1, delta=1e-3):
        """
        Compute gradient of potential field at a 3D world point using central finite differences.

        Args:
            sdf_fn: pysdf.MeshSDF or similar, callable sdf(xyz)
            point: np.array of shape (3,) world coordinates
            epsilon: distance threshold for repulsive potential
            delta: small step size for finite differencing

        Returns:
            grad: np.array of shape (3,) gradient of potential at the point
            potential: scalar potential at the point
        """
        point = np.asarray(point)
        grad = np.zeros(3)

        sdf_val = self.sdf(point)
        potential = self.potential_from_sdf(sdf_val, epsilon)

        for i in range(3):
            d = np.zeros(3)
            d[i] = delta

            f_plus = self.potential_from_sdf(self.sdf(point + d), epsilon)
            f_minus = self.potential_from_sdf(self.sdf(point - d), epsilon)

            grad[i] = (f_plus - f_minus) / (2 * delta)

        return grad, potential                          
    
    def query_geom(self, model: mujoco.MjModel, data: mujoco.MjData, geom_name: str):
        """
        Compute gradients of the SDF w.r.t. the box center's position and orientation,
        based on the vertices in collision (SDF < 0).
        """
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)

        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
            raise NotImplementedError("Only box geoms are supported in this query.")

        size = model.geom_size[geom_id][:3]  # Half-extents
        pos = data.geom_xpos[geom_id]
        mat = data.geom_xmat[geom_id].reshape(3, 3)

        # Create box vertices in local frame
        local_box = trimesh.creation.box(extents=2 * size)
        local_offsets = local_box.vertices

        # Transform vertices to world frame
        transform = np.eye(4)
        transform[:3, :3] = mat
        transform[:3, 3] = pos
        world_vertices = trimesh.transformations.transform_points(local_offsets, transform)

        sdf_values = np.array([self.sdf(v) for v in world_vertices])
        
        gradients = np.vstack([self.gradients(v)[0] for v in world_vertices])

        # ∂SDF/∂x ≈ sum of gradients (push the box away)
        position_grad = np.sum(gradients, axis=0)

        # ∂SDF/∂θ ≈ sum over cross(local_offset, gradient)
        # (how SDF changes with small rotations)
        rotation_grad = np.zeros(3)
        for offset, grad in zip(local_offsets, gradients):
            world_offset = mat @ offset  # bring local offset to world frame
            rotation_grad += np.cross(world_offset, grad)

        return {
                "position_grad": position_grad,
                "rotation_grad": rotation_grad,
                }
