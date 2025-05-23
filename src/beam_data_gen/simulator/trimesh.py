import numpy as np
import trimesh
import mujoco


class MujocoToTrimeshScene:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

        # Filter only box geoms
        self.geom_ids = [
            i for i in range(model.ngeom)
            if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
        ]
        self.geom_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            for i in self.geom_ids
        ]
        self.name_to_id = {
            name: gid for name, gid in zip(self.geom_names, self.geom_ids)
        }

        self._scene = self._build_scene()        

    def _geom_to_trimesh(self, geom_id):
        size = self.model.geom_size[geom_id][:3]
        pos = self.data.geom_xpos[geom_id]
        mat = self.data.geom_xmat[geom_id].reshape(3, 3)

        mesh = trimesh.creation.box(extents=2 * size)
        transform = np.eye(4)
        transform[:3, :3] = mat
        transform[:3, 3] = pos
        mesh.apply_transform(transform)
        return mesh

    def _build_scene(self):
        scene = trimesh.Scene()
        for name in self.geom_names:
            mesh = self._geom_to_trimesh(self.name_to_id[name])
            if mesh is not None:
                scene.add_geometry(mesh, node_name=name)
        scene = trimesh.util.concatenate(scene)
        for _ in range(4):
            scene = scene.subdivide()
        return scene

    def update_scene(self, geom_names: list[str] = None):
        """
        Update positions of selected geoms in the scene.
        If geom_names is None, all geoms are updated.
        """
        if geom_names is None:
            geom_names = self.geom_names
        else:
            self.geom_names = geom_names
        self._scene = self._build_scene()

    @property
    def scene(self):
        return self._scene
