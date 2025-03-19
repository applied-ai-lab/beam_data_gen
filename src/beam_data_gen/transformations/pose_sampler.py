import numpy as np
import scipy.interpolate as interp
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d


class PoseSamplerParams:
    """ Class for parameters for generating pose samples
    """
    def __init__(self, dt, duration, seed, velocity_mask):
        # Protected
        self.dt = dt
        self.duration = duration
        self.seed = seed
        self._velocity_mask = velocity_mask
        # Private
        self._no_samples = None
        
    @property
    def no_samples(self):
        self._no_samples = int(self.duration / self.dt)
        return self._no_samples


class PoseSampler:
    def __init__(self):
        pass
    
    def generate_smooth_transforms(self, init_pose_quat:np.array, params: PoseSamplerParams) -> np.array:
        """ Generate smooth transforms (quaternion form x,y,z,x,y,z,w)

        Args:
            params (PoseSamplerParams): Parameter class 

        Returns:
            np.array: Poses with shape [traj_len, quat]
        """
        _, vel = self.generate_smooth_velocities(params.no_samples, params.duration, params.seed)
        vel *= params._velocity_mask
        poses = self.apply_velocities_to_poses(init_pose_quat, vel, params.dt)
        return poses

    def generate_smooth_velocities(self, num_samples, time_span, seed=None):
        """Generate smooth linear and angular velocities using interpolation."""
        
        assert num_samples >= 5, f"The number of samples must be greater than 5, current value is {num_samples}."
        
        if seed:
            np.random.seed(seed)

        t = np.linspace(0, time_span, num_samples)
        
        # Generate random control points for smooth interpolation
        num_ctrl_pnts = max(num_samples // 5, 5)
        control_t = np.linspace(0, time_span, num_ctrl_pnts)  # Fewer control points
        control_v = np.random.randn(len(control_t), 6) * 0.25  # 6D velocity (linear + angular)

        # Interpolate for smooth motion
        smooth_v = np.array([
            interp.interp1d(control_t, control_v[:, i], kind='cubic', fill_value="extrapolate")(t)
            for i in range(6)
        ]).T  # Shape (num_samples, 6)

        return t, smooth_v

    def apply_velocities_to_poses(self, initial_pose, velocities, dt):
        """Apply smooth velocities to SE(3) poses."""
        # Preallocate the final poses
        traj_len = velocities.shape[0]
        
        poses = np.zeros([velocities.shape[0], 7])
        poses[0, :] = initial_pose.squeeze()        
        
        for k in range(1, traj_len):
            translation = poses[k - 1, 0:3]
            rotation = R.from_quat(poses[k - 1, 3:])

            linear_vel = velocities[k - 1, 0:3]  # Small translation update
            angular_vel = velocities[k - 1, 3:]  # Small rotation update

            # Apply translation update
            poses[k, 0:3] = translation + linear_vel * dt

            # Apply rotation update (small angle approximation)
            delta_rot = R.from_rotvec(angular_vel * dt)
            poses[k, 3:] = (delta_rot * rotation).as_quat()

        return poses

    def perturb_poses_tangent_space(self, initial_poses, noise_level, num_samples):
        """
        Perturbs SE(3) poses using uniform noise, applying rotation perturbations in tangent space.

        Args:
            initial_poses (ndarray): Shape (N, 7) where each pose is (x, y, z, qx, qy, qz, qw).
            noise_level (float): Maximum perturbation magnitude for translation & rotation.
            num_samples (int): Number of noisy samples per pose.

        Returns:
            noisy_poses (ndarray): Shape (num_samples, N, 7), perturbed poses.
        """
        num_poses = initial_poses.shape[0]
        
        # Generate uniform translation noise
        translation_noise = np.random.uniform(-noise_level, noise_level, size=(num_samples, num_poses, 3))
        
        # Generate uniform rotation noise in tangent space (Lie algebra so(3))
        tangent_rotation_noise = np.random.uniform(-noise_level, noise_level, size=(num_samples, num_poses, 3))

        noisy_poses = []
        for i in range(num_samples):
            perturbed_translations = initial_poses[:, :3] + translation_noise[i]

            # Convert original quaternions to Rotation objects
            original_rotations = R.from_quat(initial_poses[:, 3:])
            
            # Map tangent space perturbation to SO(3) using the exponential map
            delta_rotations = R.from_rotvec(tangent_rotation_noise[i])
            
            # Apply the rotation perturbation
            perturbed_rotations = delta_rotations * original_rotations
            
            # Convert back to quaternions
            perturbed_quaternions = perturbed_rotations.as_quat()
            
            # Store the noisy poses
            noisy_poses.append(np.hstack([perturbed_translations, perturbed_quaternions]))

        return np.array(noisy_poses)
    
    def smooth_so3_orientations_corrected(self, orientations, sigma=1.0):
        """
        Smooths a sequence of SO(3) orientations using a Gaussian filter in the proper tangent space.

        Args:
            orientations (ndarray): Shape (N, 4) - sequence of quaternions (qx, qy, qz, qw).
            sigma (float): Smoothing factor for the Gaussian filter.

        Returns:
            smoothed_orientations (ndarray): Shape (N, 4) - smoothed sequence of quaternions.
        """
        # Convert quaternions to Rotation objects
        rotations = R.from_quat(orientations)

        # Compute relative rotations: R_t R_{t-1}^{-1}
        relative_rotations = rotations[:-1] * rotations[1:].inv()

        # Convert to tangent space using logarithm map
        log_rotations = relative_rotations.as_rotvec()  # Now this is a true tangent space representation

        # Apply Gaussian smoothing in tangent space
        smoothed_log_rotations = gaussian_filter1d(log_rotations, sigma=sigma, axis=0)

        # Convert back to SO(3) using the exponential map
        smoothed_rotations = [rotations[0]]
        for i in range(len(smoothed_log_rotations)):
            smoothed_rotations.append(R.from_rotvec(smoothed_log_rotations[i]) * smoothed_rotations[-1])

        return R.concatenate(smoothed_rotations).as_quat()

# # Example Usage
# num_samples = 100  # Number of perturbed samples per pose
# noise_level = 0.1  # Max perturbation magnitude (meters and radians)

# # Example initial SE(3) poses (position + quaternion)
# initial_poses = np.array([
#     [0, 0, 0, 0, 0, 0, 1],  # Identity pose
#     [1, 2, 1, 0, 0, 0, 1],  # Another pose
# ])

# # Generate noisy data
# noisy_poses = perturb_poses_uniform(initial_poses, noise_level, num_samples)

# print("Noisy Poses Shape:", noisy_poses.shape)  # (num_samples, num_initial_poses, 7)


# # Example Usage
# num_samples = 100
# time_span = 5.0  # 5 seconds
# dt = time_span / num_samples

# # Example initial SE(3) poses (position + quaternion)
# initial_poses = np.array([
#     [0, 0, 0, 0, 0, 0, 1],  # Identity pose
#     [1, 2, 1, 0, 0, 0, 1],  # Another pose
# ])

# # Generate smooth velocities
# t, smooth_v = generate_smooth_velocities(num_samples, time_span, seed=42)

# # Apply velocities
# modified_poses = np.array([
#     apply_velocities_to_poses(initial_poses, smooth_v, dt)
#     for _ in range(num_samples)
# ])

# print("Modified Poses Shape:", modified_poses.shape)  # (num_samples, num_initial_poses, 7)
