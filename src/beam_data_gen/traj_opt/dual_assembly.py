from typing import List

import torch
from torch import nn
from torch.autograd import grad
import numpy as np
import mujoco
from filterpy.monte_carlo import systematic_resample
from tqdm import trange
import mujoco
from scipy.spatial.transform import Rotation as R

from beam_data_gen.traj_opt.traj_opt_base import (TrajOptParams, TrajOptBase)
from beam_data_gen.simulator.square_robot_sim import SquareRobotSim

# Squared-distance threshold for contact detection (3D position).
# 0.002 ≈ 4.5 cm effective radius — large enough to tolerate AprilTag noise and
# imperfect IK tracking on the real robot.
_CONTACT_POS_THRESHOLD = 0.001
# Extra downward offset applied to the arm descent target relative to the perceived
# beam position. Compensates for the AprilTag z being at the beam top surface while
# the gripper needs to close around the beam body 2-3 cm lower.
_GRASP_Z_DESCENT = 0.01
# Number of consecutive cycles required before declaring beam convergence
_CONVERGENCE_HYSTERESIS = 5

# Euclidean distance threshold (metres) between paired hole transforms for convergence
HOLE_CONVERGENCE_THRESHOLD = 0.002 # 3 mm
# Weight of hole-collinearity gradient relative to beam-goal gradient
_HOLE_GRADIENT_WEIGHT = 0.1
# Multiplier applied to the yaw (sin/cos) components of the beam gradient, boosting
# rotational correction relative to translational. Increase to prioritise alignment.
_YAW_GRADIENT_WEIGHT = 0.8


class ParticleTrajectories:
    def __init__(self):
        self.particles: torch.Tensor 
        self.gripper_particles: torch.Tensor
        self.indices: torch.Tensor
        self.no_live_particles = torch.Tensor
        self.loss = torch.Tensor
        
    def sample_indices(self) -> List[int]:
        traj_len = self.particles.shape[1]
        
        indices = [0] * traj_len
        particle_idx = np.random.choice(self.indices[-1, :])
        
        for k in range(traj_len - 1, -1, -1):
            particle_idx = self.indices[k, particle_idx].item()
            indices[k] = particle_idx
        
        return indices            
    
    def sample_trajectories(self, indices) -> torch.Tensor:
        _, time_steps, no_features = self.particles.shape
        particles = torch.zeros([time_steps, no_features], dtype=self.particles.dtype).to(self.particles.device)
        gripper = torch.zeros([time_steps, self.gripper_particles.shape[2]], dtype=self.particles.dtype).to(self.particles.device)
        for k, idx in enumerate(indices):
            particles[k, :] = self.particles[idx, k, :]
            gripper[k, :] = self.gripper_particles[idx, k, :]
        return particles, gripper
    
class StateParams:
    def __init__(self, state_dim:int, no_beams:int, no_hands:int, no_pins:int, device:torch.device, tol: float):
        self.device = device
        self.no_hands = no_hands
        self.no_beams = no_beams
        self.no_pins = no_pins
        self.state_dim = state_dim
        self.tol = tol


class DualArmStates:
    def __init__(self, params: StateParams):
        self.params = params
        
        # Hand poses
        self.left_pose = torch.zeros(self.params.state_dim).requires_grad_(True).to(self.params.device)
        self.right_pose = torch.zeros(self.params.state_dim).requires_grad_(True).to(self.params.device)
        
        # Hand start poses
        self.left_start = torch.zeros(self.params.state_dim).requires_grad_(True).to(self.params.device)
        self.right_start = torch.zeros(self.params.state_dim).requires_grad_(True).to(self.params.device)
        
        # Beam poses
        self._beam_poses = torch.zeros(self.params.no_beams * self.params.state_dim).view(-1, self.params.state_dim).requires_grad_(True).to(self.params.device)
        
        # Beam goals
        self._beam_goal = torch.zeros(self.params.no_beams * self.params.state_dim).view(-1, self.params.state_dim).to(self.params.device)
        
        # Pregrasp locations
        self._pregrasp = torch.zeros(self.params.no_beams * self.params.state_dim).view(-1, self.params.state_dim).requires_grad_(True).to(self.params.device)
        
        # Pregrasp goals
        self._pregrasp_goal = torch.zeros(self.params.no_beams * self.params.state_dim).view(-1, self.params.state_dim).to(self.params.device)

        self.offset = torch.tensor([0, 0, 0.16, 0, 0], dtype=torch.float32)
        self.grasp_offset = self.offset.repeat(self.params.no_beams).view(-1, self.params.state_dim).to(self.params.device)
        
    def initialise(self):
        with torch.no_grad():
            self.pregrasp = self.beam_poses.view(self.pregrasp.shape).clone().detach() + self.grasp_offset.view(self.pregrasp.shape)
        self.pregrasp.requires_grad_(True)
        return        
        
    def advance(self):
        # Update the forward dynamics of the states
        with torch.no_grad():
            self.pregrasp.view(self.params.no_beams, -1)[:, 0:2] = self.beam_poses.view(self.params.no_beams, -1)[:, 0:2].clone().detach()
            self.pregrasp.view(self.params.no_beams, -1)[:, 3:] = self.beam_poses.view(self.params.no_beams, -1)[:, 3:].clone().detach()
        return 
    
    def requires_grad(self):
        self.left_pose.requires_grad_(True)
        self.right_pose.requires_grad_(True)
        self.beam_poses.requires_grad_(True)
        self.pregrasp.requires_grad_(True)
        return
        
    
    def detach(self):
        self.left_pose = self.left_pose.detach()
        self.right_pose = self.right_pose.detach()
        self.beam_poses = self.beam_poses.detach()
        self.pregrasp = self.pregrasp.detach()
        return
    
    # Helper funcs
    def pose_quat_to_state(self, pose_quat_xyzw: np.array):
        rows, _ = pose_quat_xyzw.shape
        state = torch.zeros((rows, self.params.state_dim), dtype=torch.float32).to(self.params.device)
        
        for k in range(rows):
            state[k, 0:3] = torch.from_numpy(pose_quat_xyzw[k, 0:3])
            
            rot = R.from_quat(pose_quat_xyzw[k, 3:])
            
            state[k, 3] = torch.sin(torch.tensor(rot.as_euler(seq="xyz")[2], dtype=torch.float32).to(self.params.device))
            state[k, 4] = torch.cos(torch.tensor(rot.as_euler(seq="xyz")[2], dtype=torch.float32).to(self.params.device))
            
        return state
    
    def state_to_pose_quat(self, state: torch.tensor):
        rows, _ = state.shape
        pose_quat = np.zeros((rows, 7))

        for k in range(rows):
            pose_quat[k, 0:3] = state[k, 0:3].cpu().detach().numpy()

            rot = R.from_euler("xyz", [0, 0, np.arctan2(state[k, 3].cpu().detach().numpy(), state[k, 4].cpu().detach().numpy())])
            pose_quat[k, 3:] = rot.as_quat()

        return pose_quat

    
    # Beam Poses setters and getters
    @property
    def beam_poses(self):
        return self._beam_poses
    
    @beam_poses.setter
    def beam_poses(self, value):
        self._beam_poses = value.view(-1, self.params.state_dim)
    
    # Beam Goals setters and getters
    @property
    def beam_goal(self):
        return self._beam_goal
    
    @beam_goal.setter
    def beam_goal(self, value):
        self._beam_goal = value.view(-1, self.params.state_dim)
    
    # Pregrasp Poses setters and getters
    @property
    def pregrasp(self):
        return self._pregrasp
    
    @pregrasp.setter
    def pregrasp(self, value):
        self._pregrasp = value.view(-1, self.params.state_dim)        
    
    # Pregrasp Goal setters and getters
    @property
    def pregrasp_goal(self):
        self._pregrasp_goal = self.beam_poses.view(self._pregrasp_goal.shape) + self.grasp_offset.view(self._pregrasp_goal.shape)
        return self._pregrasp_goal
    
    @pregrasp_goal.setter
    def pregrasp_goal(self, value):
        self._pregrasp_goal = value.view(-1, self.params.state_dim)
        return
    

class DualArmGradients:
    def __init__(self, params: StateParams):
        self.params = params
        
        # Hand poses
        self.left_pose = torch.zeros(self.params.state_dim).to(self.params.device)
        self.right_pose = torch.zeros(self.params.state_dim).to(self.params.device)
        
        # Beam poses
        self.beam_poses = torch.zeros(self.params.no_beams * self.params.state_dim).view(-1, self.params.state_dim).to(self.params.device)
        
        # Pregrasp locations
        self.pregrasp = torch.zeros(self.params.no_beams * self.params.state_dim).view(-1, self.params.state_dim).to(self.params.device)
        
    def zero(self):
        self.left_pose *= 0.
        self.right_pose *= 0.
        self.beam_poses *= 0.
        self.pregrasp *= 0.
        return

class LossesContacts:
    def __init__(self, params: StateParams):
        self.params = params
        
        self._mse_sum = nn.MSELoss(reduction="sum")
        self._mse_none = nn.MSELoss(reduction='none')
        
    def calc_losses(self, states: DualArmStates):
        pass
    
    def calc_prob(self):
        pass
        
        
class HandLossesContacts(LossesContacts):
    def __init__(self, params):
        super().__init__(params)
        
        self._pregrasp_con = torch.zeros(self.params.no_hands * self.params.no_beams).to(self.params.device)
        self._pregrasp_loss = torch.zeros(self.params.no_hands * self.params.no_beams).to(self.params.device)
        # Horizontal-only (x, y, sin, cos) pregrasp loss used for the pregrasp gate.
        # Excludes z so that a descending arm is not pulled back up to pregrasp height.
        self._pregrasp_horiz_loss = torch.zeros(self.params.no_hands * self.params.no_beams).to(self.params.device)
        
        self._beam_con = torch.zeros(self.params.no_hands * self.params.no_beams).to(self.params.device)
        self._beam_loss = torch.zeros(self.params.no_hands * self.params.no_beams).to(self.params.device)
        self._beam_pos_loss = torch.zeros(self.params.no_hands * self.params.no_beams).to(self.params.device)
        self._gripper_con = torch.zeros(self.params.no_hands * self.params.no_beams).to(self.params.device)
        self._gripper_loss = torch.zeros(self.params.no_hands * self.params.no_beams).to(self.params.device)
        self._beam_loss_none = torch.zeros((self.params.no_hands * self.params.no_beams, self.params.state_dim)).to(self.params.device)
        
        self._start_loss = torch.zeros(self.params.no_hands * self.params.no_beams).to(self.params.device)
        
        # Beam target losses
        self._beam_conver_loss = torch.zeros(self.params.no_beams).to(self.params.device)
        self._beam_conver_p = torch.zeros(self.params.no_beams).to(self.params.device)
        # Pregrasp target losses
        self._pregrasp_conver_loss = torch.zeros(self.params.no_beams).to(self.params.device)
        self._pregrasp_conver_p = torch.zeros(self.params.no_beams).to(self.params.device)

    def check_gripper_state(self, states: DualArmStates):
        # Uses position-only loss populated by calc_losses
        self._gripper_con = (self._beam_pos_loss < _CONTACT_POS_THRESHOLD).type(torch.float32)
        return self._gripper_con
        
    def calc_losses(self, states: DualArmStates):
        self._pregrasp_loss[0:self.params.no_beams] = self._mse_none(states.pregrasp.view(self.params.no_beams, -1),
                                            states.left_pose.repeat(self.params.no_beams, 1)).sum(dim=1)
        self._pregrasp_loss[self.params.no_beams: 2 * self.params.no_beams] = self._mse_none(states.pregrasp.view(self.params.no_beams, -1),
                                            states.right_pose.repeat(self.params.no_beams, 1)).sum(dim=1)

        # Horizontal-only (x, y, sin, cos — dims 0,1,3,4) pregrasp loss for the descent gate.
        # Excludes z so the arm can descend through the full 16 cm hover without flipping the gate.
        _horiz_idx = [0, 1, 3, 4]
        pregrasp_horiz = states.pregrasp.view(self.params.no_beams, -1)[:, _horiz_idx]
        self._pregrasp_horiz_loss[0:self.params.no_beams] = self._mse_none(
            pregrasp_horiz, states.left_pose[_horiz_idx].repeat(self.params.no_beams, 1)).sum(dim=1)
        self._pregrasp_horiz_loss[self.params.no_beams: 2 * self.params.no_beams] = self._mse_none(
            pregrasp_horiz, states.right_pose[_horiz_idx].repeat(self.params.no_beams, 1)).sum(dim=1)
        
        # Grasp target: beam position shifted down by _GRASP_Z_DESCENT so the arm
        # descends below the AprilTag surface z to the actual graspable part of the beam.
        grasp_target = states.beam_poses.view(self.params.no_beams, -1).detach().clone()
        grasp_target[:, 2] = grasp_target[:, 2] - _GRASP_Z_DESCENT

        self._beam_loss[0:self.params.no_beams] = self._mse_none(
            grasp_target, states.left_pose.repeat(self.params.no_beams, 1)).sum(dim=1)
        self._beam_loss[self.params.no_beams: 2 * self.params.no_beams] = self._mse_none(
            grasp_target, states.right_pose.repeat(self.params.no_beams, 1)).sum(dim=1)

        self._start_loss[0:self.params.no_beams] = self._mse_none(states.beam_poses.view(self.params.no_beams, -1),
                                            states.left_start.repeat(self.params.no_beams, 1)).sum(dim=1)
        self._start_loss[self.params.no_beams: 2 * self.params.no_beams] = self._mse_none(states.beam_poses.view(self.params.no_beams, -1),
                                            states.right_start.repeat(self.params.no_beams, 1)).sum(dim=1)

        # Check which means are at target locations
        self._beam_conver_loss = self._mse_none(states.beam_poses, states.beam_goal).sum(1)
        self._pregrasp_conver_loss = self._mse_none(states.pregrasp, states.pregrasp_goal).sum(1)

        # Position-only loss for contact/gripper detection — uses the same shifted grasp
        # target so the gripper triggers when the arm reaches the actual grasp height.
        self._beam_pos_loss[0:self.params.no_beams] = self._mse_none(
            grasp_target[:, 0:3], states.left_pose[:3].repeat(self.params.no_beams, 1)).sum(dim=1)
        self._beam_pos_loss[self.params.no_beams: 2 * self.params.no_beams] = self._mse_none(
            grasp_target[:, 0:3], states.right_pose[:3].repeat(self.params.no_beams, 1)).sum(dim=1)
        return
    
    def calc_prob(self):
        # Gate on horizontal-only loss so z-descent does not flip the contact mode.
        self._pregrasp_con = (self._pregrasp_horiz_loss < 0.01).type(torch.float32)
        # Which beams are in contact (position-only threshold ~5cm per axis)
        self._beam_con = (self._beam_pos_loss < _CONTACT_POS_THRESHOLD).type(torch.float32)

        # Check which beams are converged
        self._beam_conver_p = (self._beam_conver_loss < self.params.tol).type(torch.float32)
        self._pregrasp_conver_p = (self._pregrasp_conver_loss < self.params.tol).type(torch.float32)
        # NOTE: contact is NOT released here based on pose convergence alone.
        # When USE_HOLE_CONVERGENCE=True, beam contact is released only after confirmed
        # hole-based convergence (handled by DualAssembly.gradients()).
        return
    
    
class BeamLosses(LossesContacts):
    def __init__(self, params):
        super().__init__(params)        
        self._beam_losses = torch.zeros(self.params.no_beams * self.params.state_dim).to(self.params.device)
        
    def calc_losses(self, states):
        self._beam_losses = self._mse_sum(states.beam_poses, states.beam_goal)
        return
        
class PregraspLosses(LossesContacts):
    def __init__(self, params):
        super().__init__(params)
        
        self.pregrasp_loss = torch.zeros(self.params.no_hands * self.params.no_beams).to(self.params.device)
        
    def calc_losses(self, states):
        self.pregrasp_loss = self._mse_sum(states.pregrasp, states.pregrasp_goal)
        return    
    

class DualAssembly(TrajOptBase):
    def __init__(self,
                params: TrajOptParams,
                state_params: StateParams,
                sim: SquareRobotSim,
                left_start: torch.Tensor,
                right_start: torch.Tensor,
                model: mujoco.MjModel,
                data: mujoco.MjData,
                hole_pairs: List[tuple] = None):
        
        super().__init__(params)
        
        self._state_params = state_params
        self.state_dim = self._state_params.state_dim
        
        # States
        self._states = DualArmStates(self._state_params)
        # Gradients 
        self._gradients = DualArmGradients(self._state_params)
        
        # Losses
        self._hand_losses = HandLossesContacts(self._state_params)
        self._beam_losses = BeamLosses(self._state_params)
        self._pregrasp_losses = PregraspLosses(self._state_params)
        
        self.sim = sim
        
        self._left_index = None
        self._right_index = None
        
        if sim is not None:
            self.node_names = list(self.sim._geom_to_name.values())
        
        self.left_loss = None
        self.right_loss = None
        
        self.left_start = left_start
        self.right_start = right_start
        
        # Noise weights 
        self.w = 0.0 * torch.tensor([0.01, 0.01, 0.01, 0.2, 0.2], dtype=torch.float32).to(self.params.device)
        
        # Convergence dict
        self._convergence = {}
        self._convergence_counter = {}
        self._active_pair_idx: int = None

        # Hard z floor for both arms.  Set to the table surface height plus any
        # safety margin so the planner can never drive the arms underground.
        # Set to None to disable the limit.
        self.arm_z_floor: float = 0.69

        # Missed-grasp detection.
        # _contact_hold_counter: set to _CONTACT_HOLD_CYCLES when the arm has
        #   contact, decremented each cycle otherwise.  This gives a multi-cycle
        #   window to detect a slip even though gradient steps are small and
        #   _beam_con drops quickly once the arm starts drifting.
        # _prev_arm_beam_idx: beam index each arm was targeting last cycle.
        #   Reset to 0 when the beam assignment changes so we never false-fire
        #   on a newly-assigned beam.
        self._contact_hold_counter = [0, 0]
        self._prev_arm_beam_idx    = [None, None]
        self._missed_grasp_count   = 0

        # Hole-based convergence
        # _hole_positions: (no_beams, 3) world-frame XYZ of each beam's hole; None until first
        # call to set_hole_positions()
        self._hole_positions: np.ndarray = None
        self._hole_pairs: List[tuple] = hole_pairs if hole_pairs is not None else []

        # Mujoco pointers
        self._mu_model = model
        self._mu_data = data
        
        # Containers for solutions
        self._particle_trajectories = None        
        
    def initialise(self, states: DualArmStates):
        self._states = states
        # Create container
        self._particle_trajectories = ParticleTrajectories()
        # Allocate container
        self._particle_trajectories.particles = torch.zeros([self.params.no_particles, self.params.no_steps, self._x.shape[0]], dtype=torch.float32).to(self._x.device)
        self._particle_trajectories.gripper_particles = torch.zeros([self.params.no_particles, self.params.no_steps, self._state_params.no_hands], dtype=torch.float32).to(self._x.device)
        self._particle_trajectories.indices = torch.zeros([self.params.no_steps, self.params.no_particles], dtype=torch.int32)
        self._particle_trajectories.no_live_particles = torch.zeros([self.params.no_steps], dtype=torch.int32)
        self._particle_trajectories.loss = torch.zeros([self.params.no_particles, self.params.no_steps], dtype=torch.float32)
        # Weights
        self._weights = np.ones([self.params.no_particles])
        return      
    
    def reset(self):
        self._particle_trajectories.particles.zero_()
        self._particle_trajectories.indices.zero_()
        self._particle_trajectories.no_live_particles.zero_()
        self._particle_trajectories.loss.zero_()
        self._weights = np.ones([self.params.no_particles])
        return

    def set_hole_positions(self, positions: np.ndarray) -> None:
        """Update world-frame hole positions each planning cycle.

        Args:
            positions: float array of shape (no_beams, 3) — XYZ position of each
                       beam's hole transform in the robot base frame.
        """
        self._hole_positions = positions

    def _beam_converged_by_holes(self, beam_idx: int) -> bool:
        """Return True if the hole pair that includes *beam_idx* is within
        HOLE_CONVERGENCE_THRESHOLD of each other.

        Returns False if hole positions have not been set, or if beam_idx does
        not appear in any defined hole pair (treating it as never convergeable
        via holes).
        """
        if self._hole_positions is None:
            return False
        for idx_a, idx_b in self._hole_pairs:
            if beam_idx in (idx_a, idx_b):
                dist = float(np.linalg.norm(
                    self._hole_positions[idx_a] - self._hole_positions[idx_b]))
                return dist < HOLE_CONVERGENCE_THRESHOLD
        return False
        
    
    def _select_pair(self) -> int:
        """Return the index into self._hole_pairs for the best unconverged pair.

        Picks the pair whose holes are currently closest (smallest Euclidean
        distance), so whichever pair the perception says is nearly assembled
        gets priority.  Falls back to summed beam-goal loss when hole positions
        are not yet available (e.g. in sim mode).
        """
        best_idx = None
        best_score = float('inf')
        for pair_idx, (a, b) in enumerate(self._hole_pairs):
            if a in self._convergence and b in self._convergence:
                continue
            if self._hole_positions is not None:
                score = float(np.linalg.norm(
                    self._hole_positions[a] - self._hole_positions[b]))
            else:
                score = float(self.active_left_loss[a] + self.active_right_loss[b])
            if score < best_score:
                best_score = score
                best_idx = pair_idx
        return best_idx

    # Arm must drift this far from the beam (pos-only MSE) to trigger a missed-grasp reset.
    # Kept deliberately small so the window between _beam_con dropping and detection is tight.
    _MISSED_GRASP_LOSS_THRESHOLD = _CONTACT_POS_THRESHOLD * 3.0  # 0.003 — ~5 cm 3-D distance

    # How many cycles the "recently had contact" window stays open after _beam_con drops.
    # Gradient steps are small (~0.001 pos-loss per step), so we need ~10-20 cycles for
    # _beam_pos_loss to climb from 0 → threshold after the grip is lost.
    _CONTACT_HOLD_CYCLES = 20

    def _check_missed_grasp(self) -> None:
        """Detect a missed grasp and reset pregrasp to force the arm to rise before re-gripping.

        Uses a sticky hold counter so the detection window spans multiple planning cycles.
        _beam_con drops almost immediately after contact is lost (threshold 0.001), but the
        arm drifts slowly, so we need ~10+ cycles for _beam_pos_loss to exceed the detection
        threshold.  The counter keeps the window open long enough to catch the slip.

        False-positive guards:
          - Skip if the beam is already marked converged (successful placement).
          - Reset the counter whenever the arm is re-assigned to a different beam.
        """
        no_beams = self._state_params.no_beams
        for arm_side in range(2):
            prev_idx = self._prev_arm_beam_idx[arm_side]
            if prev_idx is None:
                self._contact_hold_counter[arm_side] = 0
                continue

            loss_idx = arm_side * no_beams + prev_idx
            current_loss    = self._hand_losses._beam_pos_loss[loss_idx].item()
            currently_in_contact = self._hand_losses._beam_con[loss_idx].item() > 0.5

            # Refresh or decay the hold counter.
            if currently_in_contact:
                self._contact_hold_counter[arm_side] = self._CONTACT_HOLD_CYCLES
            else:
                self._contact_hold_counter[arm_side] = max(
                    0, self._contact_hold_counter[arm_side] - 1)

            # Skip if beam was successfully placed — this is not a miss.
            if prev_idx in self._convergence:
                continue

            # Detect: recently had contact AND arm has now drifted away from beam.
            if (self._contact_hold_counter[arm_side] > 0
                    and current_loss > self._MISSED_GRASP_LOSS_THRESHOLD):
                with torch.no_grad():
                    self._states.pregrasp[prev_idx] = (
                        self._states.beam_poses.view(no_beams, -1)[prev_idx].detach()
                        + self._states.grasp_offset[prev_idx].detach()
                    )
                self._contact_hold_counter[arm_side] = 0
                self._missed_grasp_count += 1
                print(
                    f"[MISSED GRASP] {'left' if arm_side == 0 else 'right'} arm lost beam {prev_idx}"
                    f" (loss={current_loss:.4f})"
                    " — resetting pregrasp"
                )

    def optimise(self) -> ParticleTrajectories:

        # Create or reset the containers
        if self._particle_trajectories is None:
            self.initialise(self.states)
        else:
            self.reset()
            
        # Calc losses
        self._hand_losses.calc_losses(self._states)
        self._hand_losses.calc_prob()

        # Detect missed grasp before computing any gradients this cycle.
        self._check_missed_grasp()

        # Set the initial particles to the states
        for i in range(self._params.no_particles):
            self._particle_trajectories.particles[i, 0, :] = self._x

            if self._left_index is not None:
                self._particle_trajectories.gripper_particles[i, 0, 0] = (self._hand_losses._beam_con[self._left_index] > 0.5).type(torch.float32)
            else:
                self._particle_trajectories.gripper_particles[i, 0, 0] = (self._hand_losses._beam_con[0:self._state_params.no_beams] > 0.5).any().type(torch.float32)
            
            if self._right_index is not None:
                self._particle_trajectories.gripper_particles[i, 0, 1] = (self._hand_losses._beam_con[self._state_params.no_beams + self._right_index] > 0.5).type(torch.float32)
            else:
                self._particle_trajectories.gripper_particles[i, 0, 1] = (self._hand_losses._beam_con[self._state_params.no_beams: 2 * self._state_params.no_beams] > 0.5).any().type(torch.float32)

           
        
        for k in range(1, self._params.no_steps, 1):
            for n in range(self._params.no_particles):
                
                self._states.requires_grad()
            
                gradients = self.gradients()
                
                # Update the states
                self._states.left_pose = self._states.left_pose - self.params.step_size * gradients.left_pose
                self._states.right_pose = self._states.right_pose - self.params.step_size * gradients.right_pose
                self._states.beam_poses = self._states.beam_poses - self.params.step_size * gradients.beam_poses
                self._states.pregrasp = self._states.pregrasp - self.params.step_size * gradients.pregrasp

                # Enforce arm z floor (hard constraint — overrides gradient).
                if self.arm_z_floor is not None:
                    with torch.no_grad():
                        self._states.left_pose[2]  = torch.clamp(self._states.left_pose[2],  min=self.arm_z_floor)
                        self._states.right_pose[2] = torch.clamp(self._states.right_pose[2], min=self.arm_z_floor)

                self._states.detach()       
                
                self._particle_trajectories.loss[n, k] = self._beam_losses._beam_losses.sum()
                
                # Apply noise to gradients
                self._x = torch.cat([self._states.left_pose.reshape(-1), 
                                    self._states.right_pose.reshape(-1), 
                                    self._states.beam_poses.reshape(-1),
                                    self._states.pregrasp.reshape(-1)], dim=0)
                
                self._x[0:(self._state_params.no_beams * 2 + self._state_params.no_hands) * self.state_dim] = self.normalise_pose(self._x[0:(self._state_params.no_beams * 2 + self._state_params.no_hands) * self.state_dim])
                # Sync normalised (z-clipped) pregrasp back to states so the next
                # gradient step starts from the clipped value.  Without this, the
                # gradient can drag pregrasp z below the floor indefinitely because
                # normalise_pose only clips _x, not _states.pregrasp.
                _pg_start = (self._state_params.no_hands + self._state_params.no_beams) * self.state_dim
                self._states.pregrasp = self._x[_pg_start:_pg_start + self._state_params.no_beams * self.state_dim].view(self._state_params.no_beams, self.state_dim).detach()
                self._states.pregrasp.requires_grad_(True)
                self._particle_trajectories.particles[n, k, :] = self._x
                # Gripper state
                self._particle_trajectories.gripper_particles[n, k, 0] = (self._hand_losses._beam_con[0:self._state_params.no_beams] > 0.5).any().type(torch.float32)
                self._particle_trajectories.gripper_particles[n, k, 1] = (self._hand_losses._beam_con[self._state_params.no_beams: 2 * self._state_params.no_beams] > 0.5).any().type(torch.float32)

                
                # Check collisions
                self.sim.decode_x(self._mu_data, self._x[0:(self._state_params.no_hands + self._state_params.no_beams) * self.state_dim].unsqueeze(0))
                mujoco.mj_step(self._mu_model, self._mu_data)
                
                # Check for collisions with moving beams
                _collided = False
                if self._left_index is not None:
                    if self.sim.check_collisions(self._mu_data, self.node_names[self._left_index]):
                        _collided = True
                if self._right_index is not None:
                    if self.sim.check_collisions(self._mu_data, self.node_names[self._right_index]):
                        _collided = True
                if _collided:
                    self._weights[n] = 1.0e-5
                else:
                    self._weights[n] = 1.0
                    self._particle_trajectories.no_live_particles[k] += 1
            
            # Resample particles
            self._weights /= self._weights.sum()
            
            indices = systematic_resample(self._weights)
            self._particle_trajectories.particles[:, k, :] = self._particle_trajectories.particles[indices, k, :]
            self._particle_trajectories.gripper_particles[:, k, :] = self._particle_trajectories.gripper_particles[indices, k, :]
            self._particle_trajectories.indices[k] = torch.tensor(indices)

        # Update per-arm beam tracking for next cycle's missed-grasp detection.
        # Reset the hold counter whenever the arm is assigned to a different beam so
        # we never false-fire on a beam the arm has not yet touched.
        for arm_side, new_idx in enumerate([self._left_index, self._right_index]):
            if new_idx != self._prev_arm_beam_idx[arm_side]:
                self._contact_hold_counter[arm_side] = 0
            self._prev_arm_beam_idx[arm_side] = new_idx

        return self._particle_trajectories
    
    def _calc_hole_collinearity_loss(self) -> torch.Tensor:
        """Hole-collinearity loss using perceived hole positions as targets.

        For each beam k in a pair, the perceived hole world position is inverted
        through the current (detached) beam transform to get the hole in beam-local
        coordinates.  That local offset is then re-applied through the live beam
        transform so gradients flow through position and yaw.  The resulting
        predicted hole is pulled toward the partner beam's perceived hole position.

        Returns zero if hole positions have not yet been set by set_hole_positions().
        """
        if not self._hole_pairs or self._hole_positions is None:
            return torch.zeros(1, device=self.params.device)

        beam = self._states.beam_poses  # (no_beams, state_dim), requires_grad
        loss = torch.zeros(1, device=self.params.device)

        for a, b in self._hole_pairs:
            # Perceived hole positions as fixed world-frame targets (no gradient)
            hole_a = torch.tensor(self._hole_positions[a, :2], dtype=torch.float32,
                                  device=self.params.device)
            hole_b = torch.tensor(self._hole_positions[b, :2], dtype=torch.float32,
                                  device=self.params.device)

            for k, hole_world, target in ((a, hole_a, hole_b), (b, hole_b, hole_a)):
                # Detached current beam transform (R^{-1} = [[cos, sin], [-sin, cos]])
                x_k = beam[k, 0].detach()
                y_k = beam[k, 1].detach()
                c_k = beam[k, 4].detach()   # cos θ
                s_k = beam[k, 3].detach()   # sin θ

                # Express perceived hole in beam-local frame
                dx = hole_world[0] - x_k
                dy = hole_world[1] - y_k
                lx =  c_k * dx + s_k * dy  # R^{-1} · (hole − center)
                ly = -s_k * dx + c_k * dy

                # Re-apply live beam transform with gradients
                # R(θ) · local = [[cos, −sin], [sin, cos]] · [lx, ly]
                pred_x = beam[k, 0] + beam[k, 4] * lx - beam[k, 3] * ly
                pred_y = beam[k, 1] + beam[k, 3] * lx + beam[k, 4] * ly

                # Pull predicted hole toward partner's perceived hole
                loss = loss + (pred_x - target[0]) ** 2 + (pred_y - target[1]) ** 2

        return loss

    def gradients(self):
        # Calc losses
        self._hand_losses.calc_losses(self._states)
        self._hand_losses.calc_prob()
        
        left_contacts = self._hand_losses._beam_con[0:self._state_params.no_beams]
        right_contacts = self._hand_losses._beam_con[self._state_params.no_beams:2*self._state_params.no_beams]
        
        left_pregrasp_c = self._hand_losses._pregrasp_con[0:self._state_params.no_beams]
        right_pregrasp_c = self._hand_losses._pregrasp_con[self._state_params.no_beams:2 * self._state_params.no_beams]
        
        self._beam_losses.calc_losses(self._states)
        self._pregrasp_losses.calc_losses(self._states)
        
        # Calculate beam gradients (goal-position term)
        self._gradients.beam_poses = grad(outputs=self._beam_losses._beam_losses, inputs=self.states.beam_poses, retain_graph=True)[0]

        # Secondary hole-collinearity gradient — weaker pull toward configurations
        # where paired holes are coincident, independent of goal position accuracy.
        if self._hole_pairs:
            hole_loss = self._calc_hole_collinearity_loss()
            hole_grad = grad(outputs=hole_loss, inputs=self._states.beam_poses, retain_graph=True)[0]
            self._gradients.beam_poses = self._gradients.beam_poses + _HOLE_GRADIENT_WEIGHT * hole_grad
        
        self.left_loss = self._hand_losses._beam_loss[0:self._state_params.no_beams] * (left_pregrasp_c)  + \
                            self._hand_losses._pregrasp_loss[0:self._state_params.no_beams] * (1 - left_pregrasp_c)
                            
        self.right_loss = self._hand_losses._beam_loss[self._state_params.no_beams:2*self._state_params.no_beams] * (right_pregrasp_c)  + \
                            self._hand_losses._pregrasp_loss[self._state_params.no_beams: 2*self._state_params.no_beams] * (1 - right_pregrasp_c)

        # If converged return zeros for gradients
        if self.check_convergence(self._hand_losses._beam_conver_p,
                                self._hand_losses._pregrasp_conver_p,
                                self.left_loss,
                                self.right_loss):
            self._gradients.zero()
            print(" Converged ")
            return self._gradients

        # Zero contact for beams that have confirmed hole-based convergence so the
        # arm is free to move to the next task rather than continuing to push them.
        for k in self._convergence:
            self._hand_losses._beam_con[k] = 0.0
            self._hand_losses._beam_con[self._state_params.no_beams + k] = 0.0

        self._gradients.beam_poses = (self._gradients.beam_poses * self._hand_losses._beam_con[0:self._state_params.no_beams].reshape(self._gradients.beam_poses.shape[0], 1) + \
                                        self._gradients.beam_poses * self._hand_losses._beam_con[self._state_params.no_beams:2*self._state_params.no_beams].reshape(self._gradients.beam_poses.shape[0], 1))
        # Apply noises
        noise = torch.randn_like(self._gradients.beam_poses)
        self._gradients.beam_poses += min(torch.norm(self._gradients.beam_poses, p=2.0), 1.0) * noise * self.w
        self._gradients.beam_poses = (self._gradients.beam_poses * left_contacts.reshape(self._gradients.beam_poses.shape[0], 1) + self._gradients.beam_poses * right_contacts.reshape(self._gradients.beam_poses.shape[0], 1))
        
        self._gradients.beam_poses[:, 3:5] *= _YAW_GRADIENT_WEIGHT
        
        # Calculate the gradients for the pregrasp pose
        self._gradients.pregrasp = grad(outputs=self._pregrasp_losses.pregrasp_loss, inputs=self._states.pregrasp, retain_graph=True)[0]
        
        # Left gradients
        if self._left_index is not None:
            self._gradients.left_pose = grad(self.left_loss[self._left_index], inputs=self.states.left_pose, retain_graph=True)[0] * (1. - left_contacts[self._left_index]) + \
                                        self._gradients.beam_poses[self._left_index, :] * left_contacts[self._left_index]
            self._gradients.pregrasp[self._left_index] = self._gradients.left_pose * left_pregrasp_c[self._left_index]

            # Only move upwards vertically
            if self._gradients.left_pose[2] < -0.15:
                self._gradients.left_pose[0:2] *= 0.0
                self._gradients.left_pose[3:] *= 0.0

        else:
            self._gradients.left_pose *= 0.0

        # Right gradients
        if self._right_index is not None:
            self._gradients.right_pose = grad(self.right_loss[self._right_index], inputs=self.states.right_pose, retain_graph=True)[0] * (1. - right_contacts[self._right_index]) + \
                                        self._gradients.beam_poses[self._right_index, :] * right_contacts[self._right_index]
            self._gradients.pregrasp[self._right_index] = self._gradients.right_pose * right_pregrasp_c[self._right_index]

            # Only move upwards vertically
            if self._gradients.right_pose[2] < -0.15:
                self._gradients.right_pose[0:2] *= 0.0
                self._gradients.right_pose[3:] *= 0.0

        else:
            self._gradients.right_pose *= 0.0            
        
        return self._gradients
    
    def check_convergence(self, beam_conv_p, pregrasp_conv_p, left_loss, right_loss):
                
        # Losses
        self.active_left_loss = self._hand_losses._start_loss[0: self._state_params.no_beams].clone()
        self.active_right_loss = self._hand_losses._start_loss[self._state_params.no_beams: 2 * self._state_params.no_beams].clone()
        
        # Pin penalties
        pin_indices = list(2 * k + 1 for k in range(self._state_params.no_pins))
        
        self.active_left_loss[pin_indices] *= 10.0
        self.active_right_loss[pin_indices] *= 10.0
        
        # Indices        
        left_dict = {}
        right_dict = {}
        
        # Map each beam to its hole_pairs index
        beam_to_pair = {}
        for pair_idx, (a, b) in enumerate(self._hole_pairs):
            beam_to_pair[a] = pair_idx
            beam_to_pair[b] = pair_idx

        # Re-select the active pair whenever it has just fully converged or on first call
        active_converged = (
            self._active_pair_idx is not None and
            all(k in self._convergence for k in self._hole_pairs[self._active_pair_idx])
        )
        if self._active_pair_idx is None or active_converged:
            self._active_pair_idx = self._select_pair()

        # Check if beams are in the goal location
        for k in range(self._state_params.no_beams):
            if self._beam_converged_by_holes(k):
                self._convergence_counter[k] = self._convergence_counter.get(k, 0) + 1
                if self._convergence_counter[k] >= _CONVERGENCE_HYSTERESIS:
                    self._convergence[k] = True
            else:
                self._convergence_counter[k] = 0

        # Check if no tasks to do -- converged
        if self._active_pair_idx is None:
            return True

        # Assign BOTH beams of the active pair to the two arms via 2×2 cost
        # assignment.  Converged beams stay assigned so the holding arm keeps
        # gripping while waiting for its partner — we never release a placed
        # beam to chase another pair.
        a, b = self._hole_pairs[self._active_pair_idx]
        cost_ab = float(self.active_left_loss[a]) + float(self.active_right_loss[b])
        cost_ba = float(self.active_left_loss[b]) + float(self.active_right_loss[a])
        if cost_ab <= cost_ba:
            self._left_index, self._right_index = a, b
        else:
            self._left_index, self._right_index = b, a

        return False
    
    def normalise_pose(self, pose_torch: torch.tensor):
        no_items = pose_torch.shape[0] // self.state_dim
        for k in range(no_items):
            z_min = 0.021
            # Apply the tighter arm z floor to the arm elements (first no_hands items).
            if self.arm_z_floor is not None and k < self._state_params.no_hands:
                z_min = max(z_min, self.arm_z_floor)
            pose_torch[self.state_dim * k + 2] = max(pose_torch[self.state_dim * k + 2], z_min)
            pose_torch[self.state_dim * k + 3: self.state_dim * k + 5] = \
                torch.nn.functional.normalize(pose_torch[self.state_dim * k + 3: self.state_dim * k + 5], dim=0)
        return pose_torch
    
    @property
    def particle_trajectories(self):
        if self._x is not None:
            self.initialise()
        else:
            print(" Please set_x to first to allocate data. ")
    
    @property
    def states(self):
        return self._states
    
    @states.setter
    def states(self, state_values: DualArmStates):
        self._states = state_values
        self._states.advance()
        # Update x
        self._x = torch.cat([state_values.left_pose.reshape(-1), 
                            state_values.right_pose.reshape(-1), 
                            state_values.beam_poses.reshape(-1), 
                            state_values.pregrasp.reshape(-1)], dim=0)
        return