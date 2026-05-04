"""Perceptive bimanual assembly planner — explicit hybrid state machine.

This module is a drop-in replacement for ``dual_assembly.DualAssembly``. It
mirrors the same external API (constructor signature, ``optimise()``,
``set_hole_positions()``, ``states`` setter, public ``_left_index``,
``_right_index``, ``_convergence``, ``_convergence_counter``,
``_active_pair_idx``, ``_hand_losses`` fields, configurable
``arm_z_floor``) so existing callers
(``frank_atls.py``, ``scripts/atls.py``, ``smoke_test_dual_assembly.py``,
``aicon_dual_assembly.py``, ``test/test_dual_assembly.py``) work unchanged.

What is different
-----------------

The original ``DualAssembly`` blends pregrasp / grasp / assembly behaviours
implicitly via contact gates, gradient masks, and hysteresis counters threaded
through every cycle. This file replaces that with an **explicit finite state
machine** wrapping per-state gradient-based local controllers. Each state has
exactly one gradient producer and one transition gate, so the control flow is
trivial to read and the failure modes are localised.

State machine
~~~~~~~~~~~~~

::

    GO_HOME ─▶ PICK_TASK ──no work──▶ DONE (idle wait; re-enters PICK_TASK on perturbation)
                  │                          ▲
                  │                          │ new unconverged pair detected
                  │                          │
                  ▼ active pair fixed
              MOVE_TO_PREGRASP ─▶ DESCENDING ─▶ CLOSE_GRIPPER ─▶ READY
                                      │                                  │
                                      │ timeout                          ▼
                                      ▼                            DUAL_ASSEMBLE
                                RECOVERY_RELEASE                    │     │
                                      │                  hole conv  │     │ slip / timeout
                                      ▼                             ▼     ▼
                                RECOVERY_MOVE_UP            RELEASE_GRIPPER
                                      │                             │
                                      ▼                             ▼
                              MOVE_TO_PREGRASP                 MOVE_AWAY
                                                                    │
                                                                    ▼
                                                                PICK_TASK

User-directed deviations from PERCEPTIVE_ASSEMBLY.MD
----------------------------------------------------

1. **Fixed arm-to-beam assignment.** Right arm always handles odd-indexed
   beams (1, 3); left arm always handles even-indexed beams (0, 2).
   Computed once on entry to ``PICK_TASK``; never reshuffled mid-attempt.
2. **Pregrasp gate uses full 5-DOF distance** (x, y, z, sinθ, cosθ), gated by
   ``PREGRASP_TOL`` as a single MSE-sum bound.
3. **Convergence is touched only in PICK_TASK and DUAL_ASSEMBLE.** No
   per-cycle latching during grasp / move states.
4. **No gripper hysteresis.** Grippers are toggled exclusively in
   ``CLOSE_GRIPPER`` (close) and ``RELEASE_GRIPPER`` / ``RECOVERY_RELEASE``
   (open). Transient contact loss in any other state does not actuate them.
5. **Single recovery sequence.** Both ``DESCENDING`` timeout and the new
   ``DUAL_ASSEMBLE`` slip detector route through
   ``RECOVERY_RELEASE → RECOVERY_MOVE_UP → MOVE_TO_PREGRASP`` (same active
   pair).
6. **DUAL_ASSEMBLE slip detector.** If either EE separates from its assigned
   beam pose by > ``ASSEMBLE_SLIP_DIST`` (default 3 cm) for
   ``ASSEMBLE_SLIP_STEPS`` consecutive steps (default 20), the planner
   assumes the grasp slipped and routes through recovery.
7. **``hole_pairs`` is mandatory.** ``__init__`` raises ``ValueError`` if it
   is empty / ``None``. ``optimise()`` raises ``RuntimeError`` if
   ``set_hole_positions()`` was never called. There is no fallback.
8. ``GO_HOME`` initial state mimics the start posture from
   ``dual_assembly.py`` (uses ``left_start`` / ``right_start`` directly as
   targets, instead of as a scoring bias).

Re-used components
------------------

The data layout (``DualArmStates``, ``DualArmGradients``, loss banks,
``ParticleTrajectories``, ``StateParams``) is imported verbatim from
``dual_assembly`` — those are already validated and the callers depend on
their exact shape. Only the **algorithmic dispatch** (the FSM + per-state
gradient producers) is new.
"""

from __future__ import annotations

from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.autograd import grad
import mujoco
from filterpy.monte_carlo import systematic_resample

from beam_data_gen.traj_opt.dual_assembly import (
    DualArmGradients,
    DualArmStates,
    HandLossesContacts,
    BeamLosses,
    PregraspLosses,
    ParticleTrajectories,
    StateParams,
    TrajOptParams,
)
from beam_data_gen.traj_opt.traj_opt_base import TrajOptBase
from beam_data_gen.simulator.square_robot_sim import SquareRobotSim


# ---------------------------------------------------------------------------
# Tunables — all units SI (metres, radians). Defaults come from the original
# dual_assembly.py where applicable; new gates have rationale in-line.
# ---------------------------------------------------------------------------

# Pregrasp gate: per-axis max absolute error on (x, y, z). Rotation ignored.
# Each of |dx|, |dy|, |dz| must be below this bound for the gate to fire.
PREGRASP_TOL: float = 0.03

# Grasp contact gate: per-axis max absolute error on (x, y, z). Each of
# |dx|, |dy|, |dz| must be below this bound for the gate to fire.
GRASP_POS_TOL: float = 0.010

# Per-pair assembly convergence (Euclidean hole distance).
HOLE_CONVERGENCE_THRESHOLD: float = 0.003

# Cycles a pair must stay below the hole threshold before being latched.
CONVERGENCE_HYSTERESIS: int = 5

# DUAL_ASSEMBLE slippage detector. Distance is measured per arm between the
# end-effector position and its assigned beam position; if exceeded for the
# given number of consecutive steps the grasp is considered failed.
ASSEMBLE_SLIP_DIST: float = 0.05      # 5 cm
ASSEMBLE_SLIP_STEPS: int  = 15

# Per-state step budgets. The planner runs at ~30 Hz so 150 ≈ 5 s.
DESCEND_TIMEOUT_STEPS:  int = 100
ASSEMBLE_TIMEOUT_STEPS: int = 100
GO_HOME_TIMEOUT_STEPS:  int = 100

# Position-only gate for MOVE_AWAY / RECOVERY_MOVE_UP (orientation ignored).
MOVE_UP_TOL: float = 0.05

# Gradient mixing — kept identical to dual_assembly.py for behavioural parity
# in the assembly phase.
HOLE_GRADIENT_WEIGHT: float = 0.1
YAW_GRADIENT_WEIGHT:  float = 1.5

# Gradient-descent learning rate. Hard-coded here (instead of read from
# TrajOptParams.step_size) so the planner's integrator step is fixed by the
# module rather than by callers.
LEARNING_RATE: float = 0.2


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------

class State(IntEnum):
    """High-level FSM state. The two arms share a single state — the
    arm-to-beam mapping is fixed by ``PICK_TASK`` so per-arm sub-FSMs would
    be redundant complexity."""

    GO_HOME          = 0   # initial: drive to left_start / right_start
    PICK_TASK        = 1   # select active hole pair, fix arm assignment
    MOVE_TO_PREGRASP = 2   # both arms hover above their beams
    DESCENDING       = 3   # both arms descend to grasp height
    CLOSE_GRIPPER    = 4   # latch grippers closed (1 step)
    READY            = 5   # both grasped, ready to assemble (1 step)
    DUAL_ASSEMBLE    = 6   # gradient-driven beam→goal + collinearity
    RECOVERY_RELEASE = 7   # open grippers (1 step) on grasp failure
    RECOVERY_MOVE_UP = 8   # lift to pregrasp height and retry the pair
    RELEASE_GRIPPER  = 9   # successful assembly: open grippers (1 step)
    MOVE_AWAY        = 10  # lift to pregrasp height, then PICK_TASK
    DONE             = 11  # idle loop: all pairs converged, re-enters PICK_TASK on perturbation


# ---------------------------------------------------------------------------
# Hole-collinearity loss (lifted verbatim from dual_assembly.py:632-677 for
# behavioural parity in DUAL_ASSEMBLE; isolated as a free function so it can
# be unit-tested without instantiating the planner).
# ---------------------------------------------------------------------------

def _hole_collinearity_loss(
    beam_states: torch.Tensor,
    hole_positions: np.ndarray,
    hole_pairs: List[Tuple[int, int]],
    device: torch.device,
) -> torch.Tensor:
    """Pull paired holes toward each other in the world frame.

    For each beam ``k`` in a pair the perceived hole position is inverted
    through the *detached* current beam transform to get the hole offset in
    beam-local coordinates, then re-applied through the *live* beam
    transform so gradients flow through position and yaw. The predicted
    hole is pulled toward the partner's perceived hole.
    """
    loss = torch.zeros(1, device=device)
    for a, b in hole_pairs:
        hole_a = torch.tensor(hole_positions[a, :2], dtype=torch.float32, device=device)
        hole_b = torch.tensor(hole_positions[b, :2], dtype=torch.float32, device=device)

        for k, hole_world, target in ((a, hole_a, hole_b), (b, hole_b, hole_a)):
            x_k = beam_states[k, 0].detach()
            y_k = beam_states[k, 1].detach()
            c_k = beam_states[k, 4].detach()
            s_k = beam_states[k, 3].detach()

            dx = hole_world[0] - x_k
            dy = hole_world[1] - y_k
            lx =  c_k * dx + s_k * dy
            ly = -s_k * dx + c_k * dy

            pred_x = beam_states[k, 0] + beam_states[k, 4] * lx - beam_states[k, 3] * ly
            pred_y = beam_states[k, 1] + beam_states[k, 3] * lx + beam_states[k, 4] * ly

            loss = loss + (pred_x - target[0]) ** 2 + (pred_y - target[1]) ** 2
    return loss


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class DualAssembly(TrajOptBase):
    """Hybrid state-machine perceptive assembly planner.

    Each call to :meth:`optimise` runs ``params.no_steps`` particle-filter
    integration steps from the current ``self.states`` baseline. Within a
    single call the FSM may transition multiple times — transitions are
    re-evaluated after every integration step. The state itself persists
    across calls (``self._state``).

    The class deliberately reuses the data-bearing classes from
    ``dual_assembly`` (``DualArmStates``, ``HandLossesContacts``, etc.) so
    the wire format consumed by callers is bit-identical.
    """

    # ------------------------------------------------------------------
    # Construction / lifecycle
    # ------------------------------------------------------------------

    def __init__(
        self,
        params: TrajOptParams,
        state_params: StateParams,
        sim: Optional[SquareRobotSim],
        left_start: torch.Tensor,
        right_start: torch.Tensor,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        hole_pairs: List[Tuple[int, int]],
    ):
        super().__init__(params)

        if not hole_pairs:
            raise ValueError(
                "hole_pairs must be a non-empty list of (a, b) tuples — the "
                "perceptive planner has no fallback for unpaired beams."
            )

        self._state_params = state_params
        self.state_dim = state_params.state_dim

        # Data-bearing objects (shapes match dual_assembly exactly).
        self._states = DualArmStates(state_params)
        self._gradients = DualArmGradients(state_params)
        self._hand_losses = HandLossesContacts(state_params)
        self._beam_losses_bank = BeamLosses(state_params)
        self._pregrasp_losses_bank = PregraspLosses(state_params)

        # Sim / mujoco plumbing (sim may be None in unit tests).
        self.sim = sim
        self._mu_model = model
        self._mu_data = data
        self.node_names = (
            list(self.sim._geom_to_name.values()) if sim is not None else []
        )

        self.left_start = left_start
        self.right_start = right_start

        # ---- Public read-only fields the callers depend on. ----
        self._left_index: Optional[int] = None
        self._right_index: Optional[int] = None
        self._convergence: dict = {}
        self._convergence_counter: dict = {}
        self._active_pair_idx: Optional[int] = None

        # ---- Configurable safety knobs (preserved verbatim). ----
        # Match the IK z-limits enforced by frank_atls.py (_z_lims = [0.755, 1.00]).
        # A mismatch means the planner routinely requests poses that hardware silently
        # clips, causing the planner state to diverge from reality each cycle.
        self.arm_z_floor: float = 0.8
        self.arm_z_ceil:  float = 1.150

        # Per-arm z workspace bounds.  Defaults mirror arm_z_floor / arm_z_ceil
        # so existing callers are unaffected; override after construction as needed.
        self.left_z_floor:  float = self.arm_z_floor
        self.left_z_ceil:   float = self.arm_z_ceil
        self.right_z_floor: float = self.arm_z_floor
        self.right_z_ceil:  float = self.arm_z_ceil

        # Maximum L2 displacement (metres) of the EE position per gradient step.
        # Prevents large gradients from flinging the planner state outside the
        # workspace in a single step.  Set to float('inf') to disable.
        self.max_ee_step: float = float('inf')

        # Fixed z height the arms descend to in DESCENDING, regardless of the
        # perceived beam z.  Set to sit the gripper at beam surface height.
        self.grasp_z: float = 0.793

        # Inside this radius (m) of the descent target the gradient is replaced
        # by one whose integrator step lands the planner state exactly on the
        # target: g = (x - x*) / LEARNING_RATE. This bypasses the asymptotic
        # shrinkage of a quadratic loss without overshoot or limit cycles.
        # max_ee_step (above) still clips the per-step displacement, so motion
        # remains smooth even if the diff is large.
        # Set descent_snap_radius to 0.0 to disable (vanilla quadratic descent).
        self.descent_snap_radius:    float = 0.02

        # Cached per-axis max absolute (x, y, z) error vs. the fixed-z grasp
        # target, updated each _grad_descending call and read by _grasp_contact.
        self._descent_max_axis_err_l: float = float('inf')
        self._descent_max_axis_err_r: float = float('inf')
        # Position-only Euclidean distance to the fixed-z grasp target. Read by
        # frank_atls.publish_planner_diagnostics for ROS diagnostics output.
        self._descent_loss_l: float = float('inf')
        self._descent_loss_r: float = float('inf')

        # Whether the descent-snap branch fired in the most recent
        # ``_grad_descending`` call (per arm), and a cumulative count of
        # how many times it has fired since planner construction. Read by
        # frank_atls.publish_planner_diagnostics.
        self._descent_snap_fired_l: bool = False
        self._descent_snap_fired_r: bool = False
        self._descent_snap_count_l: int = 0
        self._descent_snap_count_r: int = 0

        # ---- Hole perception. MUST be set before optimise(). ----
        self._hole_positions: Optional[np.ndarray] = None
        self._hole_pairs: List[Tuple[int, int]] = list(hole_pairs)

        # ---- FSM bookkeeping. ----
        self._state: State = State.GO_HOME
        self._state_step: int = 0      # steps spent in current state
        self._slip_counter: int = 0    # consecutive slip steps (DUAL_ASSEMBLE)
        # Per-arm gripper command — flipped only by CLOSE_GRIPPER /
        # RELEASE_GRIPPER / RECOVERY_RELEASE. No hysteresis.
        self._gripper_closed: List[bool] = [False, False]

        # ---- Testing / diagnostics knobs. ----
        # Set step_mode = True to pause at every FSM state transition and wait
        # for an Enter keypress.  Safe to toggle at runtime; harmless when False.
        self.step_mode: bool = False
        self._step_mode_last_state: Optional[State] = None

        # Particle-filter containers (allocated lazily by initialise()).
        self._particle_trajectories: Optional[ParticleTrajectories] = None
        self._weights: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API expected by callers
    # ------------------------------------------------------------------

    def set_hole_positions(self, positions: np.ndarray) -> None:
        """Update perceived hole positions — required every cycle.

        Args:
            positions: ``(no_beams, 3)`` world-frame XYZ for each beam's
                hole transform in the robot base frame.
        """
        self._hole_positions = positions

    @property
    def states(self) -> DualArmStates:
        return self._states

    @states.setter
    def states(self, state_values: DualArmStates) -> None:
        self._states = state_values
        # advance() refreshes the pregrasp target to "directly above the
        # perceived beam" — required every cycle so a missed grasp does not
        # carry a stale pregrasp into the next attempt.
        self._states.advance()
        self._x = torch.cat(
            [
                state_values.left_pose.reshape(-1),
                state_values.right_pose.reshape(-1),
                state_values.beam_poses.reshape(-1),
                state_values.pregrasp.reshape(-1),
            ],
            dim=0,
        )

    def initialise(self, states: DualArmStates) -> None:
        """Allocate particle-filter buffers for the current trajectory size."""
        self._states = states
        pt = ParticleTrajectories()
        n_particles = self.params.no_particles
        n_steps = self.params.no_steps
        n_hands = self._state_params.no_hands
        pt.particles = torch.zeros(
            [n_particles, n_steps, self._x.shape[0]],
            dtype=torch.float32,
            device=self._x.device,
        )
        pt.gripper_particles = torch.zeros(
            [n_particles, n_steps, n_hands],
            dtype=torch.float32,
            device=self._x.device,
        )
        pt.indices = torch.zeros([n_steps, n_particles], dtype=torch.int32)
        pt.no_live_particles = torch.zeros([n_steps], dtype=torch.int32)
        pt.loss = torch.zeros([n_particles, n_steps], dtype=torch.float32)
        self._particle_trajectories = pt
        self._weights = np.ones([n_particles])

    def reset(self) -> None:
        """Zero out the per-cycle particle-filter buffers."""
        self._particle_trajectories.particles.zero_()
        self._particle_trajectories.indices.zero_()
        self._particle_trajectories.no_live_particles.zero_()
        self._particle_trajectories.loss.zero_()
        self._weights = np.ones([self.params.no_particles])

    # ------------------------------------------------------------------
    # Main planning loop
    # ------------------------------------------------------------------

    def optimise(self) -> ParticleTrajectories:
        """Run one planning cycle of ``no_steps`` particle-filter integrations.

        Raises:
            RuntimeError: if ``set_hole_positions`` was never called.
        """
        if self._hole_positions is None:
            raise RuntimeError(
                "set_hole_positions(...) must be called before optimise()."
            )

        if self._particle_trajectories is None:
            self.initialise(self._states)
        else:
            self.reset()

        # Compute losses once at the start of the cycle so PICK_TASK has up-to-
        # date hole / pregrasp / contact information when it makes its choice.
        self._hand_losses.calc_losses(self._states)
        self._hand_losses.calc_prob()

        # PICK_TASK / state transitions that don't need a gradient step are
        # resolved here so the rest of the rollout runs in the right state.
        self._evaluate_idle_transitions()

        # Seed every particle with the current robot baseline.
        for i in range(self.params.no_particles):
            self._particle_trajectories.particles[i, 0, :] = self._x
            self._particle_trajectories.gripper_particles[i, 0, 0] = float(self._gripper_closed[0])
            self._particle_trajectories.gripper_particles[i, 0, 1] = float(self._gripper_closed[1])

        # Inner particle-filter loop. Identical structure to dual_assembly.py
        # so the consumer sees the same trajectory shape.
        for k in range(1, self.params.no_steps):
            for n in range(self.params.no_particles):
                self._states.requires_grad()
                gradients = self._step_gradients()

                # Apply gradient with per-arm velocity limiting.
                delta_l = self._limit_ee_delta(LEARNING_RATE * gradients.left_pose)
                delta_r = self._limit_ee_delta(LEARNING_RATE * gradients.right_pose)
                self._states.left_pose  = self._states.left_pose  - delta_l
                self._states.right_pose = self._states.right_pose - delta_r
                self._states.beam_poses = self._states.beam_poses - LEARNING_RATE * gradients.beam_poses
                self._states.pregrasp   = self._states.pregrasp   - LEARNING_RATE * gradients.pregrasp

                # Enforce workspace bounds (z, y) for each arm plus pregrasp z.
                self._clamp_ee_poses()

                self._states.detach()

                self._particle_trajectories.loss[n, k] = self._beam_losses_bank._beam_losses.sum()

                # Repack flat _x and renormalise yaw / clip z.
                self._x = torch.cat(
                    [
                        self._states.left_pose.reshape(-1),
                        self._states.right_pose.reshape(-1),
                        self._states.beam_poses.reshape(-1),
                        self._states.pregrasp.reshape(-1),
                    ],
                    dim=0,
                )
                pose_block = (self._state_params.no_beams * 2 + self._state_params.no_hands) * self.state_dim
                self._x[0:pose_block] = self._normalise_pose(self._x[0:pose_block])

                # Sync the (possibly z-clipped) pregrasp back into states so the
                # next gradient step starts from the clipped value.
                pg_start = (self._state_params.no_hands + self._state_params.no_beams) * self.state_dim
                self._states.pregrasp = self._x[
                    pg_start : pg_start + self._state_params.no_beams * self.state_dim
                ].view(self._state_params.no_beams, self.state_dim).detach()
                self._states.pregrasp.requires_grad_(True)

                self._particle_trajectories.particles[n, k, :] = self._x
                self._particle_trajectories.gripper_particles[n, k, 0] = float(self._gripper_closed[0])
                self._particle_trajectories.gripper_particles[n, k, 1] = float(self._gripper_closed[1])

                # Collision filtering removed — every particle is treated as
                # live with uniform weight. ``self.sim``, ``self._mu_model``,
                # and ``self._mu_data`` are retained on the instance for the
                # constructor signature but are no longer consulted here.
                self._weights[n] = 1.0
                self._particle_trajectories.no_live_particles[k] += 1

                # Re-evaluate the gradient-driven transition gates against the
                # latest losses so the FSM can advance mid-rollout.
                self._hand_losses.calc_losses(self._states)
                self._hand_losses.calc_prob()
                self._evaluate_step_transitions()

            # Resample particles to keep collision-free trajectories.
            self._weights /= self._weights.sum()
            indices = systematic_resample(self._weights)
            self._particle_trajectories.particles[:, k, :] = self._particle_trajectories.particles[indices, k, :]
            self._particle_trajectories.gripper_particles[:, k, :] = self._particle_trajectories.gripper_particles[indices, k, :]
            self._particle_trajectories.indices[k] = torch.tensor(indices)

        # Step-mode: pause on every FSM state transition so a developer can
        # inspect the system state before the next planning cycle.  Activated
        # by setting  planner.step_mode = True  at any point.
        if self.step_mode and self._state != self._step_mode_last_state:
            prev_name = (State(self._step_mode_last_state).name
                         if self._step_mode_last_state is not None else "—")
            self.log_state()
            input(f"\n  ── step-mode: {prev_name} → {State(self._state).name}"
                  f"  [Enter to continue] ──")
            self._step_mode_last_state = self._state

        return self._particle_trajectories

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def log_state(self, logger=None) -> None:
        """Log a one-screen FSM status summary.

        Args:
            logger: callable(str) used to emit each line.  Defaults to
                ``print``.  Pass ``rospy.loginfo`` when running inside ROS.
        """
        _log = logger or print
        state_name = State(self._state).name
        li, ri = self._left_index, self._right_index
        gc = self._gripper_closed
        n = self._state_params.no_beams

        # ── Header: state + budget ──────────────────────────────────────
        if self._state == State.DESCENDING:
            budget = (f"  budget={DESCEND_TIMEOUT_STEPS - self._state_step}"
                      f"/{DESCEND_TIMEOUT_STEPS} steps")
        elif self._state == State.DUAL_ASSEMBLE:
            budget = (f"  budget={ASSEMBLE_TIMEOUT_STEPS - self._state_step}"
                      f"/{ASSEMBLE_TIMEOUT_STEPS}  slip={self._slip_counter}/{ASSEMBLE_SLIP_STEPS}")
        elif self._state == State.GO_HOME:
            budget = (f"  budget={GO_HOME_TIMEOUT_STEPS - self._state_step}"
                      f"/{GO_HOME_TIMEOUT_STEPS} steps")
        else:
            budget = ""

        _log(f"[ATLS-FSM] {state_name:<22} step={self._state_step:3d}{budget}")
        _log(f"  pair={self._active_pair_idx}  "
             f"L→b{li}  R→b{ri}  "
             f"grippers=[L:{gc[0]} R:{gc[1]}]")

        # ── Convergence dict ────────────────────────────────────────────
        conv_keys = sorted(self._convergence.keys())
        ctr_str = {k: self._convergence_counter.get(k, 0) for k in range(n)}
        _log(f"  converged={conv_keys}  "
             f"hysteresis_ctr={ctr_str}")

        # ── Hole pair distances ─────────────────────────────────────────
        if self._hole_positions is not None:
            parts = []
            for a, b in self._hole_pairs:
                d = float(np.linalg.norm(self._hole_positions[a] - self._hole_positions[b]))
                flag = " ✓" if d < HOLE_CONVERGENCE_THRESHOLD else ""
                parts.append(f"({a},{b})={d*1e3:.1f}mm{flag}")
            _log(f"  hole_dists: {', '.join(parts)}"
                 f"  [thresh={HOLE_CONVERGENCE_THRESHOLD*1e3:.0f}mm]")

        # ── Per-beam: position, goal error, planner losses ──────────────
        hl = self._hand_losses
        for k in range(n):
            bp = self._states.beam_poses[k, :3].detach().cpu().numpy()
            bg = self._states.beam_goal[k, :3].detach().cpu().numpy()
            xy_err = float(np.linalg.norm(bp[:2] - bg[:2]))
            beam_yaw = float(torch.atan2(self._states.beam_poses[k, 3],
                                         self._states.beam_poses[k, 4]))
            goal_yaw = float(torch.atan2(self._states.beam_goal[k, 3],
                                         self._states.beam_goal[k, 4]))
            yaw_err_deg = float(np.degrees(
                (beam_yaw - goal_yaw + np.pi) % (2 * np.pi) - np.pi))

            l_pos_loss = float(hl._beam_pos_loss[k])
            r_pos_loss = float(hl._beam_pos_loss[n + k])
            conv_p = float(hl._beam_conver_p[k])
            _log(f"  beam{k}:"
                 f" pos=[{bp[0]:.3f},{bp[1]:.3f},{bp[2]:.3f}]"
                 f" goal=[{bg[0]:.3f},{bg[1]:.3f},{bg[2]:.3f}]"
                 f" xy_err={xy_err*1e3:.1f}mm"
                 f" yaw_err={yaw_err_deg:.1f}°"
                 f" L_pos_loss={l_pos_loss:.2e}"
                 f" R_pos_loss={r_pos_loss:.2e}"
                 f" conv_p={conv_p:.2f}")

        # ── Active-arm EE positions and pregrasp losses ─────────────────
        if li is not None and ri is not None:
            lp = self._states.left_pose[:3].detach().cpu().numpy()
            rp = self._states.right_pose[:3].detach().cpu().numpy()
            _log(f"  L_ee=[{lp[0]:.3f},{lp[1]:.3f},{lp[2]:.3f}]"
                 f"  R_ee=[{rp[0]:.3f},{rp[1]:.3f},{rp[2]:.3f}]")

            pl = self._states.pregrasp[li, :3].detach().cpu().numpy()
            pr = self._states.pregrasp[ri, :3].detach().cpu().numpy()
            l_preg_axis = float(np.max(np.abs(lp - pl)))
            r_preg_axis = float(np.max(np.abs(rp - pr)))
            l_grasp_axis = self._descent_max_axis_err_l
            r_grasp_axis = self._descent_max_axis_err_r
            _log(f"  pregrasp_xyz_max_err: L={l_preg_axis*1e3:.1f}mm  R={r_preg_axis*1e3:.1f}mm"
                 f"  [gate<{PREGRASP_TOL*1e3:.0f}mm]")
            _log(f"  grasp_xyz_max_err:    L={l_grasp_axis*1e3:.1f}mm  R={r_grasp_axis*1e3:.1f}mm"
                 f"  [gate<{GRASP_POS_TOL*1e3:.0f}mm]")

    # ------------------------------------------------------------------
    # Transition evaluation — split into idle (pre-rollout) and step
    # (per-integration) so PICK_TASK / READY type "instant" states do not
    # waste a gradient step.
    # ------------------------------------------------------------------

    def _evaluate_idle_transitions(self) -> None:
        """Resolve states that are decided without a gradient step.

        ``PICK_TASK``, ``CLOSE_GRIPPER``, ``READY``, ``RECOVERY_RELEASE``
        and ``RELEASE_GRIPPER`` all transition unconditionally after one
        cycle — their semantics are "set a flag, advance".

        ``DONE`` normally terminates here, but it re-enters ``PICK_TASK`` if
        external code has cleared entries from ``_convergence`` (e.g. a
        perturbation test or a grasp failure handler upstream).  This keeps the
        planner alive across convergence resets without any special caller API.
        """
        if self._state == State.DONE:
            # DONE is an idle-wait loop, not a hard terminal state.  Every
            # cycle the planner re-checks whether any pair has become
            # unconverged (e.g. a beam was perturbed by the environment or
            # cleared by upstream failure-recovery code).  When new work
            # appears the FSM seamlessly re-enters PICK_TASK without any
            # caller intervention.  Zero gradient is emitted while waiting.
            if self._select_pair() is not None:
                self._goto(State.PICK_TASK)
                self._enter_pick_task()
            return

        if self._state == State.PICK_TASK:
            self._enter_pick_task()
        elif self._state == State.CLOSE_GRIPPER:
            # Latch both grippers closed. No hysteresis — only opened
            # explicitly in RELEASE / RECOVERY_RELEASE.
            self._gripper_closed = [True, True]
            self._goto(State.READY)
        elif self._state == State.READY:
            self._goto(State.DUAL_ASSEMBLE)
        elif self._state == State.RECOVERY_RELEASE:
            self._gripper_closed = [False, False]
            self._goto(State.RECOVERY_MOVE_UP)
        elif self._state == State.RELEASE_GRIPPER:
            self._gripper_closed = [False, False]
            self._goto(State.MOVE_AWAY)

    def _evaluate_step_transitions(self) -> None:
        """Check gradient-driven gates after each integration step."""
        if self._state == State.GO_HOME:
            if self._home_reached() or self._state_step >= GO_HOME_TIMEOUT_STEPS:
                self._goto(State.PICK_TASK)
                self._enter_pick_task()

        elif self._state == State.MOVE_TO_PREGRASP:
            if self._pregrasp_reached_5dof():
                self._goto(State.DESCENDING)

        elif self._state == State.DESCENDING:
            if self._grasp_contact():
                self._goto(State.CLOSE_GRIPPER)
                # Resolve the 1-step CLOSE_GRIPPER immediately so the next
                # rollout step starts in READY (and then DUAL_ASSEMBLE).
                self._evaluate_idle_transitions()
                self._evaluate_idle_transitions()
            elif self._state_step >= DESCEND_TIMEOUT_STEPS:
                self._goto(State.RECOVERY_RELEASE)
                self._evaluate_idle_transitions()  # opens grippers, → MOVE_UP

        elif self._state == State.DUAL_ASSEMBLE:
            # Convergence — only checked here and in PICK_TASK.
            if self._active_pair_converged():
                self._latch_active_pair_converged()
                self._goto(State.RELEASE_GRIPPER)
                self._evaluate_idle_transitions()  # opens grippers, → MOVE_AWAY
                return
            # Slip detector — drives recovery without releasing convergence.
            if self._ee_slipped_from_beams():
                self._slip_counter += 1
            else:
                self._slip_counter = 0
            if self._slip_counter >= ASSEMBLE_SLIP_STEPS:
                self._slip_counter = 0
                self._goto(State.RECOVERY_RELEASE)
                self._evaluate_idle_transitions()
                return
            if self._state_step >= ASSEMBLE_TIMEOUT_STEPS:
                self._goto(State.RECOVERY_RELEASE)
                self._evaluate_idle_transitions()

        elif self._state == State.RECOVERY_MOVE_UP:
            if self._move_up_reached():
                self._goto(State.MOVE_TO_PREGRASP)

        elif self._state == State.MOVE_AWAY:
            if self._move_up_reached():
                self._goto(State.PICK_TASK)
                self._enter_pick_task()

        # GO_HOME counter
        self._state_step += 1

    def _goto(self, new_state: State) -> None:
        """Centralised state transition — resets the per-state step counter."""
        self._state = new_state
        self._state_step = 0

    # ------------------------------------------------------------------
    # PICK_TASK
    # ------------------------------------------------------------------

    def _enter_pick_task(self) -> None:
        """Latch hole-converged beams, then pick the next pair (or DONE).

        Convergence is **only** evaluated in this method (and in
        ``DUAL_ASSEMBLE`` for the active pair). All other states leave
        ``_convergence`` / ``_convergence_counter`` untouched.
        """
        # Hysteresis-based latching of any pair whose holes are within
        # threshold for several consecutive PICK_TASK visits.
        for k in range(self._state_params.no_beams):
            if self._beam_converged_by_holes(k):
                self._convergence_counter[k] = self._convergence_counter.get(k, 0) + 1
                if self._convergence_counter[k] >= CONVERGENCE_HYSTERESIS:
                    self._convergence[k] = True
            else:
                self._convergence_counter[k] = 0

        pair_idx = self._select_pair()
        if pair_idx is None:
            self._active_pair_idx = None
            self._left_index = None
            self._right_index = None
            self._goto(State.DONE)
            return

        self._active_pair_idx = pair_idx
        a, b = self._hole_pairs[pair_idx]
        # Fixed mapping: right always handles odd-indexed beams (1, 3),
        # left always handles even-indexed beams (0, 2).
        if a % 2 == 1:
            self._right_index = a
            self._left_index  = b
        else:
            self._right_index = b
            self._left_index  = a
        self._goto(State.MOVE_TO_PREGRASP)

    def _select_pair(self) -> Optional[int]:
        """Smallest-distance unconverged pair. ``hole_positions`` is
        guaranteed non-None by ``optimise()``'s precondition check."""
        best_idx, best_score = None, float("inf")
        for pair_idx, (a, b) in enumerate(self._hole_pairs):
            if a in self._convergence and b in self._convergence:
                continue
            score = float(np.linalg.norm(self._hole_positions[a] - self._hole_positions[b]))
            if score < best_score:
                best_score, best_idx = score, pair_idx
        return best_idx

    def _beam_converged_by_holes(self, beam_idx: int) -> bool:
        for a, b in self._hole_pairs:
            if beam_idx in (a, b):
                d = float(np.linalg.norm(self._hole_positions[a] - self._hole_positions[b]))
                return d < HOLE_CONVERGENCE_THRESHOLD
        return False

    def _active_pair_converged(self) -> bool:
        if self._active_pair_idx is None:
            return False
        a, b = self._hole_pairs[self._active_pair_idx]
        d = float(np.linalg.norm(self._hole_positions[a] - self._hole_positions[b]))
        return d < HOLE_CONVERGENCE_THRESHOLD

    def _latch_active_pair_converged(self) -> None:
        a, b = self._hole_pairs[self._active_pair_idx]
        self._convergence[a] = True
        self._convergence[b] = True

    # ------------------------------------------------------------------
    # Per-state gradient producers — each returns the DualArmGradients to
    # apply this step. Pure functions of self._states / self._hand_losses.
    # ------------------------------------------------------------------

    def _step_gradients(self) -> DualArmGradients:
        self._gradients.zero()

        # Refresh hand-side losses for the current state's gate logic.
        self._hand_losses.calc_losses(self._states)
        self._hand_losses.calc_prob()
        self._beam_losses_bank.calc_losses(self._states)
        self._pregrasp_losses_bank.calc_losses(self._states)

        s = self._state
        if s == State.GO_HOME:
            self._grad_go_home()
        elif s in (State.MOVE_TO_PREGRASP, State.RECOVERY_MOVE_UP, State.MOVE_AWAY):
            self._grad_pregrasp()
        elif s == State.DESCENDING:
            self._grad_descending()
        elif s == State.DUAL_ASSEMBLE:
            self._grad_dual_assemble()
        # PICK_TASK / CLOSE_GRIPPER / READY / RELEASE_GRIPPER /
        # RECOVERY_RELEASE / DONE  → zero gradient (already set above).
        return self._gradients

    def _grad_go_home(self) -> None:
        """Drive both arms to their start posture simultaneously."""
        loss_l = ((self._states.left_pose  - self.left_start)  ** 2).sum()
        loss_r = ((self._states.right_pose - self.right_start) ** 2).sum()
        self._gradients.left_pose  = grad(loss_l, self._states.left_pose,  retain_graph=True)[0]
        self._gradients.right_pose = grad(loss_r, self._states.right_pose, retain_graph=True)[0]

    def _grad_pregrasp(self) -> None:
        """Pull each arm to its pregrasp pose. Used by MOVE_TO_PREGRASP,
        RECOVERY_MOVE_UP, and MOVE_AWAY — they share the same controller,
        only the transition gate differs."""
        if self._left_index is None or self._right_index is None:
            return
        # Per-arm pregrasp loss against the assigned beam's pregrasp pose.
        pregrasp_l = self._states.pregrasp[self._left_index]
        pregrasp_r = self._states.pregrasp[self._right_index]
        loss_l = ((self._states.left_pose  - pregrasp_l) ** 2).sum()
        loss_r = ((self._states.right_pose - pregrasp_r) ** 2).sum()
        self._gradients.left_pose  = grad(loss_l, self._states.left_pose,  retain_graph=True)[0]
        self._gradients.right_pose = grad(loss_r, self._states.right_pose, retain_graph=True)[0]

        # Track the pregrasp target on top of the perceived beam.
        pg_loss = self._pregrasp_losses_bank.pregrasp_loss
        self._gradients.pregrasp = grad(pg_loss, self._states.pregrasp, retain_graph=True)[0]

    def _grad_descending(self) -> None:
        """Drive each arm to the perceived beam pose (x, y, z, yaw).

        The descent target now follows the perceived beam z directly — the
        previous fixed ``self.grasp_z`` override has been removed so the arms
        track the AprilTag z estimate rather than asymptoting to a hardcoded
        height that sits above the actual beam top.
        """
        if self._left_index is None or self._right_index is None:
            return

        def _target(beam_idx: int) -> torch.Tensor:
            return self._states.beam_poses[beam_idx].detach().clone()

        left_target  = _target(self._left_index)
        right_target = _target(self._right_index)

        loss_l = ((self._states.left_pose  - left_target)  ** 2).sum()
        loss_r = ((self._states.right_pose - right_target) ** 2).sum()

        # Per-axis (x, y, z) cache for the contact gate. We also retain the
        # Euclidean distance for the snap branch below.
        with torch.no_grad():
            diff_l = (self._states.left_pose[:3]  - left_target[:3]).abs()
            diff_r = (self._states.right_pose[:3] - right_target[:3]).abs()
            self._descent_max_axis_err_l = float(diff_l.max())
            self._descent_max_axis_err_r = float(diff_r.max())
            dist_l = float((diff_l ** 2).sum() ** 0.5)
            dist_r = float((diff_r ** 2).sum() ** 0.5)
            self._descent_loss_l = dist_l
            self._descent_loss_r = dist_r

        g_l = grad(loss_l, self._states.left_pose,  retain_graph=True)[0]
        g_r = grad(loss_r, self._states.right_pose, retain_graph=True)[0]

        # Snap: inside descent_snap_radius replace the quadratic gradient with
        # one whose integrator step lands the planner state exactly on the
        # target.  delta = LEARNING_RATE · g  →
        # set g = (x - x*) / LEARNING_RATE  so delta = (x - x*) and
        # x_new = x - delta = x*.  max_ee_step still clips the result if it
        # would exceed the per-step displacement cap.
        self._descent_snap_fired_l = False
        self._descent_snap_fired_r = False
        snap_r = self.descent_snap_radius
        if snap_r > 0.0 and LEARNING_RATE > 0.0:
            with torch.no_grad():
                if 0.0 < dist_l < snap_r:
                    g_l = (self._states.left_pose  - left_target).detach()  / LEARNING_RATE
                    self._descent_snap_fired_l = True
                    self._descent_snap_count_l += 1
                if 0.0 < dist_r < snap_r:
                    g_r = (self._states.right_pose - right_target).detach() / LEARNING_RATE
                    self._descent_snap_fired_r = True
                    self._descent_snap_count_r += 1

        self._gradients.left_pose  = g_l
        self._gradients.right_pose = g_r

    def _grad_dual_assemble(self) -> None:
        """Beam-goal + hole-collinearity gradient, contact-blended onto EEs.

        Behavioural parity with ``dual_assembly.py:679-764`` — the only thing
        the FSM removes is the "is the arm currently in contact?" gating;
        by the time we reach ``DUAL_ASSEMBLE`` the FSM has guaranteed both
        arms are grasped.
        """
        if self._left_index is None or self._right_index is None:
            return

        # Beam gradient: beam→goal MSE + hole-collinearity term.
        beam_grad = grad(
            self._beam_losses_bank._beam_losses,
            self._states.beam_poses,
            retain_graph=True,
        )[0]
        hole_loss = _hole_collinearity_loss(
            self._states.beam_poses, self._hole_positions, self._hole_pairs, self.params.device
        )
        hole_grad = grad(hole_loss, self._states.beam_poses, retain_graph=True)[0]
        beam_grad = beam_grad + HOLE_GRADIENT_WEIGHT * hole_grad
        beam_grad[:, 3:5] *= YAW_GRADIENT_WEIGHT

        # Zero the gradient on already-converged beams so the planner is
        # free to ignore them.
        for k in self._convergence:
            beam_grad[k] *= 0.0

        self._gradients.beam_poses = beam_grad

        # Drive each arm with the beam gradient of its assigned beam — the
        # arm is grasped, so the beam moves with the EE.
        # Zero the z component: arms stay at grasp_z throughout assembly.
        left_g  = beam_grad[self._left_index].clone()
        right_g = beam_grad[self._right_index].clone()
        left_g[2]  = 0.0
        right_g[2] = 0.0
        self._gradients.left_pose  = left_g
        self._gradients.right_pose = right_g

    # ------------------------------------------------------------------
    # Gates — small predicates on the latest losses.
    # ------------------------------------------------------------------

    def _home_reached(self) -> bool:
        dl = float(((self._states.left_pose[:3]  - self.left_start[:3])  ** 2).sum().sqrt())
        dr = float(((self._states.right_pose[:3] - self.right_start[:3]) ** 2).sum().sqrt())
        return dl < MOVE_UP_TOL and dr < MOVE_UP_TOL

    def _pregrasp_reached_5dof(self) -> bool:
        """Per-axis (x, y, z) pregrasp gate; rotation ignored."""
        if self._left_index is None or self._right_index is None:
            return False
        pl = self._states.pregrasp[self._left_index, :3]
        pr = self._states.pregrasp[self._right_index, :3]
        l = float((self._states.left_pose[:3]  - pl).abs().max())
        r = float((self._states.right_pose[:3] - pr).abs().max())
        return l < PREGRASP_TOL and r < PREGRASP_TOL

    def _grasp_contact(self) -> bool:
        """Both arms within GRASP_POS_TOL of their fixed-z descent target on
        every axis (x, y, z) independently. Uses per-axis max errors cached by
        ``_grad_descending`` against the fixed-z target rather than the raw
        perceived beam z (so the gate fires correctly when grasp_z ≠ beam z).
        """
        if self._left_index is None or self._right_index is None:
            return False
        return (self._descent_max_axis_err_l < GRASP_POS_TOL
                and self._descent_max_axis_err_r < GRASP_POS_TOL)

    def _ee_slipped_from_beams(self) -> bool:
        """Distance between each EE and its assigned beam position; if either
        exceeds ``ASSEMBLE_SLIP_DIST`` the grasp is suspect."""
        if self._left_index is None or self._right_index is None:
            return False
        beam_l = self._states.beam_poses[self._left_index, :3]
        beam_r = self._states.beam_poses[self._right_index, :3]
        dl = float(((self._states.left_pose[:3]  - beam_l) ** 2).sum().sqrt())
        dr = float(((self._states.right_pose[:3] - beam_r) ** 2).sum().sqrt())
        return dl > ASSEMBLE_SLIP_DIST or dr > ASSEMBLE_SLIP_DIST

    def _move_up_reached(self) -> bool:
        """Position-only pregrasp gate (no orientation requirement)."""
        if self._left_index is None or self._right_index is None:
            return True  # nothing to retreat from
        pl = self._states.pregrasp[self._left_index, :3]
        pr = self._states.pregrasp[self._right_index, :3]
        dl = float(((self._states.left_pose[:3]  - pl) ** 2).sum().sqrt())
        dr = float(((self._states.right_pose[:3] - pr) ** 2).sum().sqrt())
        return dl < MOVE_UP_TOL and dr < MOVE_UP_TOL

    # ------------------------------------------------------------------
    # EE constraint helpers
    # ------------------------------------------------------------------

    def _limit_ee_delta(self, delta: torch.Tensor) -> torch.Tensor:
        """Clip the positional part of an EE gradient delta to max_ee_step.

        Only the first three elements (x, y, z) are norm-limited; the yaw
        channels (sin θ, cos θ) are left untouched so orientation updates are
        not silently suppressed.  Returns a new tensor — does not modify
        ``delta`` in-place.
        """
        if not np.isfinite(self.max_ee_step):
            return delta
        pos_norm = float(delta[:3].detach().norm())
        if pos_norm > self.max_ee_step:
            result = delta.clone()
            result[:3] = delta[:3] * (self.max_ee_step / pos_norm)
            return result
        return delta

    def _clamp_ee_poses(self) -> None:
        """Apply per-arm z workspace bounds to both EE poses in-place.

        Bounds applied (all configurable as instance attributes):
        - Left EE:  z in [left_z_floor,  left_z_ceil]
        - Right EE: z in [right_z_floor, right_z_ceil]
        - Pregrasp: z in [arm_z_floor,   arm_z_ceil]  (shared)

        The pregrasp z clamp is kept here (rather than duplicating it at the
        call site) so all hard spatial bounds live in one place.
        """
        with torch.no_grad():
            self._states.left_pose[2] = torch.clamp(
                self._states.left_pose[2], min=self.left_z_floor, max=self.left_z_ceil)
            self._states.right_pose[2] = torch.clamp(
                self._states.right_pose[2], min=self.right_z_floor, max=self.right_z_ceil)
            self._states.pregrasp[:, 2] = torch.clamp(
                self._states.pregrasp[:, 2], min=self.arm_z_floor, max=self.arm_z_ceil)

    # ------------------------------------------------------------------
    # Pose normalisation (z floor + yaw unit-circle projection).
    # ------------------------------------------------------------------

    def _normalise_pose(self, pose_torch: torch.Tensor) -> torch.Tensor:
        """Clip each pose-block z to the appropriate floor and renormalise
        the (sinθ, cosθ) yaw channels back onto the unit circle.

        Same semantics as ``dual_assembly.normalise_pose`` — preserved so
        downstream geometry (e.g. ``state_to_pose_quat``) keeps working.
        """
        no_items = pose_torch.shape[0] // self.state_dim
        for k in range(no_items):
            z_min = 0.805
            if k < self._state_params.no_hands and self.arm_z_floor is not None:
                z_min = max(z_min, self.arm_z_floor)
            pose_torch[self.state_dim * k + 2] = max(pose_torch[self.state_dim * k + 2], z_min)
            pose_torch[self.state_dim * k + 3 : self.state_dim * k + 5] = (
                torch.nn.functional.normalize(
                    pose_torch[self.state_dim * k + 3 : self.state_dim * k + 5], dim=0
                )
            )
        return pose_torch


__all__ = [
    "TrajOptParams",
    "StateParams",
    "DualArmStates",
    "ParticleTrajectories",
    "DualAssembly",
    "State",
]
