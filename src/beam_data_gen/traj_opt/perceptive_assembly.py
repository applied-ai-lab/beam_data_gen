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
GRASP_POS_TOL: float = 0.013

# Per-pair assembly convergence (Euclidean hole distance).
HOLE_CONVERGENCE_THRESHOLD: float = 0.005

# Cycles a pair must stay below the hole threshold before being latched.
CONVERGENCE_HYSTERESIS: int = 1

# DUAL_ASSEMBLE slippage detector. Distance is measured per arm between the
# end-effector position and its assigned beam position; if exceeded for the
# given number of consecutive steps the grasp is considered failed.
ASSEMBLE_SLIP_DIST: float = 0.15      # 5 cm
ASSEMBLE_SLIP_STEPS: int  = 1000

# Per-state step budgets. The planner runs at ~30 Hz so 150 ≈ 5 s.
DESCEND_TIMEOUT_STEPS:  int = 35
ASSEMBLE_TIMEOUT_STEPS: int = 75
GO_HOME_TIMEOUT_STEPS:  int = 100

# Position-only gate for MOVE_AWAY / RECOVERY_MOVE_UP (orientation ignored).
MOVE_UP_TOL: float = 0.04
MOVE_UP_Z: float = 1.0


# Gradient mixing — kept identical to dual_assembly.py for behavioural parity
# in the assembly phase.
HOLE_GRADIENT_WEIGHT: float = 0.0
YAW_GRADIENT_WEIGHT:  float = 1.2

# Gradient-descent learning rate. Hard-coded here (instead of read from
# TrajOptParams.step_size) so the planner's integrator step is fixed by the
# module rather than by callers.
LEARNING_RATE: float = 0.25

# Snap radii (metres). Inside the radius the per-state gradient is replaced
# by one whose integrator step lands exactly on the target, bypassing the
# asymptotic shrinkage of a quadratic loss. Set to 0.0 to disable.
DESCENT_SNAP_RADIUS:  float = 0.04
ASSEMBLE_SNAP_RADIUS: float = 0.03
import os as _os
if _os.environ.get('DISABLE_ASSEMBLE_SNAP', '0') == '1':
    ASSEMBLE_SNAP_RADIUS = 0.0
if _os.environ.get('DISABLE_DESCENT_SNAP', '0') == '1':
    DESCENT_SNAP_RADIUS = 0.0


# ---------------------------------------------------------------------------
# Pin-phase tunables (see PERCEPTIVE_ASSEMBLY.MD §10).
# ---------------------------------------------------------------------------

# Per-pair Euclidean (xy) tolerance for pin insertion convergence — the
# "8 mm regrasp loop": below this, RELEASE_PIN may fire; above it, the
# planner forces a recovery + regrasp instead of releasing.
PIN_INSERT_TOL: float = 0.01
PIN_INSERT_PREGRASP_TOL: float = 0.01

# Cycles the pin must remain inside PIN_INSERT_TOL before RELEASE_PIN fires.
PIN_CONVERGENCE_HYSTERESIS: int = 1

# Yaw gate for ROTATE_PIN_INWARD (rad).
PIN_YAW_TOL: float = 0.2

# Hover height above pin / hole-midpoint during pin pregrasp (m).
PIN_PREGRASP_OFFSET_Z: float = 0.20

# Pregrasp / grasp yaw is forced to 0 during pin pickup —
# (sin θ, cos θ) = (0, 1).
PIN_GRASP_SIN: float = 0.0
PIN_GRASP_COS: float = 1.0

# "Inward" for the left arm: gripper points toward y = 0.  Left arm sits
# on +y, so inward yaw = -π/2 → (sin θ, cos θ) = (-1, 0).
INWARD_YAW_LEFT_SIN: float = -1.0
INWARD_YAW_LEFT_COS: float = 0.0

# Pin-phase home offset (m).  Both arms park at the beam-phase home
# shifted this far further from the workspace centreline (y = 0) before
# the PTU is commanded to PIN_VIEW: the left arm's y is increased by
# this amount, the right arm's y is decreased by the same.  Gives the
# camera an unobstructed view and keeps the right arm well clear of the
# left arm's working volume during pin insertion.
PIN_HOME_Y_OFFSET: float = 0.20
PIN_HOME_Z: float = 1.0
PIN_HOME_X: float = 0.4


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

    # ---- Pin insertion phase (single-arm; right held at right_start). ----
    STOW_BOTH              = 12  # both arms → pin-phase home; entered the moment beams converge, before PTU moves
    PICK_PIN               = 13  # greedy assignment: closest pin → closest unfilled pair
    MOVE_TO_PIN_PREGRASP   = 14  # left hovers above pin, yaw = 0
    DESCEND_TO_PIN         = 15  # left descends to pin, yaw = 0
    CLOSE_PIN_GRIPPER      = 16  # latch left gripper closed (1 step)
    LIFT_PIN               = 17  # lift pin to MOVE_UP_Z, yaw = 0
    ROTATE_PIN_INWARD      = 18  # rotate left wrist 90° inward (toward y = 0)
    MOVE_TO_HOLE_PREGRASP  = 19  # left moves over hole-pair midpoint, inward yaw
    INSERT_PIN             = 20  # descend pin into midpoint; 8 mm convergence gate
    RELEASE_PIN            = 21  # latch left gripper open (1 step) on success
    RETREAT_PIN            = 22  # lift to MOVE_UP_Z then PICK_PIN
    PIN_RECOVERY_RELEASE   = 23  # open gripper on grasp/insert failure (1 step)
    PIN_RECOVERY_MOVE_UP   = 24  # lift, retry MOVE_TO_PIN_PREGRASP for same pin
    ALL_DONE               = 25  # terminal: all pins placed


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

    Each call to :meth:`optimise` runs ``params.no_steps`` gradient-descent
    integration steps from the current ``self.states`` baseline. Within a
    single call the FSM may transition multiple times — transitions are
    re-evaluated after every integration step. The state itself persists
    across calls (``self._state``).

    The class deliberately reuses the data-bearing classes from
    ``dual_assembly`` (``DualArmStates``, ``HandLossesContacts``, etc.) so
    the wire format consumed by callers is bit-identical.
    """

    # ------------------------------------------------------------------
    # Pin-phase home targets — computed live from self.left_start /
    # self.right_start so they pick up the FK home pose the caller patches
    # in after construction.  Only the left arm is offset (+y by
    # PIN_HOME_Y_OFFSET); right arm parks at right_start unchanged.
    # ------------------------------------------------------------------

    @property
    def _left_pin_home(self) -> torch.Tensor:
        if self.left_start is None:
            raise RuntimeError(
                "DualAssembly.left_start is None — caller must set it (via "
                "FK on HOME_Q_LEFT) before any state that reads pin homes "
                "(STOW_BOTH and the rest of pin phase)."
            )
        h = self.left_start.detach().clone()
        h[1] = h[1] + PIN_HOME_Y_OFFSET
        return h

    @property
    def _right_pin_home(self) -> torch.Tensor:
        if self.right_start is None:
            raise RuntimeError(
                "DualAssembly.right_start is None — caller must set it (via "
                "FK on HOME_Q_RIGHT) before any state that reads pin homes."
            )
        return self.right_start.detach().clone()

    # ------------------------------------------------------------------
    # Construction / lifecycle
    # ------------------------------------------------------------------

    def __init__(
        self,
        params: TrajOptParams,
        state_params: StateParams,
        sim: Optional[SquareRobotSim],
        left_start: Optional[torch.Tensor] = None,
        right_start: Optional[torch.Tensor] = None,
        model: mujoco.MjModel = None,
        data: mujoco.MjData = None,
        hole_pairs: List[Tuple[int, int]] = None,
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

        # No silent default: callers must explicitly set these (typically
        # to FK(HOME_Q_*)) before STOW_BOTH / pin phase, or the pin-home
        # properties below will raise.
        self.left_start = left_start
        self.right_start = right_start

        # Pin-phase home: same pose as the beam-phase home, shifted
        # PIN_HOME_Y_OFFSET further away from the workspace centreline.
        # Convention: left arm sits at +y, right at -y, so the offset is
        # added on the left and subtracted on the right.  Both arms drive
        # to this pose during STOW_BOTH before the PTU is moved.
        # Pin homes are computed live from self.left_start / self.right_start
        # via @property below.  Capturing them here would freeze whatever
        # placeholder the caller passed in (atls.py constructs DualAssembly
        # with _zero starts and patches left_start/right_start afterwards).
        # Only the left arm is offset (+y by PIN_HOME_Y_OFFSET); the right
        # arm parks at right_start unchanged.

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
        self.arm_z_floor: float = 0.78
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
        self.grasp_z: float = 0.793 #UNUSED NOW

        # Per-beam flag (last DUAL_ASSEMBLE call) and cumulative count of
        # assemble-snap activations. See ASSEMBLE_SNAP_RADIUS at top of module.
        self._assemble_snap_fired: List[bool] = [False] * self._state_params.no_beams
        self._assemble_snap_count: int = 0

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
        self._slip_counter: int = 0    # consecutive slip steps (DUAL_ASSEMBLE / INSERT_PIN)
        # Per-arm gripper command — flipped only by CLOSE_GRIPPER /
        # RELEASE_GRIPPER / RECOVERY_RELEASE. No hysteresis.
        self._gripper_closed: List[bool] = [False, False]

        # ---- Pin phase bookkeeping (see PERCEPTIVE_ASSEMBLY.MD §10). ----
        # Sized lazily by ``set_pin_positions``.  ``_pin_phase_active`` is a
        # one-way latch — once true, PICK_TASK is unreachable for the rest
        # of the run.
        self._pin_positions: Optional[np.ndarray] = None
        self._pin_inserted: List[bool] = []
        self._pinned_pairs: set = set()
        self._active_pin_idx: Optional[int] = None
        self._active_hole_pair_idx: Optional[int] = None
        self._pin_phase_active: bool = False
        self._pin_insert_counter: int = 0

        # ---- Testing / diagnostics knobs. ----
        # Set step_mode = True to pause at every FSM state transition and wait
        # for an Enter keypress.  Safe to toggle at runtime; harmless when False.
        self.step_mode: bool = False
        self._step_mode_last_state: Optional[State] = None

        # Trajectory buffer (allocated lazily by initialise()). Wrapped in a
        # ``ParticleTrajectories`` purely to preserve the consumer-facing API
        # (``sample_indices`` / ``sample_trajectories``) — there is no actual
        # particle filter; this planner produces one deterministic rollout per
        # cycle. The particle dimension is kept at ``params.no_particles`` and
        # broadcast across so downstream indexing stays valid.
        self._particle_trajectories: Optional[ParticleTrajectories] = None

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

    def set_pin_positions(self, positions: np.ndarray) -> None:
        """Update perceived pin positions — required every cycle once pin
        phase is active.

        Args:
            positions: ``(N_pins, 3)`` world-frame XYZ for each pin
                (typically sourced from ``pin{0..N-1}_filtered`` TF frames).
        """
        if self._pin_positions is None:
            self._pin_inserted = [False] * positions.shape[0]
        self._pin_positions = positions

    def is_pin_view_pose_required(self) -> bool:
        """Whether the PTU should be commanded to its PIN_VIEW pose.

        True for the entire pin phase from STOW_BOTH entry onward, so
        the camera slews to PIN_VIEW in parallel with the arms moving
        to their pin-phase home.  False during beam phase and on
        ALL_DONE.
        """
        if not self._pin_phase_active:
            return False
        if self._state == State.ALL_DONE:
            return False
        return True

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
        """Allocate the trajectory buffer for the current trajectory size."""
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
        # Identity index map: ``sample_indices`` walks back through these and
        # always lands on particle 0 (the only row actually written).
        pt.indices = torch.arange(n_particles, dtype=torch.int32) \
            .unsqueeze(0).repeat(n_steps, 1)
        pt.no_live_particles = torch.full([n_steps], n_particles, dtype=torch.int32)
        pt.loss = torch.zeros([n_particles, n_steps], dtype=torch.float32)
        self._particle_trajectories = pt

    def reset(self) -> None:
        """Zero out the per-cycle trajectory buffer (preserves index map)."""
        self._particle_trajectories.particles.zero_()
        self._particle_trajectories.gripper_particles.zero_()
        self._particle_trajectories.loss.zero_()

    # ------------------------------------------------------------------
    # Main planning loop
    # ------------------------------------------------------------------

    def optimise(self) -> ParticleTrajectories:
        """Run one planning cycle of ``no_steps`` gradient-descent integrations.

        Raises:
            RuntimeError: if ``set_hole_positions`` was never called.

        Pin observations are *not* required for ``optimise()`` to run. Pin
        phase is entered the moment beams converge — typically before the
        PTU has moved to its pin-view pose, so pins are out of FOV and
        ``_pin_positions`` is still ``None``.  The FSM holds in
        ``STOW_BOTH`` (zero progression toward any pin) until a fresh
        observation arrives via ``set_pin_positions``; safe to call
        ``optimise`` repeatedly meanwhile.
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

        # Seed step 0 with the current robot baseline (broadcast across the
        # particle dim — see initialise() docstring).
        self._particle_trajectories.particles[:, 0, :] = self._x
        self._particle_trajectories.gripper_particles[:, 0, 0] = float(self._gripper_closed[0])
        self._particle_trajectories.gripper_particles[:, 0, 1] = float(self._gripper_closed[1])

        # Single deterministic gradient-descent rollout.
        for k in range(1, self.params.no_steps):
            self._states.requires_grad()
            gradients = self._step_gradients()

            # Pre-step distance gate evaluation: if a position-based
            # threshold is already satisfied (grasp contact, pair
            # converged, lift reached, …), transition the FSM now and
            # emit no further motion this step.  Without this the snap
            # branch would overshoot the gate and tow the beam through
            # the close-gripper transition.
            _pre_state = self._state
            self._evaluate_step_transitions(check_distance_only=True)
            _gate_fired = (self._state != _pre_state)

            if _gate_fired:
                # State transitioned (and any idle resolution has run).
                # Skip applying the stale gradient — the next step will
                # plan from the new state.
                self._states.detach()
            else:
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

            self._particle_trajectories.loss[:, k] = self._beam_losses_bank._beam_losses.sum()

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

            self._particle_trajectories.particles[:, k, :] = self._x
            self._particle_trajectories.gripper_particles[:, k, 0] = float(self._gripper_closed[0])
            self._particle_trajectories.gripper_particles[:, k, 1] = float(self._gripper_closed[1])

            # Post-step: time-based logic only (timeouts, slip counter,
            # state-step counter increment).  Distance gates ran BEFORE
            # the gradient against the actual robot state — running them
            # again here would fire on the planner's optimistic post-step
            # prediction and exit a state long before the real arm has
            # arrived at the gate threshold.
            self._hand_losses.calc_losses(self._states)
            self._hand_losses.calc_prob()
            self._evaluate_step_transitions(skip_distance=True)

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
            #
            # Once pin phase has been entered, the perturbation re-entry
            # is suppressed — we do not want to drop a held pin to fix a
            # drifted beam (see PERCEPTIVE_ASSEMBLY.MD §10.8).
            if self._pin_phase_active:
                return
            if self._select_pair() is not None:
                self._goto(State.PICK_TASK)
                self._enter_pick_task()
            return

        if self._state == State.ALL_DONE:
            return  # terminal — no perturbation re-entry

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
        elif self._state == State.PICK_PIN:
            self._enter_pick_pin()
        elif self._state == State.CLOSE_PIN_GRIPPER:
            # Single-arm: only the left gripper is actuated in pin phase.
            self._gripper_closed[0] = True
            self._goto(State.LIFT_PIN)
        elif self._state == State.RELEASE_PIN:
            self._gripper_closed[0] = False
            if self._active_hole_pair_idx is not None:
                self._pinned_pairs.add(self._active_hole_pair_idx)
            if self._active_pin_idx is not None:
                self._pin_inserted[self._active_pin_idx] = True
            self._pin_insert_counter = 0
            self._slip_counter = 0
            self._goto(State.RETREAT_PIN)
        elif self._state == State.PIN_RECOVERY_RELEASE:
            self._gripper_closed[0] = False
            self._pin_insert_counter = 0
            self._slip_counter = 0
            self._goto(State.PIN_RECOVERY_MOVE_UP)

    def _evaluate_step_transitions(self, check_distance_only: bool = False,
                                   skip_distance: bool = False) -> None:
        """Check gradient-driven gates around each integration step.

        Distance-based gates (``_lift_reached``, ``_grasp_contact``,
        ``_active_pair_converged``, ``_pregrasp_reached_5dof``, …) are
        evaluated against the planner's current state.  Because callers
        re-seed ``left_pose`` / ``right_pose`` from the actual robot
        state at the start of each cycle, these gates *must* run BEFORE
        the gradient step (``check_distance_only=True``) — otherwise the
        planner's unrestricted post-step prediction overshoots the gate
        and transitions while the real arm is still nowhere near the
        threshold.

        Time-based logic (timeouts, slip counter, per-state step
        counter increment) runs AFTER the gradient step
        (``skip_distance=True``).  The two passes together fully replace
        the previous single post-step evaluation.
        """
        if self._state == State.GO_HOME:
            if (not skip_distance) and self._home_reached():
                self._goto(State.PICK_TASK)
                self._enter_pick_task()
            elif (not check_distance_only) and self._state_step >= GO_HOME_TIMEOUT_STEPS:
                self._goto(State.PICK_TASK)
                self._enter_pick_task()

        elif self._state == State.MOVE_TO_PREGRASP:
            if (not skip_distance) and self._pregrasp_reached_5dof():
                self._goto(State.DESCENDING)

        elif self._state == State.DESCENDING:
            if (not skip_distance) and self._grasp_contact():
                self._goto(State.CLOSE_GRIPPER)
                # Resolve the 1-step CLOSE_GRIPPER immediately so the next
                # rollout step starts in READY (and then DUAL_ASSEMBLE).
                self._evaluate_idle_transitions()
                self._evaluate_idle_transitions()
            elif (not check_distance_only) and self._state_step >= DESCEND_TIMEOUT_STEPS:
                self._goto(State.RECOVERY_RELEASE)
                self._evaluate_idle_transitions()  # opens grippers, → MOVE_UP

        elif self._state == State.DUAL_ASSEMBLE:
            # Convergence — only checked here and in PICK_TASK.  Uses
            # the same per-beam ``_convergence_counter`` hysteresis as
            # PICK_TASK so a brief, transient touch (beams nudged
            # apart on the next cycle) does NOT trigger a latch +
            # MOVE_AWAY.  The counter is incremented pre-step against
            # the seeded actual robot state; the latch + transition
            # only fires once both active beams have held the
            # threshold for ``CONVERGENCE_HYSTERESIS`` consecutive
            # cycles.
            if (not skip_distance) and self._active_pair_idx is not None:
                a, b = self._hole_pairs[self._active_pair_idx]
                if self._active_pair_converged():
                    self._convergence_counter[a] = self._convergence_counter.get(a, 0) + 1
                    self._convergence_counter[b] = self._convergence_counter.get(b, 0) + 1
                else:
                    self._convergence_counter[a] = 0
                    self._convergence_counter[b] = 0
                if (self._convergence_counter.get(a, 0) >= CONVERGENCE_HYSTERESIS
                        and self._convergence_counter.get(b, 0) >= CONVERGENCE_HYSTERESIS):
                    self._latch_active_pair_converged()
                    self._goto(State.RELEASE_GRIPPER)
                    self._evaluate_idle_transitions()  # opens grippers, → MOVE_AWAY
                    return
            if not check_distance_only:
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
            if (not skip_distance) and self._lift_reached():
                self._goto(State.MOVE_TO_PREGRASP)

        elif self._state == State.MOVE_AWAY:
            if (not skip_distance) and self._lift_reached():
                self._goto(State.PICK_TASK)
                self._enter_pick_task()

        # ---- Pin phase ----
        elif self._state == State.STOW_BOTH:
            # Wait for both arms to park at the pin-phase home, then
            # advance to PICK_PIN.
            if (not skip_distance) and self._at_pin_home():
                self._goto(State.PICK_PIN)
                self._evaluate_idle_transitions()  # resolve PICK_PIN

        elif self._state == State.MOVE_TO_PIN_PREGRASP:
            if (not skip_distance) and self._pin_pregrasp_reached():
                self._goto(State.DESCEND_TO_PIN)

        elif self._state == State.DESCEND_TO_PIN:
            if (not skip_distance) and self._pin_grasp_contact():
                self._goto(State.CLOSE_PIN_GRIPPER)
                self._evaluate_idle_transitions()  # → LIFT_PIN
            elif (not check_distance_only) and self._state_step >= DESCEND_TIMEOUT_STEPS:
                self._goto(State.PIN_RECOVERY_RELEASE)
                self._evaluate_idle_transitions()  # → PIN_RECOVERY_MOVE_UP

        elif self._state == State.LIFT_PIN:
            if (not skip_distance) and self._left_lift_reached():
                self._goto(State.ROTATE_PIN_INWARD)

        elif self._state == State.ROTATE_PIN_INWARD:
            if (not skip_distance) and self._pin_yaw_reached():
                self._goto(State.MOVE_TO_HOLE_PREGRASP)

        elif self._state == State.MOVE_TO_HOLE_PREGRASP:
            if (not skip_distance) and self._hole_pregrasp_reached():
                self._goto(State.INSERT_PIN)

        elif self._state == State.INSERT_PIN:
            # Convergence — the "8 mm regrasp loop" entry point. RELEASE_PIN
            # is reachable ONLY from here; every other exit forces recovery.
            if (not skip_distance) and self._check_pin_insert_progress():
                self._slip_counter = 0
                self._goto(State.RELEASE_PIN)
                self._evaluate_idle_transitions()  # opens left, → RETREAT_PIN
                return
            if not check_distance_only:
                if self._pin_slipped():
                    self._slip_counter += 1
                else:
                    self._slip_counter = 0
                if self._slip_counter >= ASSEMBLE_SLIP_STEPS:
                    self._slip_counter = 0
                    self._pin_insert_counter = 0
                    self._goto(State.PIN_RECOVERY_RELEASE)
                    self._evaluate_idle_transitions()
                    return
                if self._state_step >= ASSEMBLE_TIMEOUT_STEPS:
                    self._pin_insert_counter = 0
                    self._goto(State.PIN_RECOVERY_RELEASE)
                    self._evaluate_idle_transitions()

        elif self._state == State.RETREAT_PIN:
            if (not skip_distance) and self._left_lift_reached():
                self._goto(State.PICK_PIN)
                self._evaluate_idle_transitions()  # → MOVE_TO_PIN_PREGRASP or ALL_DONE

        elif self._state == State.PIN_RECOVERY_MOVE_UP:
            if (not skip_distance) and self._left_lift_reached():
                self._goto(State.MOVE_TO_PIN_PREGRASP)

        # Per-state step counter — only advanced on the post-step pass.
        if not check_distance_only:
            self._state_step += 1

    def _goto(self, new_state: State) -> None:
        """Centralised state transition — resets the per-state step counter."""
        self._state = new_state
        self._state_step = 0

    # ------------------------------------------------------------------
    # PICK_TASK
    # ------------------------------------------------------------------

    def _enter_pick_task(self) -> None:
        """Re-evaluate convergence from current observations, then pick
        the next pair (or transition to pin phase).

        Convergence is **only** evaluated in this method (and in
        ``DUAL_ASSEMBLE`` for the active pair). All other states leave
        ``_convergence`` / ``_convergence_counter`` untouched.

        Mistake correction: a beam that was previously latched but is
        no longer within threshold (e.g. the pair was knocked apart
        during MOVE_AWAY) is unlatched here so its pair is re-selected
        and re-attempted.  The latches become effectively immutable
        once we transition to STOW_BOTH (``_pin_phase_active = True``)
        because PICK_TASK is unreachable from pin phase — so this
        re-evaluation only ever runs while there is still beam work
        outstanding.
        """
        for k in range(self._state_params.no_beams):
            if self._beam_converged_by_holes(k):
                self._convergence_counter[k] = self._convergence_counter.get(k, 0) + 1
                if self._convergence_counter[k] >= CONVERGENCE_HYSTERESIS:
                    self._convergence[k] = True
            else:
                self._convergence_counter[k] = 0
                # Unlatch — current observation says this beam is not
                # actually converged.  Without this, a brief touch
                # during DUAL_ASSEMBLE could permanently mark a pair
                # done even after it gets knocked apart.
                self._convergence.pop(k, None)

        pair_idx = self._select_pair()
        if pair_idx is None:
            self._active_pair_idx = None
            self._left_index = None
            self._right_index = None
            # Beams done → pin phase, unconditionally.  We do *not* require
            # that pin TFs have already been observed: the PTU is parked at
            # HOME during beam phase, so pins are typically out of FOV at
            # this exact moment.  STOW_BOTH parks the right arm at
            # right_start (which also frees the PTU command, see
            # FrankAtls.publish_planner_diagnostics → PTU goes to PIN_VIEW)
            # before advancing to PICK_PIN.
            self._pin_phase_active = True
            self._goto(State.STOW_BOTH)
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
        """Pick the next pair to assemble.

        Pair ``PRIORITY_PAIR_IDX`` (default 1) is attempted first
        whenever it still has at least one unconverged beam.  All
        other pairs fall back to the original smallest-current-distance
        rule.

        ``hole_positions`` is guaranteed non-None by ``optimise()``'s
        precondition check.
        """
        PRIORITY_PAIR_IDX = 1
        if 0 <= PRIORITY_PAIR_IDX < len(self._hole_pairs):
            a, b = self._hole_pairs[PRIORITY_PAIR_IDX]
            if not (a in self._convergence and b in self._convergence):
                return PRIORITY_PAIR_IDX

        best_idx, best_score = None, float("inf")
        for pair_idx, (a, b) in enumerate(self._hole_pairs):
            if pair_idx == PRIORITY_PAIR_IDX:
                continue
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

        # Clear stale snap-fired flags so a non-DESCENDING cycle never
        # appears to have fired. ``_grad_descending`` is the only producer.
        self._descent_snap_fired_l = False
        self._descent_snap_fired_r = False

        s = self._state
        if s == State.GO_HOME:
            self._grad_go_home()
        elif s == State.MOVE_TO_PREGRASP:
            self._grad_pregrasp()
        elif s in (State.RECOVERY_MOVE_UP,State.MOVE_AWAY):
            self._grad_lift()
        elif s == State.DESCENDING:
            self._grad_descending()
        elif s == State.DUAL_ASSEMBLE:
            self._grad_dual_assemble()
        elif s in (
            State.STOW_BOTH,
            State.MOVE_TO_PIN_PREGRASP,
            State.DESCEND_TO_PIN,
            State.LIFT_PIN,
            State.ROTATE_PIN_INWARD,
            State.MOVE_TO_HOLE_PREGRASP,
            State.INSERT_PIN,
            State.RETREAT_PIN,
            State.PIN_RECOVERY_MOVE_UP,
        ):
            self._grad_pin_phase()
        # PICK_TASK / CLOSE_GRIPPER / READY / RELEASE_GRIPPER /
        # RECOVERY_RELEASE / DONE / PICK_PIN / CLOSE_PIN_GRIPPER /
        # RELEASE_PIN / PIN_RECOVERY_RELEASE / ALL_DONE → zero gradient.
        return self._gradients

    def _grad_go_home(self) -> None:
        """Drive both arms to their start posture simultaneously."""
        loss_l = ((self._states.left_pose  - self.left_start)  ** 2).sum()
        loss_r = ((self._states.right_pose - self.right_start) ** 2).sum()
        self._gradients.left_pose  = grad(loss_l, self._states.left_pose,  retain_graph=True)[0]
        self._gradients.right_pose = grad(loss_r, self._states.right_pose, retain_graph=True)[0]

    def _grad_lift(self) -> None:
        """Pull up each arm to a predefined height, this does not require knowing the beam positions"""
        z_target = torch.tensor(MOVE_UP_Z, dtype=torch.float32, device=self.params.device)
        loss_l = (self._states.left_pose[2]-z_target)**2
        loss_r = (self._states.right_pose[2]-z_target)**2
        self._gradients.left_pose = grad(loss_l,self._states.left_pose, retain_graph=True)[0]
        self._gradients.right_pose = grad(loss_r,self._states.right_pose, retain_graph=True)[0]

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

        # Snap: inside descent_snap_radius replace the quadratic gradient
        # with one whose integrator step lands the planner state on the
        # target.  delta = LEARNING_RATE · g  →
        # set g = (x - x*) / LEARNING_RATE  so delta = (x - x*) and
        # x_new = x - delta = x*.  ``_limit_ee_delta`` and ``_clamp_ee_poses``
        # still apply — they remain the single source of truth for workspace
        # bounds and per-step displacement caps, and we deliberately do not
        # bypass them here.
        self._descent_snap_fired_l = False
        self._descent_snap_fired_r = False
        snap_r = DESCENT_SNAP_RADIUS
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

        # Snap: inside assemble_snap_radius (xy distance to beam_goal) replace
        # the quadratic gradient with one whose integrator step lands the beam
        # state on the goal — same trick as ``_grad_descending``. Only the
        # active pair's beams are eligible; converged beams are skipped (their
        # gradient is already zero and we don't want to disturb them).
        self._assemble_snap_fired = [False] * self._state_params.no_beams
        snap_r = ASSEMBLE_SNAP_RADIUS
        if snap_r > 0.0 and LEARNING_RATE > 0.0:
            active = ()
            if self._active_pair_idx is not None:
                active = self._hole_pairs[self._active_pair_idx]
            with torch.no_grad():
                for k in active:
                    if k in self._convergence:
                        continue
                    diff_xy = self._states.beam_poses[k, :2] - self._states.beam_goal[k, :2]
                    dist_xy = float((diff_xy ** 2).sum() ** 0.5)
                    if 0.0 < dist_xy < snap_r:
                        snap_g = (
                            self._states.beam_poses[k] - self._states.beam_goal[k]
                        ).detach() / LEARNING_RATE
                        beam_grad[k] = snap_g
                        self._assemble_snap_fired[k] = True
                        self._assemble_snap_count += 1

        self._gradients.beam_poses = beam_grad

        # Drive each arm with the beam gradient of its assigned beam — the
        # arm is grasped, so the beam moves with the EE.
        # Zero the z component: arms stay at grasp_z throughout assembly.
        left_g  = beam_grad[self._left_index].clone()
        right_g = beam_grad[self._right_index].clone()
        self._gradients.left_pose  = left_g
        self._gradients.right_pose = right_g

    # ------------------------------------------------------------------
    # Pin phase — single-arm controller, gates, and selection.
    # ------------------------------------------------------------------

    def _grad_pin_phase(self) -> None:
        """Pin-phase controller.

        Right arm always pursues ``_right_pin_home`` for the entire pin
        phase (the beam-phase home shifted by ``PIN_HOME_Y_OFFSET`` away
        from y = 0).  Left arm pursues a state-specific target built by
        ``_left_target_for_state``; in ``STOW_BOTH`` the left target is
        ``_left_pin_home`` so both arms park at the pin-phase home before
        the PTU is commanded to PIN_VIEW.

        Snap branch reuses ``DESCENT_SNAP_RADIUS`` for ``DESCEND_TO_PIN``
        and ``ASSEMBLE_SNAP_RADIUS`` for ``INSERT_PIN`` so the pin pickup /
        insertion gradient lands exactly on the target inside the radius
        (same trick as ``_grad_descending``).
        """
        loss_r = ((self._states.right_pose - self._right_pin_home) ** 2).sum()
        self._gradients.right_pose = grad(
            loss_r, self._states.right_pose, retain_graph=True
        )[0]

        target = self._left_target_for_state()
        if target is None:
            return  # missing perception → left holds

        loss_l = ((self._states.left_pose - target) ** 2).sum()
        g_l = grad(loss_l, self._states.left_pose, retain_graph=True)[0]

        snap_r = 0.0
        if self._state == State.DESCEND_TO_PIN:
            snap_r = DESCENT_SNAP_RADIUS
        elif self._state == State.INSERT_PIN:
            snap_r = ASSEMBLE_SNAP_RADIUS

        if snap_r > 0.0 and LEARNING_RATE > 0.0:
            with torch.no_grad():
                diff = self._states.left_pose - target
                dist = float((diff[:3] ** 2).sum() ** 0.5)
                if 0.0 < dist < snap_r:
                    g_l = diff.detach() / LEARNING_RATE

        self._gradients.left_pose = g_l

    def _left_target_for_state(self) -> Optional[torch.Tensor]:
        """Build the 5-DOF left-arm target for the current pin-phase state.

        Returns ``None`` when required perception is missing — the
        gradient producer interprets that as "left holds".
        """
        s = self._state
        device = self.params.device

        def _t(x, y, z, sin_yaw, cos_yaw) -> torch.Tensor:
            return torch.tensor(
                [float(x), float(y), float(z), float(sin_yaw), float(cos_yaw)],
                dtype=torch.float32, device=device,
            )

        if s == State.STOW_BOTH:
            # Drive left to the pin-phase home; right is independently
            # pulled there by ``_grad_pin_phase``'s right loss.
            return self._left_pin_home.detach().clone()

        if s in (State.MOVE_TO_PIN_PREGRASP, State.DESCEND_TO_PIN):
            if self._active_pin_idx is None or self._pin_positions is None:
                return None
            pin = self._pin_positions[self._active_pin_idx]
            z_offset = (PIN_PREGRASP_OFFSET_Z
                        if s == State.MOVE_TO_PIN_PREGRASP else 0.0)
            return _t(pin[0], pin[1], pin[2] + z_offset,
                      PIN_GRASP_SIN, PIN_GRASP_COS)

        if s == State.LIFT_PIN:
            cur = self._states.left_pose.detach()
            return _t(cur[0], cur[1], MOVE_UP_Z, PIN_GRASP_SIN, PIN_GRASP_COS)

        if s == State.ROTATE_PIN_INWARD:
            cur = self._states.left_pose.detach()
            return _t(cur[0], cur[1], MOVE_UP_Z,
                      INWARD_YAW_LEFT_SIN, INWARD_YAW_LEFT_COS)

        if s in (State.MOVE_TO_HOLE_PREGRASP, State.INSERT_PIN, State.RETREAT_PIN):
            if self._active_hole_pair_idx is None or self._hole_positions is None:
                return None
            a, b = self._hole_pairs[self._active_hole_pair_idx]
            mid = 0.5 * (self._hole_positions[a] + self._hole_positions[b])
            z = float(mid[2]) if s == State.INSERT_PIN else MOVE_UP_Z
            return _t(mid[0], mid[1], z,
                      INWARD_YAW_LEFT_SIN, INWARD_YAW_LEFT_COS)

        if s == State.PIN_RECOVERY_MOVE_UP:
            cur = self._states.left_pose.detach()
            return _t(cur[0], cur[1], MOVE_UP_Z, cur[3], cur[4])

        return None

    def _enter_pick_pin(self) -> None:
        """Greedy pin/pair assignment by closest-distance scoring.

        Iterates the cartesian product of (unplaced pins) × (converged
        but unfilled hole pairs) and picks the minimum
        ``‖pin − midpoint‖`` candidate.  Resets the per-active counters
        before the next state runs.  If no candidate exists, transitions
        to ``ALL_DONE``.
        """
        # Latch all beams as converged for the duration of pin phase.
        # Why: the PTU slews to PIN_VIEW for pin perception, which
        # changes the camera extrinsics enough that the beam AprilTag
        # poses drift out of the hole-distance threshold even when the
        # physical assembly has not moved.  Re-evaluating beam
        # convergence against this drifted perception would spuriously
        # un-latch pairs and block PICK_PIN.
        for a, b in self._hole_pairs:
            self._convergence[a] = True
            self._convergence[b] = True

        best, best_score = None, float("inf")
        for pair_idx, (a, b) in enumerate(self._hole_pairs):
            if pair_idx in self._pinned_pairs:
                continue
            if not (a in self._convergence and b in self._convergence):
                continue
            mid = 0.5 * (self._hole_positions[a] + self._hole_positions[b])
            for pin_idx, placed in enumerate(self._pin_inserted):
                if placed:
                    continue
                d = float(np.linalg.norm(self._pin_positions[pin_idx] - mid))
                if d < best_score:
                    best_score = d
                    best = (pin_idx, pair_idx)

        if best is None:
            n_pins = len(self._pin_inserted)
            n_pins_placed = sum(self._pin_inserted)
            print(
                "[perceptive_assembly] _enter_pick_pin → ALL_DONE  "
                f"hole_pairs={len(self._hole_pairs)} "
                f"pinned_pairs={sorted(self._pinned_pairs)} "
                f"pins_registered={n_pins} pins_placed={n_pins_placed} "
                f"convergence_keys={sorted(self._convergence.keys())} "
                f"hole_positions_set={self._hole_positions is not None} "
                f"pin_positions_set={self._pin_positions is not None}"
            )
            self._active_pin_idx = None
            self._active_hole_pair_idx = None
            self._goto(State.ALL_DONE)
            return

        self._active_pin_idx, self._active_hole_pair_idx = best
        self._pin_insert_counter = 0
        self._slip_counter = 0
        self._goto(State.MOVE_TO_PIN_PREGRASP)

    # ---- Pin-phase gates ----

    def _at_pin_home(self) -> bool:
        """Both arms within ``MOVE_UP_TOL`` (Euclidean, position only) of
        their pin-phase home pose."""
        dl = float(((self._states.left_pose[:3]  - self._left_pin_home[:3])  ** 2)
                   .sum().sqrt())
        dr = float(((self._states.right_pose[:3] - self._right_pin_home[:3]) ** 2)
                   .sum().sqrt())
        return dl < MOVE_UP_TOL and dr < MOVE_UP_TOL

    def _pin_pregrasp_reached(self) -> bool:
        if self._active_pin_idx is None or self._pin_positions is None:
            return False
        pin = self._pin_positions[self._active_pin_idx]
        target = torch.tensor(
            [pin[0], pin[1], pin[2] + PIN_PREGRASP_OFFSET_Z],
            dtype=torch.float32, device=self.params.device,
        )
        return float((self._states.left_pose[:3] - target).abs().max()) < PREGRASP_TOL

    def _pin_grasp_contact(self) -> bool:
        if self._active_pin_idx is None or self._pin_positions is None:
            return False
        pin = self._pin_positions[self._active_pin_idx]
        target = torch.tensor(
            [pin[0], pin[1], pin[2]],
            dtype=torch.float32, device=self.params.device,
        )
        return float((self._states.left_pose[:3] - target).abs().max()) < GRASP_POS_TOL

    def _left_lift_reached(self) -> bool:
        return abs(float(self._states.left_pose[2]) - MOVE_UP_Z) < MOVE_UP_TOL

    def _pin_yaw_reached(self) -> bool:
        cur_sin = float(self._states.left_pose[3])
        cur_cos = float(self._states.left_pose[4])
        cur_yaw = float(np.arctan2(cur_sin, cur_cos))
        target_yaw = float(np.arctan2(INWARD_YAW_LEFT_SIN, INWARD_YAW_LEFT_COS))
        err = (cur_yaw - target_yaw + np.pi) % (2 * np.pi) - np.pi
        return abs(err) < PIN_YAW_TOL

    def _hole_pregrasp_reached(self) -> bool:
        if self._active_hole_pair_idx is None or self._hole_positions is None:
            return False
        a, b = self._hole_pairs[self._active_hole_pair_idx]
        mid = 0.5 * (self._hole_positions[a] + self._hole_positions[b])
        target = torch.tensor(
            [mid[0], mid[1], MOVE_UP_Z],
            dtype=torch.float32, device=self.params.device,
        )
        return float((self._states.left_pose[:3] - target).abs().max()) < PIN_INSERT_PREGRASP_TOL

    def _check_pin_insert_progress(self) -> bool:
        """Advance / reset the insert hysteresis counter and return True
        when the pin has stayed inside ``PIN_INSERT_TOL`` of the hole-pair
        midpoint for ``PIN_CONVERGENCE_HYSTERESIS`` consecutive steps.

        This is the gate that controls the "8 mm regrasp loop": only when
        this returns True does ``RELEASE_PIN`` fire; otherwise the planner
        is forced through recovery and re-grasps the pin.
        """
        if self._active_hole_pair_idx is None or self._hole_positions is None:
            return False
        a, b = self._hole_pairs[self._active_hole_pair_idx]
        mid_xy = 0.5 * (self._hole_positions[a, :2] + self._hole_positions[b, :2])
        ee_xy = self._states.left_pose[:2].detach().cpu().numpy()
        d = float(np.linalg.norm(ee_xy - mid_xy))
        if d < PIN_INSERT_TOL:
            self._pin_insert_counter += 1
        else:
            self._pin_insert_counter = 0
        return self._pin_insert_counter >= PIN_CONVERGENCE_HYSTERESIS

    def _pin_slipped(self) -> bool:
        if self._active_hole_pair_idx is None or self._hole_positions is None:
            return False
        a, b = self._hole_pairs[self._active_hole_pair_idx]
        mid_xy = 0.5 * (self._hole_positions[a, :2] + self._hole_positions[b, :2])
        ee_xy = self._states.left_pose[:2].detach().cpu().numpy()
        return float(np.linalg.norm(ee_xy - mid_xy)) > ASSEMBLE_SLIP_DIST

    # ------------------------------------------------------------------
    # Gates — small predicates on the latest losses.
    # ------------------------------------------------------------------

    def _lift_reached(self) -> bool:
        dl = abs(float(self._states.left_pose[2]-MOVE_UP_Z))
        dr = abs(float(self._states.right_pose[2]-MOVE_UP_Z))
        return dl < MOVE_UP_TOL and dr < MOVE_UP_TOL

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
            z_min = 0.75
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
