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
   ``CONFIG.grasp.pregrasp_tol`` as a single MSE-sum bound.
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
   beam pose by > ``CONFIG.beam_assemble.slip_dist`` (default 3 cm) for
   ``CONFIG.beam_assemble.slip_steps`` consecutive steps (default 20), the planner
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
# Tunables — all values live in ``perceptive_config.CONFIG``, grouped
# hierarchically by FSM phase / responsibility.  The module-level names
# below are kept as backwards-compat aliases for external callers
# (tests, smoke tests, trace scripts) that import individual constants.
# Internal code reads ``CONFIG.<group>.<name>`` directly.
# ---------------------------------------------------------------------------

from beam_data_gen.traj_opt.perceptive_config import CONFIG

# Beam-phase grasp gates.
PREGRASP_TOL  = CONFIG.grasp.pregrasp_tol
GRASP_POS_TOL = CONFIG.grasp.grasp_pos_tol

# DUAL_ASSEMBLE convergence + slip + timeout.
HOLE_CONVERGENCE_THRESHOLD = CONFIG.beam_assemble.hole_convergence_threshold
CONVERGENCE_HYSTERESIS     = CONFIG.beam_assemble.convergence_hysteresis
ASSEMBLE_SLIP_DIST         = CONFIG.beam_assemble.slip_dist
ASSEMBLE_SLIP_STEPS        = CONFIG.beam_assemble.slip_steps
ASSEMBLE_TIMEOUT_STEPS     = CONFIG.beam_assemble.timeout_steps

# Per-state step budgets.
DESCEND_TIMEOUT_STEPS = CONFIG.descending.timeout_steps
GO_HOME_TIMEOUT_STEPS = CONFIG.go_home.timeout_steps

# MOVE_AWAY / RECOVERY_MOVE_UP / LIFT_PIN / RETREAT_PIN gate.
MOVE_UP_TOL = CONFIG.move_up.tol
MOVE_UP_Z   = CONFIG.move_up.z

# Gradient mixing + integrator step.
HOLE_GRADIENT_WEIGHT = CONFIG.gradient.hole_weight
YAW_GRADIENT_WEIGHT  = CONFIG.gradient.yaw_weight
LEARNING_RATE        = CONFIG.gradient.learning_rate

# Snap radii.
DESCENT_SNAP_RADIUS       = CONFIG.snap.descent_radius
ASSEMBLE_SNAP_RADIUS      = CONFIG.snap.assemble_radius
PIN_PREGRASP_SNAP_RADIUS  = CONFIG.snap.pin_pregrasp_radius
PIN_INSERTION_SNAP_RADIUS = CONFIG.snap.pin_insertion_radius

# Pin-phase pickup.
PIN_PREGRASP_OFFSET_Z = CONFIG.pin.pickup.pregrasp_offset_z
PIN_GRASP_OFFSET_Z    = CONFIG.pin.pickup.grasp_offset_z
PIN_GRASP_SIN         = CONFIG.pin.pickup.grasp_yaw_sin
PIN_GRASP_COS         = CONFIG.pin.pickup.grasp_yaw_cos

# Pin-phase rotate.
PIN_YAW_TOL          = CONFIG.pin.rotate.yaw_tol
INWARD_YAW_LEFT_SIN  = CONFIG.pin.rotate.inward_yaw_left_sin
INWARD_YAW_LEFT_COS  = CONFIG.pin.rotate.inward_yaw_left_cos

# Pin-phase insertion.
PIN_INSERT_PREGRASP_TOL    = CONFIG.pin.insertion.pregrasp_tol
PIN_INSERT_Z_TOL           = CONFIG.pin.insertion.z_tol
PIN_INSERT_TOL             = CONFIG.pin.insertion.xy_tol
PIN_CONVERGENCE_HYSTERESIS = CONFIG.pin.insertion.convergence_hysteresis
PIN_INSERT_TIMEOUT         = CONFIG.pin.insertion.timeout_steps

# Pin-phase home.
PIN_HOME_Y_OFFSET = CONFIG.pin.home.y_offset


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
    LIFT_PIN               = 17  # lift pin to CONFIG.move_up.z, yaw = 0
    ROTATE_PIN_INWARD      = 18  # rotate left wrist 90° inward AND move EE to rotate_pin_target_xyz
    MOVE_TO_HOLE_PREGRASP  = 19  # left moves over hole-pair midpoint, inward yaw
    COMPUTE_PIN_OFFSET     = 27  # hold rotate target; sample pin↔hand offset for pregrasp/insertion correction
    INSERT_PIN             = 20  # descend pin into midpoint; 8 mm convergence gate
    RELEASE_PIN            = 21  # latch left gripper open (1 step) on success
    RETREAT_PIN            = 22  # lift to CONFIG.move_up.z then PICK_PIN
    PIN_RECOVERY_RELEASE   = 23  # open gripper on grasp/insert failure (1 step)
    PIN_RECOVERY_MOVE_UP   = 24  # lift, retry MOVE_TO_PIN_PREGRASP for same pin
    ALL_DONE               = 25  # terminal: all pins placed
    RECOVER_INSERTION_PREGRASP = 26  # lift recovery_z_delta then retry MOVE_TO_HOLE_PREGRASP

    # Pulls the two arms apart along y by CONFIG.recovery_perturbation.y_delta
    # before RECOVERY_RELEASE.  Inserted on the DUAL_ASSEMBLE → recovery path
    # only; gripper stays closed so the beams are physically separated.
    RECOVERY_PERTURBATION  = 28


class PtuView(IntEnum):
    """Desired PTU viewpoint, selected per FSM state.

    FORWARD points the camera at the workspace centre (good for beam +
    hole AprilTag tracking).  SIDE points to the pin staging area (good
    for pin AprilTag tracking).  The mapping is consumed by the FSM
    adapter, which translates it to concrete pan/tilt setpoints.
    """
    FORWARD = 0
    SIDE    = 1


# Per-state PTU viewpoint.  States not listed default to FORWARD.
_PTU_SIDE_STATES: frozenset = frozenset({
    # Pin pickup phase: PTU points to the pin staging area so AprilTags
    # on the pins are in view.
    State.STOW_BOTH,
    State.PICK_PIN,
    State.MOVE_TO_PIN_PREGRASP,
    State.DESCEND_TO_PIN,
    State.CLOSE_PIN_GRIPPER,
    State.LIFT_PIN,
    # Post-insertion retreat + pickup-failure recovery: swing back to
    # the pin staging area before re-attempting PICK_PIN.
    State.RETREAT_PIN,
    State.PIN_RECOVERY_RELEASE,
    State.PIN_RECOVERY_MOVE_UP,
})


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
        h[1] = h[1] + CONFIG.pin.home.y_offset
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

        # ---- RECOVERY_PERTURBATION targets. Captured at state entry from
        # the current EE pose; cleared on exit. xyz only (orientation held
        # via gradient on first 3 dims).
        self._perturbation_target_left:  Optional[torch.Tensor] = None
        self._perturbation_target_right: Optional[torch.Tensor] = None

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
        # Absolute z target for RECOVER_INSERTION_PREGRASP — captured
        # at the moment MOVE_TO_HOLE_PREGRASP times out so the lift is
        # always ``recovery_z_delta`` above the stuck pose.
        self._insertion_recovery_target_z: Optional[float] = None

        # ---- Estimated pin position (calibration result, fed by wrapper). ----
        # The wrapper (FrankAtls) runs a high-rate (~30 Hz) calibrator that
        # samples ``pin_world − hand_world`` over ~3 s while the planner
        # holds at the rotate target, then pushes the averaged translation
        # here via ``set_estimated_pin_position``.  Stored as
        # ``(translation_xyz_world, quaternion_xyzw)`` for API symmetry,
        # but **only the translation is consumed** by the planner — the
        # pin orientation is taken to equal the hand orientation, so the
        # hand yaw target during MOVE_TO_HOLE_PREGRASP / INSERT_PIN /
        # RECOVER_INSERTION_PREGRASP is unmodified by this offset (the
        # wrapper currently always pushes identity for the quaternion).
        # ``None`` means "not yet calibrated for the active pin";
        # COMPUTE_PIN_OFFSET holds at the rotate target until this
        # becomes non-None.  Reset to None in ``_enter_pick_pin`` and in
        # the RELEASE_PIN idle transition.
        self._estimated_pin_position: Optional[Tuple[np.ndarray, np.ndarray]] = None

        # Measured (FK-on-/joint_states) EE positions in base_link, pushed
        # by the wrapper each cycle via ``set_actual_hand_poses``.  Used
        # *only* for pin-offset calibration so the offset captures the
        # geometric grasp bias rather than planner-vs-controller tracking
        # error (the planner's ``_states.left_pose`` is the gradient-evolved
        # commanded pose, not the actual EE).  Falls back to
        # ``_states.left_pose`` when unset (sim / unit tests).
        self._actual_hand_left:  Optional[np.ndarray] = None
        self._actual_hand_right: Optional[np.ndarray] = None

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

    def set_estimated_pin_position(
        self,
        translation: Optional[np.ndarray],
        quaternion:  Optional[np.ndarray] = None,
    ) -> None:
        """Push the 6-DOF hand→pin transform from the wrapper-side
        calibrator.  ``translation`` is the world-frame ``pin - hand``
        position (3,); ``quaternion`` is the world-frame ``q_pin · q_hand⁻¹``
        rotation (4, xyzw).  Pass ``translation=None`` to clear.

        Calibration is captured at the inward rotate yaw, so the planner
        treats the offset as world-frame-constant for the duration of
        the current pin.  ``_enter_pick_pin`` resets this back to None
        so each new pin gets a fresh calibration window.
        """
        if translation is None:
            self._estimated_pin_position = None
            return
        t = np.asarray(translation, dtype=np.float64).reshape(3)
        if quaternion is None:
            q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        else:
            q = np.asarray(quaternion, dtype=np.float64).reshape(4)
            n = float(np.linalg.norm(q))
            if n > 0.0:
                q = q / n
        self._estimated_pin_position = (t, q)

    def set_actual_hand_poses(
        self,
        left_xyz:  Optional[np.ndarray],
        right_xyz: Optional[np.ndarray],
    ) -> None:
        """Push the measured EE positions (FK on real /joint_states) for
        the current cycle.  Used by COMPUTE_PIN_OFFSET so the calibrated
        offset reflects the true pin↔hand geometry rather than the
        planner's commanded pose."""
        self._actual_hand_left  = (
            None if left_xyz  is None else np.asarray(left_xyz,  dtype=np.float64))
        self._actual_hand_right = (
            None if right_xyz is None else np.asarray(right_xyz, dtype=np.float64))

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

    def desired_ptu_view(self) -> PtuView:
        """Desired PTU viewpoint for the current FSM state.

        Beam phase, hole-insertion phase (``ROTATE_PIN_INWARD`` …
        ``RELEASE_PIN``), and ``ALL_DONE`` resolve to ``FORWARD`` so the
        camera tracks beams and holes accurately.  The pin pickup
        sub-phase and recovery/retreat states resolve to ``SIDE`` so
        the camera tracks the pin staging area.
        """
        if self._state in _PTU_SIDE_STATES:
            return PtuView.SIDE
        return PtuView.FORWARD

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
                delta_l = self._limit_ee_delta(CONFIG.gradient.learning_rate * gradients.left_pose)
                delta_r = self._limit_ee_delta(CONFIG.gradient.learning_rate * gradients.right_pose)
                self._states.left_pose  = self._states.left_pose  - delta_l
                self._states.right_pose = self._states.right_pose - delta_r
                self._states.beam_poses = self._states.beam_poses - CONFIG.gradient.learning_rate * gradients.beam_poses
                self._states.pregrasp   = self._states.pregrasp   - CONFIG.gradient.learning_rate * gradients.pregrasp

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
            budget = (f"  budget={CONFIG.descending.timeout_steps - self._state_step}"
                      f"/{CONFIG.descending.timeout_steps} steps")
        elif self._state == State.DUAL_ASSEMBLE:
            budget = (f"  budget={CONFIG.beam_assemble.timeout_steps - self._state_step}"
                      f"/{CONFIG.beam_assemble.timeout_steps}  slip={self._slip_counter}/{CONFIG.beam_assemble.slip_steps}")
        elif self._state == State.GO_HOME:
            budget = (f"  budget={CONFIG.go_home.timeout_steps - self._state_step}"
                      f"/{CONFIG.go_home.timeout_steps} steps")
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
                flag = " ✓" if d < CONFIG.beam_assemble.hole_convergence_threshold else ""
                parts.append(f"({a},{b})={d*1e3:.1f}mm{flag}")
            _log(f"  hole_dists: {', '.join(parts)}"
                 f"  [thresh={CONFIG.beam_assemble.hole_convergence_threshold*1e3:.0f}mm]")

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
                 f"  [gate<{CONFIG.grasp.pregrasp_tol*1e3:.0f}mm]")
            _log(f"  grasp_xyz_max_err:    L={l_grasp_axis*1e3:.1f}mm  R={r_grasp_axis*1e3:.1f}mm"
                 f"  [gate<{CONFIG.grasp.grasp_pos_tol*1e3:.0f}mm]")

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
            # The 6-DOF estimated pin position belongs to *this* pin's
            # grasp — clear it so the next pin pickup must recalibrate.
            self._estimated_pin_position = None
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
            elif (not check_distance_only) and self._state_step >= CONFIG.go_home.timeout_steps:
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
            elif (not check_distance_only) and self._state_step >= CONFIG.descending.timeout_steps:
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
                if (self._convergence_counter.get(a, 0) >= CONFIG.beam_assemble.convergence_hysteresis
                        and self._convergence_counter.get(b, 0) >= CONFIG.beam_assemble.convergence_hysteresis):
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
                if self._slip_counter >= CONFIG.beam_assemble.slip_steps:
                    self._slip_counter = 0
                    self._goto(State.RECOVERY_PERTURBATION)
                    self._enter_recovery_perturbation()
                    return
                if self._state_step >= CONFIG.beam_assemble.timeout_steps:
                    self._goto(State.RECOVERY_PERTURBATION)
                    self._enter_recovery_perturbation()

        elif self._state == State.RECOVERY_PERTURBATION:
            # Pull the beams apart in y while the grippers are still closed,
            # then fall through to RECOVERY_RELEASE.  Time-out gracefully so
            # a missing target (e.g. enter() not called) cannot stall the FSM.
            reached = (not skip_distance) and self._perturbation_reached()
            timed_out = ((not check_distance_only)
                         and self._state_step >= CONFIG.recovery_perturbation.timeout_steps)
            if reached or timed_out:
                self._perturbation_target_left = None
                self._perturbation_target_right = None
                self._goto(State.RECOVERY_RELEASE)
                self._evaluate_idle_transitions()  # opens grippers, → MOVE_UP

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
            elif (not check_distance_only) and self._state_step >= CONFIG.descending.timeout_steps:
                self._goto(State.PIN_RECOVERY_RELEASE)
                self._evaluate_idle_transitions()  # → PIN_RECOVERY_MOVE_UP

        elif self._state == State.LIFT_PIN:
            if (not skip_distance) and self._left_lift_reached():
                self._goto(State.ROTATE_PIN_INWARD)

        elif self._state == State.ROTATE_PIN_INWARD:
            if (not skip_distance) and self._rotate_pin_pose_reached():
                # Calibration is now driven by the wrapper at TF rate; the
                # planner only holds the rotate target and waits for the
                # 6-DOF transform to be pushed via set_estimated_pin_position.
                self._estimated_pin_position = None
                self._goto(State.COMPUTE_PIN_OFFSET)

        elif self._state == State.COMPUTE_PIN_OFFSET:
            # Hold at the rotate target until the wrapper pushes a
            # calibrated 6-DOF transform.  Gradient producer keeps the EE
            # on target via _grad_pin_phase; we only watch the variable.
            if self._estimated_pin_position is not None:
                self._goto(State.MOVE_TO_HOLE_PREGRASP)

        elif self._state == State.MOVE_TO_HOLE_PREGRASP:
            if (not skip_distance) and self._hole_pregrasp_reached():
                self._goto(State.INSERT_PIN)
            elif (not check_distance_only) and self._state_step >= CONFIG.pin.insertion.pregrasp_timeout_steps:
                # Snapshot stuck z, then lift recovery_z_delta above it.
                self._insertion_recovery_target_z = (
                    float(self._states.left_pose[2])
                    + CONFIG.pin.insertion.recovery_z_delta
                )
                self._goto(State.RECOVER_INSERTION_PREGRASP)

        elif self._state == State.RECOVER_INSERTION_PREGRASP:
            if (not skip_distance) and self._insertion_recovery_lift_reached():
                self._goto(State.MOVE_TO_HOLE_PREGRASP)

        elif self._state == State.INSERT_PIN:
            # Linear pipeline — no retry.  Two exits, both → RELEASE_PIN:
            #   1. success: z and xy both at the hole-pair midpoint;
            #   2. step budget exhausted (PIN_INSERT_TIMEOUT) — gives
            #      up and moves on so a bad insertion can't deadlock
            #      the FSM.
            if (not skip_distance) and self._check_pin_insert_progress():
                self._goto(State.RELEASE_PIN)
                self._evaluate_idle_transitions()  # opens left, → RETREAT_PIN
                return
            if (not check_distance_only) and self._state_step >= CONFIG.pin.insertion.timeout_steps:
                self._pin_insert_counter = 0
                self._goto(State.RELEASE_PIN)
                self._evaluate_idle_transitions()
                return

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
                if self._convergence_counter[k] >= CONFIG.beam_assemble.convergence_hysteresis:
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
                return d < CONFIG.beam_assemble.hole_convergence_threshold
        return False

    def _active_pair_converged(self) -> bool:
        if self._active_pair_idx is None:
            return False
        a, b = self._hole_pairs[self._active_pair_idx]
        d = float(np.linalg.norm(self._hole_positions[a] - self._hole_positions[b]))
        return d < CONFIG.beam_assemble.hole_convergence_threshold

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
        elif s == State.RECOVERY_PERTURBATION:
            self._grad_recovery_perturbation()
        elif s in (
            State.STOW_BOTH,
            State.MOVE_TO_PIN_PREGRASP,
            State.DESCEND_TO_PIN,
            State.LIFT_PIN,
            State.ROTATE_PIN_INWARD,
            State.COMPUTE_PIN_OFFSET,
            State.MOVE_TO_HOLE_PREGRASP,
            State.INSERT_PIN,
            State.RETREAT_PIN,
            State.PIN_RECOVERY_MOVE_UP,
            State.RECOVER_INSERTION_PREGRASP,
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
        z_target = torch.tensor(CONFIG.move_up.z, dtype=torch.float32, device=self.params.device)
        loss_l = (self._states.left_pose[2]-z_target)**2
        loss_r = (self._states.right_pose[2]-z_target)**2
        self._gradients.left_pose = grad(loss_l,self._states.left_pose, retain_graph=True)[0]
        self._gradients.right_pose = grad(loss_r,self._states.right_pose, retain_graph=True)[0]

    def _enter_recovery_perturbation(self) -> None:
        """Capture xyz targets that pull both arms apart along y by
        ``CONFIG.recovery_perturbation.y_delta``.

        Direction is decided by the entry y-ordering of the two arms — the
        arm currently at higher y moves further in +y, the other in -y —
        so the beams always separate regardless of arm-to-beam assignment.
        """
        delta = CONFIG.recovery_perturbation.y_delta
        with torch.no_grad():
            l_xyz = self._states.left_pose[:3].detach().clone()
            r_xyz = self._states.right_pose[:3].detach().clone()
            sign_l = 1.0 if float(l_xyz[1]) >= float(r_xyz[1]) else -1.0
            l_target = l_xyz.clone()
            r_target = r_xyz.clone()
            l_target[1] = l_target[1] + sign_l * delta
            r_target[1] = r_target[1] - sign_l * delta
        self._perturbation_target_left  = l_target
        self._perturbation_target_right = r_target

    def _grad_recovery_perturbation(self) -> None:
        """Drive each EE xyz to its captured perturbation target."""
        if (self._perturbation_target_left is None
                or self._perturbation_target_right is None):
            return
        lt = self._perturbation_target_left
        rt = self._perturbation_target_right
        loss_l = ((self._states.left_pose[:3]  - lt) ** 2).sum()
        loss_r = ((self._states.right_pose[:3] - rt) ** 2).sum()
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
        snap_r = CONFIG.snap.descent_radius
        if snap_r > 0.0 and CONFIG.gradient.learning_rate > 0.0:
            with torch.no_grad():
                if 0.0 < dist_l < snap_r:
                    g_l = (self._states.left_pose  - left_target).detach()  / CONFIG.gradient.learning_rate
                    self._descent_snap_fired_l = True
                    self._descent_snap_count_l += 1
                if 0.0 < dist_r < snap_r:
                    g_r = (self._states.right_pose - right_target).detach() / CONFIG.gradient.learning_rate
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
        beam_grad = beam_grad + CONFIG.gradient.hole_weight * hole_grad
        beam_grad[:, 3:5] *= CONFIG.gradient.yaw_weight

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
        snap_r = CONFIG.snap.assemble_radius
        if snap_r > 0.0 and CONFIG.gradient.learning_rate > 0.0:
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
                        ).detach() / CONFIG.gradient.learning_rate
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
        phase (the beam-phase home shifted by ``CONFIG.pin.home.y_offset`` away
        from y = 0).  Left arm pursues a state-specific target built by
        ``_left_target_for_state``; in ``STOW_BOTH`` the left target is
        ``_left_pin_home`` so both arms park at the pin-phase home before
        the PTU is commanded to PIN_VIEW.

        Snap branch uses ``CONFIG.snap.descent_radius`` for ``DESCEND_TO_PIN``,
        ``CONFIG.snap.pin_pregrasp_radius`` for ``MOVE_TO_HOLE_PREGRASP`` and
        ``CONFIG.snap.pin_insertion_radius`` for ``INSERT_PIN`` so the pickup /
        pre-insertion / insertion gradient lands exactly on the target
        inside the radius (same trick as ``_grad_descending``).
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
            snap_r = CONFIG.snap.descent_radius
        elif self._state == State.MOVE_TO_HOLE_PREGRASP:
            snap_r = CONFIG.snap.pin_pregrasp_radius
        elif self._state == State.INSERT_PIN:
            snap_r = CONFIG.snap.pin_insertion_radius

        if snap_r > 0.0 and CONFIG.gradient.learning_rate > 0.0:
            with torch.no_grad():
                diff = self._states.left_pose - target
                dist = float((diff[:3] ** 2).sum() ** 0.5)
                if 0.0 < dist < snap_r:
                    g_l = diff.detach() / CONFIG.gradient.learning_rate

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
            z_offset = (CONFIG.pin.pickup.pregrasp_offset_z
                        if s == State.MOVE_TO_PIN_PREGRASP else CONFIG.pin.pickup.grasp_offset_z)
            return _t(pin[0], pin[1], pin[2] + z_offset,
                      CONFIG.pin.pickup.grasp_yaw_sin, CONFIG.pin.pickup.grasp_yaw_cos)

        if s == State.LIFT_PIN:
            cur = self._states.left_pose.detach()
            return _t(cur[0], cur[1], CONFIG.move_up.z, CONFIG.pin.pickup.grasp_yaw_sin, CONFIG.pin.pickup.grasp_yaw_cos)

        if s in (State.ROTATE_PIN_INWARD, State.COMPUTE_PIN_OFFSET):
            return _t(CONFIG.pin.offset.rotate_target_x,
                      CONFIG.pin.offset.rotate_target_y,
                      CONFIG.pin.offset.rotate_target_z,
                      CONFIG.pin.rotate.inward_yaw_left_sin, CONFIG.pin.rotate.inward_yaw_left_cos)

        if s in (State.MOVE_TO_HOLE_PREGRASP, State.INSERT_PIN, State.RETREAT_PIN):
            if self._active_hole_pair_idx is None or self._hole_positions is None:
                return None
            a, b = self._hole_pairs[self._active_hole_pair_idx]
            mid = 0.5 * (self._hole_positions[a] + self._hole_positions[b])
            x, y = float(mid[0]), float(mid[1])
            if s == State.INSERT_PIN:
                z = float(mid[2])
            elif s == State.MOVE_TO_HOLE_PREGRASP:
                z = float(mid[2]) + CONFIG.pin.insertion.pregrasp_z_delta
            else:  # RETREAT_PIN — generic lift to the workspace MOVE_UP_Z.
                z = CONFIG.move_up.z
            # Apply calibrated pin↔hand offset so the *pin* (not the hand)
            # lands on the hole-pair midpoint.  Translation only — pin
            # orientation is taken to equal hand orientation, so the
            # hand yaw target is unmodified (inward yaw).  RETREAT_PIN
            # is a generic lift and ignores the offset entirely.
            if s != State.RETREAT_PIN and self._estimated_pin_position is not None:
                t_off, _q_off = self._estimated_pin_position
                x -= float(t_off[0])
                y -= float(t_off[1])
                z -= float(t_off[2])
            return _t(x, y, z,
                      CONFIG.pin.rotate.inward_yaw_left_sin,
                      CONFIG.pin.rotate.inward_yaw_left_cos)

        if s == State.RECOVER_INSERTION_PREGRASP:
            # Lift to the snapshot z (current EE z + recovery_z_delta
            # captured at timeout), preserving xy and INWARD yaw so the
            # subsequent MOVE_TO_HOLE_PREGRASP retry comes from directly
            # above the same hole-pair midpoint.
            if self._active_hole_pair_idx is None or self._hole_positions is None:
                return None
            a, b = self._hole_pairs[self._active_hole_pair_idx]
            mid = 0.5 * (self._hole_positions[a] + self._hole_positions[b])

            cur = self._states.left_pose.detach()
            target_z = (self._insertion_recovery_target_z
                        if self._insertion_recovery_target_z is not None
                        else float(cur[2])+CONFIG.pin.insertion.recovery_z_delta)
            x, y = float(mid[0]), float(mid[1])
            if self._estimated_pin_position is not None:
                t_off, _q_off = self._estimated_pin_position
                x -= float(t_off[0])
                y -= float(t_off[1])
            return _t(x, y, target_z,
                      CONFIG.pin.rotate.inward_yaw_left_sin,
                      CONFIG.pin.rotate.inward_yaw_left_cos)

        if s == State.PIN_RECOVERY_MOVE_UP:
            cur = self._states.left_pose.detach()
            return _t(cur[0], cur[1], CONFIG.move_up.z, cur[3], cur[4])

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
        # Each new pin needs a fresh calibration — old offset belongs to
        # the previous (now placed) pin's grasp.
        self._estimated_pin_position = None
        self._goto(State.MOVE_TO_PIN_PREGRASP)

    # ---- Pin-phase gates ----

    def _at_pin_home(self) -> bool:
        """Both arms within ``CONFIG.move_up.tol`` (Euclidean, position only) of
        their pin-phase home pose."""
        dl = float(((self._states.left_pose[:3]  - self._left_pin_home[:3])  ** 2)
                   .sum().sqrt())
        dr = float(((self._states.right_pose[:3] - self._right_pin_home[:3]) ** 2)
                   .sum().sqrt())
        return dl < CONFIG.move_up.tol and dr < CONFIG.move_up.tol

    def _pin_pregrasp_reached(self) -> bool:
        if self._active_pin_idx is None or self._pin_positions is None:
            return False
        pin = self._pin_positions[self._active_pin_idx]
        target = torch.tensor(
            [pin[0], pin[1], pin[2] + CONFIG.pin.pickup.pregrasp_offset_z],
            dtype=torch.float32, device=self.params.device,
        )
        return float((self._states.left_pose[:3] - target).abs().max()) < CONFIG.pin.pickup.pregrasp_tol

    def _pin_grasp_contact(self) -> bool:
        if self._active_pin_idx is None or self._pin_positions is None:
            return False
        pin = self._pin_positions[self._active_pin_idx]
        target = torch.tensor(
            [pin[0], pin[1], pin[2]+CONFIG.pin.pickup.grasp_offset_z],
            dtype=torch.float32, device=self.params.device,
        )
        return float((self._states.left_pose[:3] - target).abs().max()) < CONFIG.grasp.grasp_pos_tol

    def _left_lift_reached(self) -> bool:
        return abs(float(self._states.left_pose[2]) - CONFIG.move_up.z) < CONFIG.move_up.tol

    def _insertion_recovery_lift_reached(self) -> bool:
        """Left EE within move_up.tol of the recovery z snapshot."""
        if self._insertion_recovery_target_z is None:
            return True  # safety: no snapshot → don't deadlock the retry.
        return (abs(float(self._states.left_pose[2]) - self._insertion_recovery_target_z)
                < CONFIG.move_up.tol)

    def _rotate_pin_pose_reached(self) -> bool:
        """Both yaw and EE XYZ within tolerance of ``rotate_pin_target_xyz``.

        Used to advance ROTATE_PIN_INWARD → COMPUTE_PIN_OFFSET only after
        the arm has actually settled at the calibration pose, so the
        observed pin↔hand offset reflects the steady-state grasp.
        """
        target = torch.tensor(
            [CONFIG.pin.offset.rotate_target_x,
             CONFIG.pin.offset.rotate_target_y,
             CONFIG.pin.offset.rotate_target_z],
            dtype=torch.float32, device=self.params.device,
        )
        d = float((self._states.left_pose[:3] - target).abs().max())
        return d < CONFIG.move_up.tol and self._pin_yaw_reached()

    def _pin_yaw_reached(self) -> bool:
        cur_sin = float(self._states.left_pose[3])
        cur_cos = float(self._states.left_pose[4])
        cur_yaw = float(np.arctan2(cur_sin, cur_cos))
        target_yaw = float(np.arctan2(CONFIG.pin.rotate.inward_yaw_left_sin, CONFIG.pin.rotate.inward_yaw_left_cos))
        err = (cur_yaw - target_yaw + np.pi) % (2 * np.pi) - np.pi
        return abs(err) < CONFIG.pin.rotate.yaw_tol

    def _hole_pregrasp_reached(self) -> bool:
        if self._active_hole_pair_idx is None or self._hole_positions is None:
            return False
        a, b = self._hole_pairs[self._active_hole_pair_idx]
        mid = 0.5 * (self._hole_positions[a] + self._hole_positions[b])
        # Gate fires when the *estimated pin* (measured_hand + offset) is
        # within tolerance of the hole-pair midpoint in xy. Falls back to
        # the planner pose when no measured FK / offset has been pushed
        # yet (sim / pre-calibration).
        if self._actual_hand_left is not None:
            hand = self._actual_hand_left.astype(np.float64)
        else:
            hand = self._states.left_pose[:3].detach().cpu().numpy().astype(np.float64)
        if self._estimated_pin_position is not None:
            offset = self._estimated_pin_position[0]
        else:
            offset = np.zeros(3, dtype=np.float64)
        estimated_pin_xy = (hand + offset)[:2]
        return float(np.max(np.abs(estimated_pin_xy - np.asarray(mid[:2], dtype=np.float64)))) < CONFIG.pin.insertion.pregrasp_tol

    def _check_pin_insert_progress(self) -> bool:
        """Advance / reset the insert hysteresis counter and return True
        when the pin EE has reached the hole-pair midpoint in z (primary
        — confirms the pin has descended into the hole) AND xy (sanity)
        for ``CONFIG.pin.insertion.convergence_hysteresis`` consecutive steps.

        Sole exit gate from ``INSERT_PIN``: there is no slip / timeout
        recovery path.  Once this returns True the FSM advances to
        ``RELEASE_PIN`` and the pipeline moves on to the next pin.
        """
        if self._active_hole_pair_idx is None or self._hole_positions is None:
            return False
        a, b = self._hole_pairs[self._active_hole_pair_idx]
        mid = 0.5 * (self._hole_positions[a] + self._hole_positions[b])
        # Gate fires when the *estimated pin* (measured_hand + offset) is
        # within tolerance of the hole-pair midpoint in z (primary) and
        # xy (sanity).  Falls back to the planner pose when no measured
        # FK / offset has been pushed yet.
        if self._actual_hand_left is not None:
            hand = self._actual_hand_left.astype(np.float64)
        else:
            hand = self._states.left_pose[:3].detach().cpu().numpy().astype(np.float64)
        if self._estimated_pin_position is not None:
            offset = self._estimated_pin_position[0]
        else:
            offset = np.zeros(3, dtype=np.float64)
        estimated_pin = hand + offset
        mid_np = np.asarray(mid, dtype=np.float64)
        z_err  = abs(float(estimated_pin[2]) - float(mid_np[2]))
        xy_err = float(np.linalg.norm(estimated_pin[:2] - mid_np[:2]))
        if z_err < CONFIG.pin.insertion.z_tol and xy_err < CONFIG.pin.insertion.xy_tol:
            self._pin_insert_counter += 1
        else:
            self._pin_insert_counter = 0
        return self._pin_insert_counter >= CONFIG.pin.insertion.convergence_hysteresis

    # ------------------------------------------------------------------
    # Gates — small predicates on the latest losses.
    # ------------------------------------------------------------------

    def _lift_reached(self) -> bool:
        dl = abs(float(self._states.left_pose[2]-CONFIG.move_up.z))
        dr = abs(float(self._states.right_pose[2]-CONFIG.move_up.z))
        return dl < CONFIG.move_up.tol and dr < CONFIG.move_up.tol

    def _perturbation_reached(self) -> bool:
        if (self._perturbation_target_left is None
                or self._perturbation_target_right is None):
            return True
        with torch.no_grad():
            dl = float((self._states.left_pose[:3].detach()  - self._perturbation_target_left ).norm())
            dr = float((self._states.right_pose[:3].detach() - self._perturbation_target_right).norm())
        tol = CONFIG.recovery_perturbation.tol
        return dl < tol and dr < tol

    def _home_reached(self) -> bool:
        dl = float(((self._states.left_pose[:3]  - self.left_start[:3])  ** 2).sum().sqrt())
        dr = float(((self._states.right_pose[:3] - self.right_start[:3]) ** 2).sum().sqrt())
        return dl < CONFIG.move_up.tol and dr < CONFIG.move_up.tol

    def _pregrasp_reached_5dof(self) -> bool:
        """Per-axis (x, y, z) pregrasp gate; rotation ignored."""
        if self._left_index is None or self._right_index is None:
            return False
        pl = self._states.pregrasp[self._left_index, :3]
        pr = self._states.pregrasp[self._right_index, :3]
        l = float((self._states.left_pose[:3]  - pl).abs().max())
        r = float((self._states.right_pose[:3] - pr).abs().max())
        return l < CONFIG.grasp.pregrasp_tol and r < CONFIG.grasp.pregrasp_tol

    def _grasp_contact(self) -> bool:
        """Both arms within CONFIG.grasp.grasp_pos_tol of their fixed-z descent target on
        every axis (x, y, z) independently. Uses per-axis max errors cached by
        ``_grad_descending`` against the fixed-z target rather than the raw
        perceived beam z (so the gate fires correctly when grasp_z ≠ beam z).
        """
        if self._left_index is None or self._right_index is None:
            return False
        return (self._descent_max_axis_err_l < CONFIG.grasp.grasp_pos_tol
                and self._descent_max_axis_err_r < CONFIG.grasp.grasp_pos_tol)

    def _ee_slipped_from_beams(self) -> bool:
        """Distance between each EE and its assigned beam position; if either
        exceeds ``CONFIG.beam_assemble.slip_dist`` the grasp is suspect."""
        if self._left_index is None or self._right_index is None:
            return False
        beam_l = self._states.beam_poses[self._left_index, :3]
        beam_r = self._states.beam_poses[self._right_index, :3]
        dl = float(((self._states.left_pose[:3]  - beam_l) ** 2).sum().sqrt())
        dr = float(((self._states.right_pose[:3] - beam_r) ** 2).sum().sqrt())
        return dl > CONFIG.beam_assemble.slip_dist or dr > CONFIG.beam_assemble.slip_dist

    def _move_up_reached(self) -> bool:
        """Position-only pregrasp gate (no orientation requirement)."""
        if self._left_index is None or self._right_index is None:
            return True  # nothing to retreat from
        pl = self._states.pregrasp[self._left_index, :3]
        pr = self._states.pregrasp[self._right_index, :3]
        dl = float(((self._states.left_pose[:3]  - pl) ** 2).sum().sqrt())
        dr = float(((self._states.right_pose[:3] - pr) ** 2).sum().sqrt())
        return dl < CONFIG.move_up.tol and dr < CONFIG.move_up.tol

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
