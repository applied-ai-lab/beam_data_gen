"""Hierarchical configuration for the perceptive assembly planner.

All tunables live here, grouped by FSM phase / responsibility, so a
reader can find the constants relevant to a single state without
scanning the whole planner.  Access pattern:

    from beam_data_gen.traj_opt.perceptive_config import CONFIG
    if z_err < CONFIG.pin.insertion.z_tol:
        ...

Units are SI (metres, radians) unless noted.  The defaults are the
values previously hardcoded in ``perceptive_assembly.py`` — change
them here to retune.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Beam phase
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GraspConfig:
    """Gates that fire during MOVE_TO_PREGRASP / DESCENDING."""
    # Per-axis max |Δx|, |Δy|, |Δz| for the pregrasp gate to fire.
    pregrasp_tol: float = 0.02
    # Per-axis max |Δx|, |Δy|, |Δz| for the grasp-contact gate to fire.
    grasp_pos_tol: float = 0.013


@dataclass(frozen=True)
class BeamAssembleConfig:
    """DUAL_ASSEMBLE convergence + slip + timeout."""
    # Per-pair Euclidean hole-distance threshold for pair convergence.
    hole_convergence_threshold: float = 0.005
    # Cycles a pair must stay below the threshold before being latched.
    convergence_hysteresis: int = 1
    # Slip detector: EE→beam distance > slip_dist for slip_steps cycles
    # → grasp considered failed → RECOVERY_RELEASE.
    slip_dist:  float = 0.15
    slip_steps: int   = 1000
    # Step budget for DUAL_ASSEMBLE before timeout → RECOVERY_RELEASE.
    timeout_steps: int = 75


@dataclass(frozen=True)
class DescendingConfig:
    """Beam-phase DESCENDING state."""
    timeout_steps: int = 35


@dataclass(frozen=True)
class GoHomeConfig:
    """GO_HOME state."""
    timeout_steps: int = 100


# ---------------------------------------------------------------------------
# Common (shared across phases)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MoveUpConfig:
    """Position-only gate for MOVE_AWAY / RECOVERY_MOVE_UP / LIFT_PIN /
    RETREAT_PIN / PIN_RECOVERY_MOVE_UP — orientation ignored."""
    tol: float = 0.04
    z:   float = 1.0


@dataclass(frozen=True)
class RecoveryPerturbationConfig:
    """RECOVERY_PERTURBATION — pull both arms apart along the y-axis
    before releasing the grippers, so the next grasp attempt does not
    re-engage the same near-converged-but-stuck configuration."""
    # Per-arm y-axis displacement (m) applied at state entry. Arms move
    # in opposite directions along y, away from each other.
    y_delta: float = 0.10
    x_delta: float = 0.03
    # Per-arm Euclidean tolerance (m) on reaching the captured target.
    tol: float = 0.02
    # Step budget before the FSM bails to RECOVERY_RELEASE anyway.
    timeout_steps: int = 20


@dataclass(frozen=True)
class GradientConfig:
    """Gradient mixing + integrator step.  Kept identical to
    ``dual_assembly.py`` for behavioural parity in the assembly phase."""
    hole_weight:   float = 0.0
    yaw_weight:    float = 1.2
    # Hard-coded here (instead of read from TrajOptParams.step_size) so
    # the planner's integrator step is fixed by the config rather than
    # by callers.
    learning_rate: float = 0.25


@dataclass(frozen=True)
class SnapConfig:
    """Per-state snap radii.  Inside the radius the gradient is replaced
    by one whose integrator step lands exactly on the target, bypassing
    the asymptotic shrinkage of a quadratic loss.  Set any radius to 0.0
    to disable that snap."""
    # Beam phase.
    descent_radius:     float = 0.06   # DESCENDING
    assemble_radius:    float = 0.03   # DUAL_ASSEMBLE
    # Pin phase — independent of beam phase so they can be tuned in
    # isolation.  descent_radius above is reused for DESCEND_TO_PIN.
    pin_pregrasp_radius:  float = 0.05  # MOVE_TO_HOLE_PREGRASP
    pin_insertion_radius: float = 0.00  # INSERT_PIN


# ---------------------------------------------------------------------------
# Pin phase (see PERCEPTIVE_ASSEMBLY.MD §10)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PinPickupConfig:
    """MOVE_TO_PIN_PREGRASP / DESCEND_TO_PIN."""
    # Hover height above pin during MOVE_TO_PIN_PREGRASP.
    pregrasp_offset_z: float = 0.10
    # Per-axis max |Δx|, |Δy|, |Δz| for the MOVE_TO_PIN_PREGRASP gate to
    # fire. Decoupled from the beam-phase pregrasp tolerance so pin
    # pickup can be tightened independently.
    pregrasp_tol:      float = 0.015
    # Z offset above pin centre at the contact gate.
    grasp_offset_z:    float = 0.00
    # Pregrasp / grasp yaw is forced to 0 during pin pickup —
    # (sin θ, cos θ) = (0, 1).
    grasp_yaw_sin: float = 0.0
    grasp_yaw_cos: float = 1.0


@dataclass(frozen=True)
class PinRotateConfig:
    """ROTATE_PIN_INWARD."""
    # Yaw error tolerance (rad) for the rotate gate.
    yaw_tol: float = 0.2
    # "Inward" yaw target for the left arm: gripper toward y = 0.
    # Left arm sits on +y so inward yaw is around -π/4 (cf. -π/2 would
    # be a full quarter turn — these defaults are -π/4, sin/cos of
    # ±0.707107).
    inward_yaw_left_sin: float = -1.0
    inward_yaw_left_cos: float =  0.0


@dataclass(frozen=True)
class PinInsertionConfig:
    """MOVE_TO_HOLE_PREGRASP / INSERT_PIN.

    Linear pipeline: success or timeout, both go to RELEASE_PIN.
    There is no slip / recovery branch in pin insertion."""
    # Pre-insertion hover gate (max |Δx|, |Δy|, |Δz|).
    pregrasp_tol: float = 0.002
    # Hover height for MOVE_TO_HOLE_PREGRASP, expressed as a delta
    # ABOVE the active hole-pair midpoint z (not absolute).  The pin
    # then descends from (mid_xy, mid_z + pregrasp_z_delta) to mid_z
    # during INSERT_PIN.
    pregrasp_z_delta: float = 0.09
    # Insertion success criterion — z is primary (confirms the pin
    # has descended into the hole); xy is a sanity check.
    z_tol:  float = 0.01
    xy_tol: float = 0.01
    # Cycles the EE must stay inside the success region before
    # RELEASE_PIN fires.
    convergence_hysteresis: int = 1
    # Step budget before INSERT_PIN gives up and releases anyway —
    # prevents a misaligned insertion deadlocking the FSM.
    timeout_steps: int = 10
    # Step budget for MOVE_TO_HOLE_PREGRASP before the FSM bails out
    # and lifts to RECOVER_INSERTION_PREGRASP for a retry.
    pregrasp_timeout_steps: int = 50
    # Vertical lift (m) applied during RECOVER_INSERTION_PREGRASP —
    # the EE moves up this far above wherever it was when the
    # MOVE_TO_HOLE_PREGRASP timeout fired, then retries.
    recovery_z_delta: float = 0.12


@dataclass(frozen=True)
class PinOffsetConfig:
    """COMPUTE_PIN_OFFSET — calibrate the 6-DOF pin↔hand transform
    before insertion.

    After ROTATE_PIN_INWARD, the hand is held at
    ``(rotate_target_x, rotate_target_y, rotate_target_z)`` (a pose
    chosen so the pin AprilTag remains inside the FORWARD camera FOV).
    The wrapper-side calibrator (``FrankAtls.maybe_run_pin_calibration``)
    samples both the pin world pose and the hand world pose at TF rate
    (~30 Hz) for ``calibration_duration`` seconds, averages translation
    by mean and rotation by quaternion eigenvector mean, and pushes the
    resulting 6-DOF transform into the planner via
    ``set_estimated_pin_position``.  The planner subtracts translation
    + rotates the hand yaw target by the inverse of the calibrated yaw
    delta during MOVE_TO_HOLE_PREGRASP / INSERT_PIN /
    RECOVER_INSERTION_PREGRASP so the *pin* (not the hand) lands on the
    hole-pair midpoint, compensating for the imperfect grasp pose."""
    # Hand pose held during ROTATE_PIN_INWARD and COMPUTE_PIN_OFFSET.
    rotate_target_x: float = 0.28
    rotate_target_y: float = 0.0
    rotate_target_z: float = 1.0
    # Wall-clock duration (s) over which the wrapper-side calibrator
    # accumulates pin/hand TF samples before pushing the averaged
    # 6-DOF transform.  Longer = more averaging, but COMPUTE_PIN_OFFSET
    # blocks the planner cycle for this long once per pin pickup.
    calibration_duration: float = 5.0
    # Maximum allowed Euclidean distance between perceived pin and hand
    # for a sample to be accepted into the calibration average. Rejects
    # stale / mis-associated tag detections that would otherwise poison
    # the mean.
    max_pin_hand_dist: float = 0.05


@dataclass(frozen=True)
class PinHomeConfig:
    """Pin-phase home pose (both arms park here at STOW_BOTH).

    Both arms park at the beam-phase home shifted ``y_offset`` further
    from the workspace centreline (y = 0): left's y is increased by
    this amount, right's y is decreased.  Gives the camera an
    unobstructed view and keeps the right arm clear of the left arm's
    working volume during pin insertion."""
    y_offset: float = 0.20
    tolerance: float = 0.06


@dataclass(frozen=True)
class PinPhaseConfig:
    pickup:    PinPickupConfig    = field(default_factory=PinPickupConfig)
    rotate:    PinRotateConfig    = field(default_factory=PinRotateConfig)
    insertion: PinInsertionConfig = field(default_factory=PinInsertionConfig)
    offset:    PinOffsetConfig    = field(default_factory=PinOffsetConfig)
    home:      PinHomeConfig      = field(default_factory=PinHomeConfig)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PerceptiveConfig:
    grasp:          GraspConfig         = field(default_factory=GraspConfig)
    beam_assemble:  BeamAssembleConfig  = field(default_factory=BeamAssembleConfig)
    descending:     DescendingConfig    = field(default_factory=DescendingConfig)
    go_home:        GoHomeConfig        = field(default_factory=GoHomeConfig)
    move_up:        MoveUpConfig        = field(default_factory=MoveUpConfig)
    recovery_perturbation: RecoveryPerturbationConfig = field(default_factory=RecoveryPerturbationConfig)
    gradient:       GradientConfig      = field(default_factory=GradientConfig)
    snap:           SnapConfig          = field(default_factory=SnapConfig)
    pin:            PinPhaseConfig      = field(default_factory=PinPhaseConfig)


# Module-level singleton.  Imported by ``perceptive_assembly``.
CONFIG: PerceptiveConfig = PerceptiveConfig()
