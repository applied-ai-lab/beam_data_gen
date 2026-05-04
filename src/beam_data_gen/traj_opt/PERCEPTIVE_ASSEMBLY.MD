# PERCEPTIVE ASSEMBLY — Hybrid State-Machine Planner

This document describes `beam_data_gen.traj_opt.perceptive_assembly.DualAssembly`,
a drop-in replacement for `dual_assembly.DualAssembly`. It mirrors the same
constructor signature and public API (`optimise()`, `set_hole_positions()`,
`states` setter, `_left_index`, `_right_index`, `_convergence`,
`_convergence_counter`, `_active_pair_idx`, `_hand_losses`, `arm_z_floor`) but
replaces the implicit gradient/contact-gate blend with an **explicit finite
state machine** wrapping per-state gradient-based local controllers.

This file is the spec the code in `perceptive_assembly.py` actually
implements. The earlier "originally proposed" sketch (sticky 2×2 assignment,
MSE-sum gates, gripper hysteresis, etc.) is not what was built; deliberate
deviations are called out inline.

---

## 1. Overview

### 1.1 State machine

```
   GO_HOME ─▶ PICK_TASK ──no work──▶ DONE  ──perturbed──┐
                  │                                     │
                  ▼ active pair fixed                   │
              MOVE_TO_PREGRASP ─▶ DESCENDING ─▶ CLOSE_GRIPPER ─▶ READY
                                      │                                │
                                      │ timeout                        ▼
                                      ▼                          DUAL_ASSEMBLE
                                RECOVERY_RELEASE                  │      │
                                      │                converged  │      │ slip / timeout
                                      ▼                           ▼      ▼
                                RECOVERY_MOVE_UP           RELEASE_GRIPPER
                                      │                           │
                                      ▼                           ▼
                              MOVE_TO_PREGRASP                MOVE_AWAY
                                                                  │
                                                                  ▼
                                                              PICK_TASK
```

Twelve states (`State` IntEnum):

| ID | Name | Purpose |
|---|---|---|
| 0 | `GO_HOME` | Drive both arms to `left_start` / `right_start`. |
| 1 | `PICK_TASK` | Latch hole-converged beams, choose next pair, fix arm assignment. |
| 2 | `MOVE_TO_PREGRASP` | Hover both arms above their assigned beams. |
| 3 | `DESCENDING` | Drive both arms onto the perceived beam pose. |
| 4 | `CLOSE_GRIPPER` | One-step latch — both grippers closed. |
| 5 | `READY` | One-step pass-through to `DUAL_ASSEMBLE`. |
| 6 | `DUAL_ASSEMBLE` | Beam→goal + hole-collinearity gradient until pair converges. |
| 7 | `RECOVERY_RELEASE` | One-step latch — both grippers opened on grasp/slip failure. |
| 8 | `RECOVERY_MOVE_UP` | Lift to `MOVE_UP_Z` then re-enter `MOVE_TO_PREGRASP`. |
| 9 | `RELEASE_GRIPPER` | One-step latch — both grippers opened on success. |
| 10 | `MOVE_AWAY` | Lift to `MOVE_UP_Z` then re-enter `PICK_TASK`. |
| 11 | `DONE` | Idle wait; re-enters `PICK_TASK` if a previously converged pair becomes unconverged. |

There are **no per-arm sub-FSMs**. Both arms share a single state — the
arm-to-beam mapping is fixed in `PICK_TASK` so per-arm sub-FSMs would be
redundant complexity. Both arms gate on conjunction: e.g. `MOVE_TO_PREGRASP`
exits only when *both* arms satisfy the pregrasp gate.

### 1.2 Tunables (module-level constants)

All units SI (m, rad). These are module-level constants in
`perceptive_assembly.py`; per-instance knobs are listed in §6.

| Symbol | Default | Meaning |
|---|---|---|
| `PREGRASP_TOL` | `0.02 m` | Per-axis max abs error on (x, y, z) for pregrasp gate. |
| `GRASP_POS_TOL` | `0.01 m` | Per-axis max abs error on (x, y, z) for grasp contact gate. |
| `HOLE_CONVERGENCE_THRESHOLD` | `0.003 m` | Per-pair Euclidean hole distance for assembly convergence. |
| `CONVERGENCE_HYSTERESIS` | `5` cycles | PICK_TASK visits a pair must remain below the hole threshold before being latched. |
| `ASSEMBLE_SLIP_DIST` | `0.03 m` | EE↔beam Euclidean distance threshold for the slip detector. |
| `ASSEMBLE_SLIP_STEPS` | `15` steps | Consecutive slip-detected steps before triggering recovery. |
| `DESCEND_TIMEOUT_STEPS` | `35` | Max steps in `DESCENDING` before recovery. |
| `ASSEMBLE_TIMEOUT_STEPS` | `100` | Max steps in `DUAL_ASSEMBLE` before recovery. |
| `GO_HOME_TIMEOUT_STEPS` | `100` | Max steps in `GO_HOME` before forcing transition. |
| `MOVE_UP_TOL` | `0.05 m` | Position-only Euclidean gate for lift / home reached. |
| `MOVE_UP_Z` | `1.0 m` | Target z for `RECOVERY_MOVE_UP` / `MOVE_AWAY`. |
| `HOLE_GRADIENT_WEIGHT` | `0.2` | Hole-collinearity term weight in `DUAL_ASSEMBLE`. |
| `YAW_GRADIENT_WEIGHT` | `0.8` | Multiplicative boost on the (sinθ, cosθ) channels of the beam gradient. |
| `LEARNING_RATE` | `0.3` | Gradient-descent step size. Hard-coded here, *not* read from `TrajOptParams.step_size`. |

### 1.3 Deliberate deviations from the original sketch

These are the eight items called out in the source docstring:

1. **Fixed arm-to-beam assignment.** Right arm always handles odd-indexed
   beams (1, 3); left arm always handles even-indexed beams (0, 2). The
   mapping is computed once on entry to `PICK_TASK` and never reshuffled
   mid-attempt. This replaces the original sticky cost-based 2×2 assignment.
2. **Pregrasp gate uses per-axis (x, y, z) max abs error**, not a 5-DOF
   MSE-sum. Each axis must be below `PREGRASP_TOL` independently. Rotation
   is ignored.
3. **Convergence is touched only in `PICK_TASK` and `DUAL_ASSEMBLE`.** No
   per-cycle latching during grasp / move states.
4. **No gripper hysteresis.** Grippers are toggled exclusively in
   `CLOSE_GRIPPER` (close) and `RELEASE_GRIPPER` / `RECOVERY_RELEASE` (open).
   Transient contact loss in any other state does not actuate them.
5. **Single recovery sequence.** Both `DESCENDING` timeout and the
   `DUAL_ASSEMBLE` slip detector route through
   `RECOVERY_RELEASE → RECOVERY_MOVE_UP → MOVE_TO_PREGRASP` (same active pair).
6. **`DUAL_ASSEMBLE` slip detector.** If either EE separates from its
   assigned beam pose by > `ASSEMBLE_SLIP_DIST` for `ASSEMBLE_SLIP_STEPS`
   consecutive steps, the planner assumes the grasp slipped and routes
   through recovery (without releasing convergence on the active pair).
7. **`hole_pairs` is mandatory.** `__init__` raises `ValueError` if it is
   empty / `None`. `optimise()` raises `RuntimeError` if
   `set_hole_positions()` was never called. There is no fallback to
   `start_loss[a] + start_loss[b]`-style scoring as in the original.
8. **`GO_HOME` is a real state**, not a scoring bias. It uses `left_start` /
   `right_start` directly as quadratic targets.

---

## 2. State semantics

### 2.1 GO_HOME

Initial state. Drives both arms to the supplied `left_start` / `right_start`
poses simultaneously via independent quadratic losses. Transitions to
`PICK_TASK` when **both** arms are within `MOVE_UP_TOL` (Euclidean, position
only) of their start, or after `GO_HOME_TIMEOUT_STEPS`, whichever comes
first. On transition, `_enter_pick_task()` runs immediately.

### 2.2 PICK_TASK

The only place (besides `DUAL_ASSEMBLE` for the active pair) where
`_convergence` / `_convergence_counter` are touched.

1. For every beam `k`, increment `_convergence_counter[k]` if the beam's
   pair currently satisfies `‖hole_a − hole_b‖ < HOLE_CONVERGENCE_THRESHOLD`,
   else reset it. Once the counter reaches `CONVERGENCE_HYSTERESIS`, latch
   `_convergence[k] = True`.
2. Pick the unconverged pair with the smallest current `‖hole_a − hole_b‖`.
   Pairs with both beams already in `_convergence` are skipped.
3. If no unconverged pair exists, clear `_active_pair_idx` /
   `_left_index` / `_right_index` and transition to **DONE**.
4. Otherwise set `_active_pair_idx`, fix the arm mapping by parity (odd
   beam → right, even beam → left), and transition to `MOVE_TO_PREGRASP`.

### 2.3 MOVE_TO_PREGRASP

`_grad_pregrasp` controller. Each arm pursues its assigned beam's pregrasp
pose (the "directly above the perceived beam" pose, refreshed every cycle
by `DualArmStates.advance()`). Independently, the pregrasp tensors
themselves are pulled by `_pregrasp_losses_bank.pregrasp_loss` so they track
the current beam observation.

Gate (`_pregrasp_reached_5dof`, despite the name): position-only per-axis
max abs error; both arms must satisfy `max(|Δx|, |Δy|, |Δz|) < PREGRASP_TOL`.
On success → `DESCENDING`.

### 2.4 DESCENDING

`_grad_descending` controller. Each arm pursues the *full* perceived beam
pose (x, y, z, sinθ, cosθ) of its assigned beam. The earlier fixed-z
override (`self.grasp_z`) is intentionally bypassed — the arm tracks the
AprilTag z directly so the contact gate fires at real beam height, not at
a hardcoded value sitting above it.

**Snap branch.** Inside `descent_snap_radius` (default 4 cm) of the target,
the quadratic gradient is replaced with `g = (x − x*) / LEARNING_RATE` so
the integrator step lands exactly on the target. This bypasses the
asymptotic shrinkage of a quadratic loss without overshoot. `max_ee_step`
and `_clamp_ee_poses` still apply, so motion remains smooth and bounded.
Set `descent_snap_radius = 0.0` to disable. Whether the snap fired and a
cumulative count are exposed per arm for diagnostics.

Per-axis (x, y, z) errors are cached in `_descent_max_axis_err_{l,r}` for
the contact gate; Euclidean distance to target is cached in
`_descent_loss_{l,r}` for ROS diagnostics.

Exit gates:
- `_grasp_contact()`: both arms have `max(|Δx|, |Δy|, |Δz|) < GRASP_POS_TOL`.
  → `CLOSE_GRIPPER`. The two idle-transition resolutions following are
  invoked immediately so the next rollout step starts in `DUAL_ASSEMBLE`.
- `_state_step >= DESCEND_TIMEOUT_STEPS` → `RECOVERY_RELEASE`.

### 2.5 CLOSE_GRIPPER (1-step idle)

`_evaluate_idle_transitions` sets `_gripper_closed = [True, True]` and
transitions to `READY`. No gradient.

### 2.6 READY (1-step idle)

Pass-through; immediately transitions to `DUAL_ASSEMBLE`. No gradient.
Kept as an explicit state for consistency with the original sketch and to
provide a clean diagnostic landmark in logs.

### 2.7 DUAL_ASSEMBLE

`_grad_dual_assemble` controller. Mirrors the contact-blended assembly
phase of the original `dual_assembly.py` (~lines 679–764):

```
beam_grad  = ∂(beam_pos_loss)/∂beam_poses
           + HOLE_GRADIENT_WEIGHT · ∂(hole_collinearity)/∂beam_poses
beam_grad[:, 3:5] *= YAW_GRADIENT_WEIGHT
beam_grad[k] *= 0   for k in _convergence
```

The hole-collinearity term is `_hole_collinearity_loss` lifted verbatim
from `dual_assembly.py` (free function in this module so it can be
unit-tested without instantiating the planner). It pulls each pair's
predicted hole positions toward each other in the world frame, with
gradients flowing through both position and yaw.

**Snap branch.** Inside `assemble_snap_radius` (default 4 cm xy distance to
beam goal), the active pair's beam gradients are replaced with
`(beam_pose − beam_goal) / LEARNING_RATE`, snapping the beam onto the
goal in one step. Already-converged beams are skipped; their gradient is
zeroed by the masking step above and not disturbed by the snap.

**EE coupling.** `_gradients.left_pose` / `_gradients.right_pose` are taken
as the beam gradient of the assigned beam, with the z-channel zeroed —
arms remain at grasp z throughout assembly. The original's
"contact-mode blend" is not needed because the FSM has guaranteed both
arms are grasped before reaching this state.

Exit gates (in evaluation order):
1. **Success** — `‖hole_a − hole_b‖ < HOLE_CONVERGENCE_THRESHOLD` for the
   active pair. Latch both beams of the pair into `_convergence`.
   → `RELEASE_GRIPPER`, idle-transition immediately resolves to
   `MOVE_AWAY`.
2. **Slip** — `_ee_slipped_from_beams()` (`‖ee − beam‖ > ASSEMBLE_SLIP_DIST`
   for either arm) increments `_slip_counter`; otherwise it resets. When
   the counter reaches `ASSEMBLE_SLIP_STEPS`, reset and route through
   `RECOVERY_RELEASE`. **`_convergence` is not cleared** — recovery retries
   the same pair from a fresh grasp.
3. **Timeout** — `_state_step >= ASSEMBLE_TIMEOUT_STEPS` →
   `RECOVERY_RELEASE`.

### 2.8 RECOVERY_RELEASE (1-step idle)

Sets `_gripper_closed = [False, False]` and transitions to
`RECOVERY_MOVE_UP`. No gradient.

### 2.9 RECOVERY_MOVE_UP

`_grad_pregrasp` controller (the per-state dispatch routes
`RECOVERY_MOVE_UP` to `_grad_pregrasp`, identical to `MOVE_TO_PREGRASP` /
`MOVE_AWAY`). On `_lift_reached()` (both arms within `MOVE_UP_TOL` of
`MOVE_UP_Z`) → `MOVE_TO_PREGRASP`, retrying the *same* `_active_pair_idx`.

> **Note.** `_grad_lift` exists in source but is unreachable: the
> `_step_gradients` dispatch matches `RECOVERY_MOVE_UP` / `MOVE_AWAY` in
> the prior `_grad_pregrasp` branch, so the `_grad_lift` branch never
> fires. The active controller is `_grad_pregrasp` for both move-up
> states; the only thing the lift-reached gate cares about is the EE z
> coordinate vs `MOVE_UP_Z`, which the pregrasp pose typically already
> satisfies after a fresh `advance()`.

### 2.10 RELEASE_GRIPPER (1-step idle)

Sets `_gripper_closed = [False, False]` and transitions to `MOVE_AWAY`.
No gradient.

### 2.11 MOVE_AWAY

Same controller as `MOVE_TO_PREGRASP`. On `_lift_reached()` →
`PICK_TASK`, with `_enter_pick_task()` invoked immediately.

### 2.12 DONE

Idle wait. `_evaluate_idle_transitions` re-runs `_select_pair()` every
cycle; if a previously converged pair becomes unconverged (e.g. an
external perturbation cleared an entry from `_convergence`, or a beam
moved out of tolerance), the FSM seamlessly re-enters `PICK_TASK` without
caller intervention. Zero gradient is emitted while waiting.

### 2.13 Idle-transition fast-pathing

`_evaluate_idle_transitions` resolves states whose semantics are "set a
flag, advance" without consuming a gradient step:
`PICK_TASK`, `CLOSE_GRIPPER`, `READY`, `RECOVERY_RELEASE`,
`RELEASE_GRIPPER`, and the `DONE → PICK_TASK` re-entry. It is invoked:

- Once before the rollout (`optimise()`),
- Inside the rollout immediately after a transition into one of those
  states (chained calls in `_evaluate_step_transitions` resolve up to two
  back-to-back idle states in one go, so e.g. `DESCENDING → CLOSE_GRIPPER
  → READY → DUAL_ASSEMBLE` is resolved in a single rollout step).

---

## 3. Loss / gradient producers

| State(s) | Producer | Notes |
|---|---|---|
| `GO_HOME` | `_grad_go_home` | Quadratic to `left_start` / `right_start`. |
| `MOVE_TO_PREGRASP`, `RECOVERY_MOVE_UP`, `MOVE_AWAY` | `_grad_pregrasp` | Quadratic to assigned beam's pregrasp pose; pregrasp tensors tracked separately by `_pregrasp_losses_bank`. |
| `DESCENDING` | `_grad_descending` | Quadratic to perceived beam pose; per-axis error cache; snap branch inside `descent_snap_radius`. |
| `DUAL_ASSEMBLE` | `_grad_dual_assemble` | Beam-goal MSE + `HOLE_GRADIENT_WEIGHT` × hole collinearity; `YAW_GRADIENT_WEIGHT` boost on (sinθ, cosθ); converged beams masked; snap branch inside `assemble_snap_radius`; arm gradients = beam grad of assigned beam with z zeroed. |
| `PICK_TASK`, `CLOSE_GRIPPER`, `READY`, `RELEASE_GRIPPER`, `RECOVERY_RELEASE`, `DONE` | (none) | Zero gradient — these are idle / latch states. |

`_grad_lift` is defined but unreachable (see §2.9 note).

---

## 4. Integration step

Per rollout step (`optimise()` runs `params.no_steps - 1` of these):

```
gradients = self._step_gradients()
delta_l = self._limit_ee_delta(LEARNING_RATE * gradients.left_pose)
delta_r = self._limit_ee_delta(LEARNING_RATE * gradients.right_pose)
left_pose  -= delta_l
right_pose -= delta_r
beam_poses -= LEARNING_RATE * gradients.beam_poses
pregrasp   -= LEARNING_RATE * gradients.pregrasp
self._clamp_ee_poses()        # per-arm z bounds
self._states.detach()
... pack into _x, _normalise_pose (z floor + yaw unit-circle), repack ...
self._evaluate_step_transitions()
```

`LEARNING_RATE` is a module constant, *not* `TrajOptParams.step_size` —
the planner's integrator step is fixed by the module rather than by
callers.

---

## 5. End-effector limits — what is actually enforced

Hard constraints are clamped after each gradient step (not added as soft
penalties). Compared to the originally proposed envelope, the implemented
set is much smaller — most of the workspace-shaping constraints described
in the original §4 (x-extent, y-partition, gripper yaw range, max yaw
step) are **not implemented**.

### 5.1 Per-arm z bounds (implemented)

Applied in `_clamp_ee_poses` every step:

- **Left EE z** ∈ `[left_z_floor, left_z_ceil]` (defaults
  `arm_z_floor = 0.78`, `arm_z_ceil = 1.150`).
- **Right EE z** ∈ `[right_z_floor, right_z_ceil]` (same defaults).
- **Pregrasp z** ∈ `[arm_z_floor, arm_z_ceil]` (shared, applied in the
  same call).

`arm_z_floor = 0.78` matches the IK z-limits enforced by
`frank_atls.py`'s `_z_lims = [0.755, 1.00]`. Mismatches mean the planner
routinely requests poses that hardware silently clips, causing planner
state to drift from reality each cycle.

`_normalise_pose` additionally enforces a z floor of
`max(0.75, arm_z_floor)` on the arm pose blocks and a hard 0.75 floor on
beam / pregrasp blocks; it also re-projects the (sinθ, cosθ) channels
back onto the unit circle.

### 5.2 Per-step EE displacement cap (implemented)

`_limit_ee_delta` clips the *positional* part of an EE step to
`max_ee_step` (default `inf` — disabled). Yaw channels are deliberately
left untouched so orientation updates are not silently suppressed.

### 5.3 Active-only gradient masking (implemented)

`_grad_pregrasp`, `_grad_descending`, `_grad_dual_assemble` early-return
without writing gradients when `_left_index is None or _right_index is
None`. Combined with the gradient zero-init in `_step_gradients`, idle
arms receive zero gradient.

### 5.4 NOT implemented

The following sections of the originally proposed limit set are **not**
present in the code; do not assume they hold:

- **X-extent** (`x_min` / `x_max`) — no x clamp anywhere.
- **Y-partition** (centreline half-space per arm) — no y clamp anywhere.
- **Z ceiling on beams / pregrasp** — only `arm_z_ceil` on the EE blocks.
- **Per-step yaw step cap** (`max_yaw_step`) — yaw deltas are unclipped.
- **Gripper yaw range vs beam yaw** — no clamp.

If any of these become safety-relevant, they need to be added to
`_clamp_ee_poses` / `_normalise_pose` (and the table here updated).

---

## 6. Per-instance configuration knobs

Set on the `DualAssembly` instance (defaults in parentheses):

| Attribute | Default | Purpose |
|---|---|---|
| `arm_z_floor` | `0.78` | Shared z floor for arms + pregrasp clamps. |
| `arm_z_ceil` | `1.150` | Shared z ceiling for the EE clamps. |
| `left_z_floor` / `left_z_ceil` | inherits | Per-arm override for the left EE. |
| `right_z_floor` / `right_z_ceil` | inherits | Per-arm override for the right EE. |
| `max_ee_step` | `inf` | Per-step EE positional displacement cap (m). |
| `descent_snap_radius` | `0.04` | Snap-to-target radius in `DESCENDING`. `0.0` disables. |
| `assemble_snap_radius` | `0.04` | Snap-to-goal xy radius in `DUAL_ASSEMBLE`. `0.0` disables. |
| `grasp_z` | `0.793` | **Unused** — the descent target now follows the perceived beam z directly. Retained for back-compat. |
| `step_mode` | `False` | If `True`, `optimise()` blocks on `input()` after every state transition for interactive debugging. |

Diagnostic / read-only fields populated by the planner each cycle:

- `_descent_loss_l`, `_descent_loss_r` — Euclidean EE→target distance in
  `DESCENDING` (used by `frank_atls.publish_planner_diagnostics`).
- `_descent_max_axis_err_l`, `_descent_max_axis_err_r` — per-axis max abs
  error against the descent target (drives `_grasp_contact`).
- `_descent_snap_fired_l`, `_descent_snap_fired_r`,
  `_descent_snap_count_l`, `_descent_snap_count_r` — snap branch
  telemetry.
- `_assemble_snap_fired[k]`, `_assemble_snap_count` — assemble snap
  telemetry.
- `_active_pair_idx`, `_left_index`, `_right_index`, `_convergence`,
  `_convergence_counter`, `_gripper_closed`, `_slip_counter`, `_state`,
  `_state_step` — FSM state, exposed for ROS diagnostics
  (`/atls/fsm_state`, `/atls/active_pair`, `/atls/slip_counter`, etc.).

---

## 7. Diagnostics (`log_state`)

`DualAssembly.log_state(logger=print)` produces a one-screen status block
per call. Used by the ROS layer (`frank_atls.publish_planner_diagnostics`,
`step_mode`) to dump:

- Current state name, per-state step counter, and remaining budget for
  states with timeouts (`DESCENDING`, `DUAL_ASSEMBLE`, `GO_HOME`).
- Active pair, per-arm beam assignment, gripper booleans.
- `_convergence` keys and `_convergence_counter` per beam.
- Per-pair hole distance with a `✓` flag when below
  `HOLE_CONVERGENCE_THRESHOLD`.
- Per-beam: position, goal, xy error (mm), yaw error (deg),
  L/R position loss, convergence probability.
- L / R EE positions and pregrasp targets.
- Pregrasp / grasp per-axis max errors with the configured gate
  thresholds.

---

## 8. Public API contract

Constructor:

```python
DualAssembly(
    params: TrajOptParams,
    state_params: StateParams,
    sim: Optional[SquareRobotSim],
    left_start: torch.Tensor,
    right_start: torch.Tensor,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hole_pairs: List[Tuple[int, int]],   # mandatory, non-empty
)
```

Cycle protocol (used by `frank_atls.py`):

```python
planner.set_hole_positions(hole_positions_np)   # required every cycle
planner.states = current_dual_arm_states         # advance() runs in setter
trajectories = planner.optimise()                # ParticleTrajectories
```

Failure modes:

- `ValueError` from `__init__` if `hole_pairs` is empty / `None`.
- `RuntimeError` from `optimise()` if `set_hole_positions` was never
  called.

The `ParticleTrajectories` returned has shape
`[no_particles, no_steps, _x.size]`, with row 0 the deterministic rollout
broadcast across particles (no actual particle filter — the wrapping
preserves the consumer-facing `sample_indices` / `sample_trajectories`
API).

---

## 9. Re-used components

The data-bearing classes are imported verbatim from `dual_assembly`:

- `DualArmStates`, `DualArmGradients` — pose/state/gradient containers.
- `HandLossesContacts`, `BeamLosses`, `PregraspLosses` — loss banks
  (`calc_losses` / `calc_prob`).
- `ParticleTrajectories`, `StateParams`, `TrajOptParams`.

Only the **algorithmic dispatch** (FSM + per-state gradient producers) is
new; the wire format consumed by callers is bit-identical to
`dual_assembly`.
