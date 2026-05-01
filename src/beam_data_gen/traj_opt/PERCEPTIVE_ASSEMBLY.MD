# PERCEPTIVE ASSEMBLY — Hybrid State-Machine Planner

This document specifies the perceptive variant of `DualAssembly`. It functionally
mirrors `dual_assembly.py` (same constructor signature and public API) but
replaces the implicit gradient/contact-gate blend with an **explicit hybrid
state machine** wrapped around per-state gradient-based local planners.

The body below is split into:

1. **Logic verification** of the original sketch against `dual_assembly.py`.
2. **Corrected & expanded state-machine specification.**
3. **Suggested improvements** over the original implementation.
4. **End-effector limits** that govern the entire model (new clause, last
   section).

---

## 1. Verification of the originally proposed logic

The originally proposed sketch was checked line-by-line against
`dual_assembly.py`. Findings:

| Claim in original sketch | Verdict | Notes |
|---|---|---|
| `STATE=GO_HOME` on init drives to a starting posture and then transitions to `PICK_TASK` | **Partially correct.** The original planner has *no* explicit go-home: it uses `left_start` / `right_start` only as a scoring bias inside `check_convergence` (via `_start_loss`, weighted ×10 on pin indices, see `dual_assembly.py:769-776`). Add this as a real state in the perceptive version. | OK to add, but call it out as **new behaviour**. |
| `PICK_TASK` selects the closest unconverged pair via hole positions | **Correct.** Matches `_select_pair()` (`dual_assembly.py:489-510`): pair scored by `‖hole_a − hole_b‖`, fallback to `active_left_loss[a] + active_right_loss[b]` when hole positions are not yet set. | Keep. |
| Termination condition: "if both [pairs] are converged, set state to GO_HOME" | **Wrong wording.** Termination is per-pair convergence accumulated into `self._convergence` over `_CONVERGENCE_HYSTERESIS = 5` cycles; the planner finishes when *every* pair is in `self._convergence`, i.e. `_select_pair()` returns `None`. It does not loop back to GO_HOME and re-iterate. | Use `IDLE` / `DONE` as the terminal state. |
| `DUAL_GRAB`: "right arm always on beams (0,1), left arm always on (2,3)" | **WRONG.** The original uses a **sticky cost-based 2×2 assignment** of the *two beams of the active pair* to the two arms (`check_convergence`, `dual_assembly.py:815-829`): if `{prev_l, prev_r} == {a, b}` reuse the previous mapping, otherwise pick whichever of `(a,b)` ↔ (left, right) or `(b,a)` ↔ (left, right) has lower summed start-loss. There is no fixed beam-index → arm mapping. | Replace with **cost-based, sticky** assignment. |
| Pregrasp gate: "L2 within 1 cm and 5°" | **Wrong threshold.** Original gate is `_pregrasp_horiz_loss < 0.01` over `(x, y, sinθ, cosθ)` only — z is intentionally excluded so descent does not flip the gate (`dual_assembly.py:280-287, 322-323`). Numerically that is ~MSE-sum of ~0.01, i.e. roughly ~7 cm horizontal *or* a few deg yaw, whichever dominates. | Spec should match: horizontal-only gate, with z gating handled by a separate `DESCENDING` controller. |
| Contact / grasp gate: "L2 within 3 mm, 5°" | **Wrong threshold.** Original uses **position-only sum-of-squared-error** `< _CONTACT_POS_THRESHOLD = 5e-5` (`dual_assembly.py:19, 271, 325`). That is ~7 mm radius in 3D, with no orientation term. The 3 mm threshold mentioned in the sketch is `HOLE_CONVERGENCE_THRESHOLD`, used for assembly convergence, not grasp contact. | Separate the two thresholds explicitly. |
| `CLOSE_GRIPPER` on grasp contact | **Correct in spirit, missing hysteresis.** Original closes immediately on contact but only re-opens after `_gripper_open_hold = 5` consecutive lost-contact cycles (`dual_assembly.py:528-547`). | Carry hysteresis through. |
| `READY` waiting state until both arms grasped, then `DUAL_ASSEMBLE` | **Correct concept**, but absent in the original — it is implicit in the contact-mode blending of beam vs. pregrasp loss. Making it explicit is a strict improvement. | Keep, with a timeout (see §3). |
| `DUAL_ASSEMBLE`: "minimise beam→goal + collinearity, max 10 s / 300 steps" | **Correct in shape.** Original blends `_beam_losses` (beam→goal MSE-sum) with `_calc_hole_collinearity_loss()` weighted by `_HOLE_GRADIENT_WEIGHT = 0.3`, plus `_YAW_GRADIENT_WEIGHT = 0.8` boost on the yaw components. Convergence is hole-based, hysteresis 5 cycles. No explicit time budget in the original; adding one is fine, but it must drop into `RELEASE_GRIPPER` only on actual hole convergence — a timeout should fall back to `MOVE_TO_PREGRASP`, not progress the assembly. | Time-budget timeout exits to `RECOVERY_PREGRASP`, not `RELEASE_GRIPPER`. |
| `RELEASE_GRIPPER` then `MOVE_AWAY` | **Correct.** Original implicitly does this via `advance()` resetting `pregrasp` to "directly above the perceived beam" each cycle (`dual_assembly.py:108-118`) and zeroing horizontal motion when the descent gradient is large negative (`dual_assembly.py:743-745, 757-759`). | Make explicit. |

**Bottom line:** the broad shape of the originally proposed FSM is sound, but
several specific clauses must be corrected (arm-to-beam assignment, pregrasp
gate, grasp contact threshold, termination behaviour). The corrected
specification is in §2.

---

## 2. Corrected state-machine specification

Tunables (all configurable, defaults from `dual_assembly.py`):

| Symbol | Default | Source | Meaning |
|---|---|---|---|
| `pregrasp_horiz_tol` | `0.01` (MSE-sum over xy + sinθ + cosθ) | `_pregrasp_con` gate | Horizontal-only pregrasp gate. |
| `grasp_pos_tol` | `5e-5` (sum-sq, ~7 mm 3D) | `_CONTACT_POS_THRESHOLD` | Grasp contact gate (position only). |
| `hole_conver_tol` | `3 mm` | `HOLE_CONVERGENCE_THRESHOLD` | Per-pair assembly convergence. |
| `conver_hysteresis` | `5` cycles | `_CONVERGENCE_HYSTERESIS` | Cycles required before latching. |
| `gripper_open_hold` | `5` cycles | `_gripper_open_hold` | Lost-contact cycles before re-open. |
| `descent_offset_z` | `0.02 m` (+ per-arm trim) | `_GRASP_Z_DESCENT` | How far below the AprilTag z to drive the gripper. |
| `pregrasp_offset_z` | `0.16 m` | `DualArmStates.offset[2]` | Pregrasp hover height above beam. |
| `descend_timeout_steps` | `150` (~5 s) | new | Max steps in `DESCENDING` before recovery. |
| `assemble_timeout_steps` | `300` (~10 s) | new | Max steps in `DUAL_ASSEMBLE` before recovery. |
| `move_away_tol` | `0.02 m` (position only) | new | Pregrasp re-acquisition gate after release. |
| `hole_grad_weight` | `0.3` | `_HOLE_GRADIENT_WEIGHT` | Hole-collinearity term weight. |
| `yaw_grad_weight` | `0.8` | `_YAW_GRADIENT_WEIGHT` | Yaw-channel gradient boost. |

### Top-level states

```
                ┌──────────────┐
   on init ───▶ │   GO_HOME    │
                └─────┬────────┘
                      ▼
                ┌──────────────┐  no unconverged pair
                │  PICK_TASK   │ ─────────────▶ DONE
                └─────┬────────┘
                      ▼ active_pair ← argmin ‖hole_a − hole_b‖
                ┌──────────────┐
                │  DUAL_GRAB   │   (per-arm sub-FSM, both arms must reach READY)
                └─────┬────────┘
                      ▼ both READY
                ┌──────────────┐    timeout
                │ DUAL_ASSEMBLE│────────────▶ RECOVERY_PREGRASP ─▶ DUAL_GRAB
                └─────┬────────┘
                      ▼ hole-pair converged (hysteresis 5)
                ┌──────────────┐
                │RELEASE_GRIPPER│
                └─────┬────────┘
                      ▼
                ┌──────────────┐
                │  MOVE_AWAY   │ ─▶ PICK_TASK
                └──────────────┘
```

### STATE = GO_HOME *(new in perceptive variant)*

Drive both arms to `left_start` / `right_start` using a dedicated quadratic
loss; gradients clamped by the EE-limit block (§4). Transition to `PICK_TASK`
when both arms are within `move_away_tol` (position only) of their start poses,
or after 3 s, whichever first.

### STATE = PICK_TASK

Mirror of `_select_pair()`:

1. Update per-beam convergence using `_beam_converged_by_holes`. Latch into
   `self._convergence` after `conver_hysteresis` consecutive successful checks.
2. If every pair has both beams in `self._convergence`, transition to **DONE**
   (terminal). The original also returns `True` from `check_convergence` in this
   case (`dual_assembly.py:806-807`).
3. Otherwise pick the unconverged pair with the smallest current
   `‖hole_a − hole_b‖`; fall back to `start_loss[a] + start_loss[b]` if hole
   positions are not yet available.
4. Set `self._active_pair_idx`. Transition to `DUAL_GRAB`.

### STATE = DUAL_GRAB

**Arm-to-beam assignment (corrected):** for the chosen `(a, b) =
hole_pairs[active_pair_idx]`:

- If `{prev_left_idx, prev_right_idx} == {a, b}` → reuse the previous mapping
  (sticky).
- Otherwise pick whichever of `(left=a, right=b)` or `(left=b, right=a)` has
  the smaller summed start-loss. This is the literal logic at
  `dual_assembly.py:815-829` and prevents arm-swap races on transient
  cost-flips.

`DUAL_GRAB` runs **two independent per-arm sub-FSMs in parallel** until both
reach `READY`:

```
        ┌──────────────────┐  horiz-pregrasp gate
        │ MOVE_TO_PREGRASP │──────────────────▶ DESCENDING
        └──────────────────┘
                                 ▲              │
                  recovery       │              │ descend_timeout
                                 │              ▼
        ┌──────────────────┐    grasp_pos gate
        │   DESCENDING     │─────────────────▶ CLOSE_GRIPPER
        └──────────────────┘
                                                │ gripper closed (instant)
                                                ▼
                                         ┌────────────┐
                                         │   READY    │
                                         └────────────┘
```

#### SUB = MOVE_TO_PREGRASP_{LEFT,RIGHT}

Gradient drives the EE toward `pregrasp = beam_pose + (0, 0, pregrasp_offset_z,
0, 0)` (refreshed every cycle by `DualArmStates.advance()`,
`dual_assembly.py:108-118`). Transition to `DESCENDING` when
`_pregrasp_horiz_loss[arm, beam_idx] < pregrasp_horiz_tol` (xy + yaw only — z
deliberately excluded so the descent gate does not pull the arm back up).

#### SUB = DESCENDING_{LEFT,RIGHT}

Gradient drives the EE toward `grasp_target = beam_pose − (0, 0,
descent_offset_z + arm.z_trim, 0, 0)`. Transition to `CLOSE_GRIPPER` when
`_beam_pos_loss[arm, beam_idx] < grasp_pos_tol` (position-only, no orientation
term — that matches the original `_CONTACT_POS_THRESHOLD` logic).

If `descend_timeout_steps` elapse without contact → transition back to
`MOVE_TO_PREGRASP` (recovery). The original handles this implicitly by the
pregrasp being reset to "directly above the perceived beam" every cycle; making
it an explicit recovery edge is a hard guarantee.

#### SUB = CLOSE_GRIPPER_{LEFT,RIGHT}

Latch `gripper_closed[arm] = True` instantly on entering this state. From here
on, contact loss does **not** open the gripper for `gripper_open_hold` cycles
(matches `dual_assembly.py:528-547`). Transition to `READY` immediately.

#### SUB = READY_{LEFT,RIGHT}

Hold pose (zero gradient on that arm). When **both** arms are `READY`,
transition the global state to `DUAL_ASSEMBLE`.

If contact is lost while in `READY` for ≥ `gripper_open_hold` cycles → drop
back to `MOVE_TO_PREGRASP` for that arm and notify the global FSM (which stays
in `DUAL_GRAB`).

### STATE = DUAL_ASSEMBLE

Beam gradient = `∇ beam_goal_loss + hole_grad_weight · ∇ hole_collinearity_loss`,
with `yaw_grad_weight` applied to channels 3:5 (the `(sinθ, cosθ)` block). EE
gradients are taken as the beam gradient when the arm is in contact (the same
contact-mode blend as `dual_assembly.py:738-739, 752-753`).

Termination:

- **Success:** the active pair's hole distance has been < `hole_conver_tol`
  for `conver_hysteresis` cycles → mark both beams in `self._convergence`,
  transition to `RELEASE_GRIPPER`.
- **Timeout** (`assemble_timeout_steps`): transition to `RECOVERY_PREGRASP`
  (lift both arms back to pregrasp without releasing the grippers, then
  re-enter `DUAL_ASSEMBLE` from a clean baseline). Do **not** progress to
  `RELEASE_GRIPPER` on a timeout — releasing without convergence drops the
  assembly.

### STATE = RELEASE_GRIPPER

Set `gripper_closed[*] = False`. Transition to `MOVE_AWAY`.

### STATE = MOVE_AWAY

Lift both arms back to the `pregrasp` pose of the just-finished pair, using the
same pregrasp gradient as `MOVE_TO_PREGRASP` but gated on **position-only**
`< move_away_tol = 0.02 m` (no orientation requirement). This matches the
original behaviour where horizontal motion is zeroed while the descent
gradient is below `−0.15` (`dual_assembly.py:743-745, 757-759`), so the arm
rises before drifting laterally.

When both arms have cleared, transition to `PICK_TASK`.

### STATE = DONE *(terminal)*

Zero all gradients; report success.

### STATE = RECOVERY_PREGRASP *(transient, off-FSM-diagram for clarity)*

Lift the active arms to pregrasp height while keeping grippers closed. Gate
identical to `MOVE_AWAY` but with grippers untouched. Re-enters
`DUAL_ASSEMBLE`.

---

## 3. Suggested improvements over `dual_assembly.py`

These are deltas the perceptive variant should adopt, beyond just making the
FSM explicit:

1. **Explicit `RECOVERY_PREGRASP` for assemble-timeout.** The original has no
   well-defined behaviour when assembly stalls; the perceptive variant should
   lift, re-baseline, and retry rather than spin in the gradient bowl.
2. **Explicit `DESCENDING` timeout.** The original implicitly retries through
   `advance()`'s pregrasp reset; making it a hard timeout edge guarantees
   forward progress and produces a logged event for offline review.
3. **Per-arm grasp z trim is part of the public state** (already present:
   `left_z_trim`, `right_z_trim`). Keep this knob and document it as a
   calibration parameter, not a workaround.
4. **Sticky 2×2 assignment is preserved** (this is correct in the original;
   the perceptive variant must not regress to per-cycle re-assignment).
5. **Drop the live-particle filter / mujoco collision check from the
   per-state controllers** if not needed — it adds non-trivial overhead and
   the FSM gates already cover the failure modes it was guarding. (Or keep it
   only inside `DUAL_ASSEMBLE`, where collisions matter most.)
6. **Single source of truth for transitions.** All gates listed above must
   key off the same `_hand_losses` fields the original computes, so behaviour
   stays bit-for-bit comparable in the contact-blend regime.
7. **Logging/telemetry:** emit one structured record per state transition
   (state, arm, beam_idx, gate value, pair distance) so missed grasps and
   timeout recoveries can be diagnosed offline.

---

## 4. End-effector limits (governing constraints)

These limits apply uniformly to every state. They are hard constraints —
clamped after each gradient step, *not* added as soft penalties — so the FSM
can never drive the robot outside the safe envelope regardless of which
controller is active.

### 4.1 Workspace partitioning (no arm crosses the centreline)

Each arm has a half-space it may not leave, expressed in the robot base frame:

- **Left arm:** `y_left ≥ y_centre + arm_half_clearance`.
- **Right arm:** `y_right ≤ y_centre − arm_half_clearance`.

Defaults: `y_centre = 0.0 m`, `arm_half_clearance = 0.02 m` (2 cm dead zone
either side of the centreline to avoid wrist–wrist contact).

The same clamp is applied to the *pregrasp* targets so the gradient never
asks for a cross-body grasp; if the perceived beam lies on the wrong side
for the assigned arm, the 2×2 assignment in `DUAL_GRAB` will already have
swapped arms, and the per-arm clamp is the safety net for the edge case where
both holes of a pair sit on the same side.

### 4.2 Z bounds (table floor + ceiling)

- `z_min = arm_z_floor = 0.65 m` (existing, `dual_assembly.py:407`). Hard
  floor — the gripper may never descend below this in any state.
- `z_max = 1.20 m` (new). Hard ceiling — protects the lift column from
  driving into its mechanical end stop and keeps the AprilTag inside the
  PTU camera FOV.

Both are applied in `normalise_pose` (existing function,
`dual_assembly.py:833-843`), extended to clamp on both sides for arm
elements (the first `no_hands` blocks) and on `z_min` only for beam /
pregrasp blocks.

### 4.3 X-extent (front/back reach limit)

- `x_min = 0.30 m` (no closer to the base than 30 cm — wrist self-collision
  with torso).
- `x_max = 0.95 m` (Kinova Gen3 reachable workspace edge with the lift
  column at nominal height).

Same clamp mechanism as §4.2.

### 4.4 Per-step velocity / step-norm cap

Gradient descent uses `step_size · ‖grad‖`. To prevent any single step from
producing a discontinuous EE jump, clamp the *step* (not the gradient):

- `max_xyz_step = 0.02 m` per step (≈ 60 cm/s at the planner's 30 Hz
  cadence).
- `max_yaw_step = 5°` per step (≈ 150 °/s).

Concretely:

```python
delta = step_size * grad
delta_xyz = delta[0:3]
norm = delta_xyz.norm()
if norm > max_xyz_step:
    delta_xyz *= max_xyz_step / norm
# yaw channels live in (sinθ, cosθ); clamp the implied angle change before
# re-normalising.
```

This applies to every state. Without it, a sudden hole-position jump from
perception (an AprilTag re-acquisition, a frame drop) can produce an
unbounded gradient and a single planner cycle that the controller cannot
follow.

### 4.5 Grasp-frame yaw range

While in `DESCENDING` and `CLOSE_GRIPPER`, restrict the gripper yaw to
`[−90°, +90°]` relative to the beam yaw to forbid grasps that cross the
wrist's natural range. Implemented by clamping the live `(sinθ, cosθ)` after
projection onto the unit circle (extension of `normalise_pose`).

### 4.6 Active-only gradient masking

Arms not currently assigned to a beam (`_left_index is None` or
`_right_index is None`) have their gradient zeroed (existing behaviour,
`dual_assembly.py:747-748, 761-762`). Keep this — combined with the
half-space clamp it guarantees an idle arm cannot drift across the workspace.

---

### Summary table of EE limits

| Limit | Default | Applied where | Existing? |
|---|---|---|---|
| `arm_z_floor` | 0.65 m | every step | yes |
| `arm_z_ceiling` | 1.20 m | every step | new |
| `arm_x_min / x_max` | 0.30 / 0.95 m | every step | new |
| `y_centre / half_clearance` | 0.0 / 0.02 m | every step (per-arm half-space) | new |
| `max_xyz_step` | 0.02 m | every step | new |
| `max_yaw_step` | 5° | every step | new |
| `gripper_yaw_range` | ±90° vs beam yaw | DESCENDING, CLOSE_GRIPPER | new |
| Idle-arm gradient mask | zero | non-active arms | yes |

All limits are exposed as attributes on `DualAssembly` so callers can tune
them per experiment without subclassing.
