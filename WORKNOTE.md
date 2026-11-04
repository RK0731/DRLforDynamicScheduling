# WORKNOTE — Latest Work & Pending Changes

> **Maintainer handoff note.** Read this alongside [`AGENTS.md`](AGENTS.md) (architecture & invariants) and [`README.md`](README.md) (usage).
> **Last updated:** 2026-11-04

This note records (1) the most recent work completed in this repository and (2) changes that are designed/discussed or identified but **not yet implemented**, so that agents and developers can pick up cleanly.

---

## 1. Latest Work Completed

### 1.1 Code Quality Refactor (readability + rich comments)

All core modules were refactored to be more readable while keeping detailed explanatory comments (the maintainer reads the source directly). Explicit imports replaced wildcard imports (`from .exc import *`, `from ..scheduler.sequencing_rule import *`), and type hints / docstrings were added.

| File | Changes |
| :--- | :--- |
| `main.py` | Reorganized CLI groups, added docstrings. Fixed broken/ambiguous arguments. |
| `src/utilities.py` | `setup_logger`, `create_logger`, `draw_gantt_chart` hardened; added `SimulationDatabase`. |
| `src/simulator/simulator.py` | `Shopfloor` lifecycle cleaned; added `_build_run_config`; fixed setting verification. |
| `src/simulator/event.py` | `Narrator` / `Recorder` refactored; added structured event/breakdown/schedule buffers + streaming persistence. |
| `src/simulator/machine.py` | Main production loop restructured into explicit Part I/II/III phases; strategic-idleness logic corrected. |
| `src/simulator/job.py` | `Job` lifecycle hooks clarified; added initial trajectory/PT snapshots for DB persistence. |
| `src/scheduler/scheduler.py` | `CentralScheduler`, `GurobiOptimizer`, `ORTools` refactored; solver return signatures updated. |
| `src/scheduler/sequencing_rule.py` | Added `SPT`, `LWKR`; typed return values. |
| `src/DRL/brain_sequencing.py` | `pathlib` paths, `getattr` instead of `exec`, optimizer `param_groups` LR updates. |

### 1.2 Bug Fixes

1. **OR-Tools decimal-time truncation (schedule/execution MISMATCH).**
   `random_MTBF`, `random_MTTR`, and `pt_cv` produce `0.1`-resolution timestamps (e.g. `104.9`). CP-SAT is integer-only; the old code used `int(env.now)` / `int(release_T)`, truncating `.9 → .0`. Fixed with a `SCALE = 10` transform (all times scaled to integer tenths, divided back on extraction). See `ORTools.SCALE` in `src/scheduler/scheduler.py`.

2. **OR-Tools had a single-objective model.**
   Aligned with Gurobi by adding the 3-tier hierarchical objective (cumulative tardiness → makespan → sum of operation start times) via weighted combination.

3. **Gurobi non-convex bilinear disjunctive constraints.**
   Replaced the MIQCP product `(varOpBeginT + pt) * (1 - varJobPrec)` with a linear **Big-M** MILP formulation (`big_m` computed from release times + total remaining PT).

4. **Passive schedule desynchronization.**
   `CentralScheduler.solve_without_optimization()` previously appended to stale schedules and never updated machines. It now resets `self.schedule`, rebuilds `self.j_op_by_schedule`, and calls `update_machine_after_optimization()`.

5. **Strategic idleness / central-schedule desync race.**
   `Machine` used to pop from `CentralScheduler.schedule[m_idx]` *before* waiting in `strategic_idle`. It now inspects `next_job_in_schedule` without popping, and only pops via `draw_from_schedule` at the exact moment it commits to processing (no `yield` between pop and `after_decision`).

6. **Miscellaneous fixes.**
   - `Narrator.__init__` referenced `self.m_idx` in the fallback branch → `AttributeError`. Fixed.
   - `Shopfloor.verify_simulation_setting()` referenced non-existent `SequencingMethod.opt_scheduler` → crash whenever `-pt_v` / `-rnd_mttr` used. Fixed to check `GurobiOptimizer` / `ORTools`.
   - `Str.Idle end` logged even when not in strategic idleness; `Job 0` printed as `None` (`>= 0` vs `> 0`). Fixed.
   - `release_T` / `hidden_release_T` / `Job.available_T` were left stale after early completion under stochastic settings. Now synchronized to `env.now`.
   - Non-preemptive breakdowns now wait for `status != "processing"` (not just `hidden_release_T`).
   - `draw_gantt_chart` only ran when `draw_gantt > 0`; now also runs when `save_gantt` is enabled.
   - Gantt x-axis range was overwritten each loop iteration; now `max()` across all operations.
   - `LOG_CONFIG` was mutated when disabling the console handler (leaked across runs); now deep-copied.
   - Log pruning iterated non-directory files (`shutil.rmtree` on files could crash); now filters `is_dir()`.
   - Matplotlib `font_manager` flooded `sim.log` at `DEBUG`; root logger set to `WARNING` while `sim_logger` stays `DEBUG`.
   - `main.py`: `-pt_r` now `nargs=2, type=int` (was `type=List[int]`), `-mtbf` / `-mttr` now `type=float`, `-ns` now `store_true`, `-sqc` now `choices`.
    - **LIFO mis-routing (job visits a machine multiple times).** `SequencingMethod.LIFO` returns `-1`; `Machine.after_operation` then did `self.queue.pop(-1)`, which pops whatever job is *last after new arrivals append during processing*, routing the wrong job to its next station (jobs accumulated 6–8 operations instead of 5 and revisited machines). Fixed by normalizing `self.sqc_decision_pos` to a non-negative index in `after_decision` (the `chosen_pos` was already normalized for the `decisions` table, but the instance attribute was left negative). Verified: a fresh LIFO run now has 0 jobs with ≠5 operations and 0 duplicate machine visits.

### 1.3 Persistent SQLite Database Logging

Replaced purely textual logging with a structured, queryable SQLite database at `logs/simulation.db` (configurable via `-db` / `--db_path`).

- Seven tables: `runs`, `jobs`, `operations`, `decisions`, `breakdowns`, `schedule_cycles`, `sim_events`.
- Enabled WAL mode + foreign-key enforcement.
- Schema/indexes live in `SimulationDatabase.SCHEMA_SQL` (`src/utilities.py`).
- `Recorder` buffers records during `env.run()`; `Narrator.post_simulation()` persists them.
- Text logs (`logs/<timestamp>/sim.log`, `gantt_chart.png`) are still written, but retention is now **`keep=3`** (only the 3 most recent folders); the database file is never pruned.

### 1.3.1 Decision-Replay Snapshots (`decisions` table)

Added a dedicated `decisions` table (one row per sequencing decision) so a decision can be rebuilt offline without re-running the simulation:

- `sim_t`, `m_idx`, `decision_type`, `j_idx`, `op_seq` — when, where, and what was chosen.
- `queue_json` — the **exact ordered queue** `[job_index1, job_index2, ...]` as seen by the machine (`self.queue` in arrival order).
- `chosen_pos` — index into `queue_json` (guarantees `queue_json[chosen_pos] == j_idx`).
- `candidates_json` — per-candidate features at decision time, as an object keyed by `j_idx`: `arrival_t`, `wait_t`, `due_t`, `ttd`, `slack`, `current_pt`, `remaining_work`, `num_remaining_ops`, `critical_ratio`. Queue order is authoritative in `queue_json`.
- Reserved DRL columns (currently `NULL`): `state_json`, `action`, `reward`, `next_state_json`, `episode_done`.

Captured in `Machine.after_decision()` **before** `Job.after_decision()` and before the queue is mutated, via `Recorder.record_decision()`.

**Verification:** `decisions` cardinality equals `operations` (e.g., 5,455 for `span=10,000`); `queue_json[chosen_pos] == j_idx` held for **all** decisions across `ORTools`/`FIFO`/`CR` and the 10k run; every decision joined to a matching `operations` row. `decisions` remained the authoritative snapshot even where timeline reconstruction is ambiguous (same-timestamp arrivals: 5,441/5,455 reconstructable, the other 14 resolved exactly by the recorded queue).

### 1.4 Streaming Flush (bounded memory for long runs)

For very long spans, `sim_events` (the dominant table) is now streamed to disk instead of being buffered for the whole episode:

- `Recorder.FLUSH_EVENTS_THRESHOLD = 10_000` — when the in-memory event buffer reaches this size, it is flushed and cleared.
- `Recorder.start_run()` → `SimulationDatabase.create_run()` inserts the `runs` row **upfront** so streamed child rows satisfy FK constraints.
- `Recorder._flush_events()` → `SimulationDatabase.append_events()` (batched insert).
- `Recorder.finalize_episode()` → drains remaining events, calls `SimulationDatabase.append_children()` for `jobs`/`operations`/`decisions`/`breakdowns`/`schedule_cycles`, then `SimulationDatabase.finalize_run()` `UPDATE`s the `runs` row with final KPIs.

**Verification:** `FIFO`, `span=10,000`, `seed=7` → 1,091 jobs, 5,455 operations, 22,890 events, ~1.2 s wall time, all jobs completed, **0 orphan rows** (FK integrity intact).

### 1.5 Documentation

- `README.md` — rewritten (overview, structure, CLI table, SQLite section).
- `AGENTS.md` — created: DJSSP domain model, module guide, DRL design, and core invariants.
- `math_modeling/JSP_model.md` — notation corrected; linear Big-M formulation; hierarchical objectives.

### 1.6 Offline Shopfloor State Reconstruction (`src/analysis/`)

Added `ShopfloorStateReconstructor` (`src/analysis/state_reconstruction.py`), which replays a run's recorded trace to rebuild the **complete shopfloor condition at any time** `t` — without re-running SimPy. This is the foundation for offline RL observation/transition construction.

- `snapshot(t, tie="post")` — pure timeline reconstruction from `operations` + `breakdowns` + `schedule_cycles` + `sim_events`:
  - per machine: `status` (`idle` / `processing` / `strategic_idle` / `down`), `current_job`, ordered waiting `queue` with candidate features, `breakdown` (expected vs. actual remaining repair), `next_job_in_schedule`, `release_T` / `hidden_release_T`;
  - per in-system job: `status`, `current_machine`, `op_seq`, `due_t`, `wait_t`, `completed_ops`, `remaining_machines`, `remaining_expected_pt`, `remaining_actual_pt`, `tardiness_so_far`, `flowtime_so_far`;
  - `summary` aggregate counts and any `decisions` recorded exactly at `t`.
- `snapshot_at_decision(decision_id)` / `iter_snapshots()` — the exact **pre-decision** state: the deciding machine's queue comes verbatim from `decisions.queue_json` (authoritative), and every other machine is reconstructed from the timeline. Operations that start at the same timestamp `t` are classified by decision order (`decision_id`): a machine whose decision preceded this one is already processing, one whose decision follows is still queuing.
- Queue arrival order is resolved exactly via `sim_events.JOB_IN.event_id` (monotonic insertion order), falling back to `(arrival_t, j_idx)`.
- `tie="post"` (default) treats a job starting exactly at `t` as already processing; `tie="pre"` treats it as still queued (used only by `snapshot`, since `snapshot_at_decision` resolves ties via decision order).
- Expected vs. actual values are both exposed (matching the simulator's `release_T` vs. `hidden_release_T` convention).
- CLI: `python -m src.analysis --run_id 1 --time 77.0` or `--decision_id 42`.

**Verification:** head-to-head against a live instrumented simulator (monkeypatched `Machine.after_decision` captured the true state of every machine at each decision) — `snapshot_at_decision` matched all 825 machine-instants across a `span=300` FIFO run and all 825 across a clean LIFO run with **0 mismatches**. DB cross-checks (queue/chosen/candidate features, per-operation processing/waiting classification and `wait_t`, per-machine non-overlap, breakdown-vs-processing non-overlap) passed with 0 errors across 10 clean runs. Strategic idleness and stochastic expected-vs-actual were also verified.

---

## 2. Verification Performed

- Smoke-tested schedulers: `GurobiOptimizer` (license dependent), `ORTools`, `FIFO`, `LIFO`, `Slack`, `CR`, `SPT`, `LWKR`.
- Reproduced historic seeds (`1910729402`, `6971944126`) under `ORTools`; schedule/execution mismatches that were caused by time truncation are resolved (remaining mismatches only occur from genuine stochastic MTTR/PT cases).
- Stochastic settings (`-pt_v`, `-rnd_mttr`, custom `-pt_r 2 12`) exercised without exceptions.
- Multi-run DB persistence verified (runs, jobs, operations, breakdowns, schedule cycles, events).
- Long-run streaming verified as described in §1.4.
- Offline reconstruction verified as described in §1.6 (440/440 decisions exact, arbitrary-time sampling, stochastic expected-vs-actual).

---

## 3. Not Yet Implemented / Known Gaps

> Priority: **P0** = needed for the core research goal, **P1** = important, **P2** = cleanup/nice-to-have.

### 3.1 DRL is not wired end-to-end (P0)

The DRL layer is currently **disconnected** from the simulator:

- `Brain` (`src/DRL/brain_sequencing.py`) is never imported anywhere.
- `Machine.complete_experience()` (in `src/simulator/machine.py`) calls `self.event_creator`, `sequencing_data_generation()`, `build_state()`, and `reward_function()`, **none of which exist** on `Machine`. It is never invoked because no SimPy process consumes `sequencing_learning_event`.
- `Narrator.build_sqc_experience_repository()` / `complete_experience()` exist but are never called by the simulation.
- `src/DRL/reward_function.py` only has a stub (`RWD_func.fun1` returns `None`). `Brain` expects `Machine.get_reward{...}` methods that do not exist.
- `src/DRL/brain_routing.py` is **empty** (placeholder for a routing agent).
- `SequencingMethod.DRL_scheduler` is a no-op placeholder; `main.py` can select it but it will not schedule anything meaningful.

**To do:** define the state/action/reward interface on `Machine` (`sequencing_data_generation`, `build_state`, `reward_function`, `get_reward*`), connect `Brain` to `Shopfloor`, and add a run mode that trains/evaluates the DRL agent.

### 3.2 `drl_transitions` table (P1, partially addressed)

Originally deferred ("we'll design the DRL table later"). The new `decisions` table now carries reserved transition columns (`state_json`, `action`, `reward`, `next_state_json`, `episode_done`) alongside the exact queue/candidate snapshot, so a separate table may be unnecessary. **Remaining work:** populate these columns once the DRL state/reward interface is wired (see §3.1), and decide whether to keep a separate `drl_transitions` table for multi-step/offline replay or write directly into `decisions`.

### 3.3 Simulator/DB edge cases (P1)

- **Multi-process `SimulatorMultiThread` + SQLite:** each process writes to the same `logs/simulation.db`. WAL helps, but high-concurrency writes (e.g., many `thread_no`) have not been stress-tested. Consider per-process DB files or a writer queue if used at scale.
- **`sim_events` verbosity toggle:** no CLI switch yet to disable event streaming during large DRL training sweeps (would save disk). Suggested flag: `-ev` / `--log_events {full,summary,off}`.
- **Failed-run persistence:** if an exception occurs before `post_simulation()`, the `runs` row exists but KPIs remain `NULL` and buffered events may be lost. Consider flushing/finalizing in a `finally` block.
- **`create_logger`** (`src/utilities.py`) is a legacy alternative that reads `config/logger_config.json`; it is unused. Either wire it up or remove it.

### 3.4 Simulator/optimizer leftovers (P2)

- `Machine.overstay_check()` is defined but never called (no job-removal policy is active).
- `src/scheduler/ORToolsJSP.py` is a standalone teaching example, not used by the package.
- `src/DRL/pytorch_padding.py` is a reference snippet (variable-length padding/packing), not integrated.
- `Brain.check_parameter()` references `self.event_creator.due_tightness` / `pt_range` / `E_utliz`; verify attribute names once DRL is wired.
- Gantt chart does not yet overlay `scheduled_begin_t` (`operations` table) for visual scheduled-vs-actual comparison.
- `schedule_cycles` and `operations` tables could expose helper query functions in `src/utilities.py` (e.g., reload a Gantt chart from DB).

### 3.5 Testing (P1)

- No automated unit/integration tests (`test.py` at repo root is a stub containing only a Gurobi import).
- Suggested coverage: deterministic run reproducibility, DB schema/foreign-key integrity, OR-Tools scaling correctness, strategic-idleness behavior.

---

## 4. Suggested Next Steps (ordered)

1. Wire DRL: implement `Machine` DRL hooks + connect `Brain` to `Shopfloor`; add a `-drl` train/eval entry point.
2. Populate the reserved DRL columns in `decisions` (`state_json`, `action`, `reward`, `next_state_json`) or add a dedicated `drl_transitions` table if multi-step replay is needed.
3. Add `sim_events` verbosity flag and stress-test multi-process DB writes.
4. Add a minimal test suite for the invariants in `AGENTS.md` §3.
5. Clean up unused modules (`create_logger`, `overstay_check`, or `ORToolsJSP.py` if intended to stay as docs).

---

## 5. Quick Reference

- **Database query examples:**
  ```sql
  -- Benchmark schedulers
  SELECT sqc_method, e_utliz, AVG(mean_tardiness) AS avg_tard, AVG(wall_time_s) AS avg_wall
  FROM runs GROUP BY sqc_method, e_utliz;

  -- Gantt data for one run
  SELECT j_idx, m_idx, begin_t, actual_pt FROM operations WHERE run_id = 1 ORDER BY m_idx, begin_t;

  -- Scheduled vs. actual start-time mismatches
  SELECT run_id, j_idx, m_idx, scheduled_begin_t, begin_t FROM operations
  WHERE scheduled_begin_t IS NOT NULL AND ABS(scheduled_begin_t - begin_t) > 0.01;

  -- Replay every sequencing decision (exact queue + chosen job + candidate features)
  SELECT sim_t, m_idx, decision_type, queue_json, chosen_pos, j_idx, candidates_json
  FROM decisions WHERE run_id = 1 ORDER BY sim_t;
  ```
- **Rebuild one decision in Python:** read `decisions.queue_json` / `candidates_json` for the chosen `(run_id, sim_t)`; no timeline reconstruction needed.
- **Run a demo:** `python main.py -sqc ORTools -seed 42 -draw 0`
- **Inspect DB:** `sqlite3 logs/simulation.db` (`.tables`, `.schema decisions`).
