# WORKNOTE — Latest Work & Pending Changes

> **Maintainer handoff note.** Read this alongside [`AGENTS.md`](AGENTS.md) (architecture & invariants) and [`README.md`](README.md) (usage).
> **Last updated:** 2026-11-09

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
| `src/scheduler/scheduler.py` | `CentralScheduler`, `GurobiOptimizer`, `ORTools` refactored; solver return signatures updated; added the `OPTIMIZATION_SOLVERS` registry + `resolve_optimization_solver` / `is_optimization_scheduler`. |
| `src/scheduler/sequencing_rule.py` | Added `SPT`, `LWKR`; typed return values; removed the dummy `GurobiOptimizer` / `ORTools` / `DRL_scheduler` classmethods (see §1.7). |
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
   - `Shopfloor.verify_simulation_setting()` referenced non-existent `SequencingMethod.opt_scheduler` → crash whenever `-pt_v` / `-rnd_mttr` used. Now uses `is_optimization_scheduler()` (registry lookup, see §1.7).
   - `Str.Idle end` logged even when not in strategic idleness; `Job 0` printed as `None` (`>= 0` vs `> 0`). Fixed.
   - `release_T` / `hidden_release_T` / `Job.available_T` were left stale after early completion under stochastic settings. Now synchronized to `env.now`.
   - Non-preemptive breakdowns now wait for `status != "processing"` (not just `hidden_release_T`).
   - `draw_gantt_chart` only ran when `draw_gantt > 0`; now also runs when `save_gantt` is enabled.
   - Gantt x-axis range was overwritten each loop iteration; now `max()` across all operations.
   - `LOG_CONFIG` was mutated when disabling the console handler (leaked across runs); now deep-copied.
   - Log pruning iterated non-directory files (`shutil.rmtree` on files could crash); now filters `is_dir()`.
   - Matplotlib `font_manager` flooded `sim.log` at `DEBUG`; root logger set to `WARNING` while `sim_logger` stays `DEBUG`.
   - `main.py`: `-pt_r` now `nargs=2, type=int` (was `type=List[int]`), `-mtbf` / `-mttr` now `type=float`, `-ns` now `store_true`, `-sqc` now `choices`.
    - **LIFO mis-routing (job visits a machine multiple times).** `SequencingMethod.LIFO` used to return Python's `-1` alias; `Machine.after_operation` then did `self.queue.pop(-1)`, which pops whatever job is *last after new arrivals append during processing*, routing the wrong job to its next station (jobs accumulated 6–8 operations instead of 5 and revisited machines). **Fixed at the source:** `LIFO` now returns the concrete index `len(jobs) - 1`; `Machine.after_decision` additionally normalizes `self.sqc_decision_pos` to a non-negative index as a defensive guard for custom rules (the `chosen_pos` stored in `decisions` was already normalized, but the instance attribute could be left negative). Verified: a fresh LIFO run has 0 jobs with ≠5 operations and 0 duplicate machine visits.

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

### 1.7 Graceful Scheduler Dispatch (no dummy classmethods)

Previously `SequencingMethod` carried three no-op placeholders (`GurobiOptimizer`, `ORTools`, `DRL_scheduler`) that existed only so the CLI could name a scheduler; `Narrator` then compared the callable's `__name__` to decide whether to build a `CentralScheduler`. That was fragile (renaming a class silently disabled optimization) and left misleading entries in the `-sqc` help.

Now:

- `src/scheduler/scheduler.py` owns a single registry, `OPTIMIZATION_SOLVERS = {"GurobiOptimizer": GurobiOptimizer, "ORTools": ORTools}`, plus:
  - `resolve_optimization_solver(sqc_method)` → solver class or `None` (accepts a registry name string, a class, or a callable);
  - `is_optimization_scheduler(sqc_method)` → `True` iff the registry resolves a solver.
- `main.py` builds `-sqc` choices from the **real** heuristic classmethods + `OPTIMIZATION_SOLVERS.keys()`; a heuristic name resolves to its callable, an optimization name stays a string looked up by `CentralScheduler`.
- `CentralScheduler` accepts an optional injected `solver_cls` for one-off experiments.
- `sequencing_rule.py` now contains only pure dispatching heuristics (`FIFO`, `LIFO`, `Slack`, `CR`, `SPT`, `LWKR`); the DRL policy is supplied as a callable by `Brain` (it assigns `Machine.job_sequencing` directly), so `DRL_scheduler` was removed.

**Override recipes:**

```python
# (a) Replace a backend globally — the CLI name keeps working.
from src.scheduler import scheduler as S
class MyORTools(S.ORTools): ...
S.OPTIMIZATION_SOLVERS["ORTools"] = MyORTools      # `-sqc ORTools` now uses MyORTools

# (b) Inject a one-off backend without touching the registry.
Shopfloor(..., sqc_method="ORTools", solver_cls=MyORTools)
```

**Verification:** `py_compile` clean; `FIFO`/`LIFO` smoke runs (`span=200`, seed 12345) → 110 ops each, 0 jobs with ≠5 ops, 0 duplicate machine visits, 0 `chosen_pos` violations; `ORTools` run persisted with 25 decisions; `--help` lists `{CR,FIFO,LIFO,LWKR,SPT,Slack,GurobiOptimizer,ORTools}`; both override recipes invoked the subclass 4× on a `span=50` run.

### 1.8 P0/P1 Robustness Fixes (solver, failures, multi-process, reconstruction)

Implemented the review's P0/P1 items:

1. **Solver status guards + passive fallback.**
   - `ORTools` now returns `None` when CP-SAT reports `INFEASIBLE`/`UNKNOWN` instead of extracting garbage from `solver.Value()`.
   - `GurobiOptimizer` checks `model.SolCount == 0` before touching `var.X`.
   - `CentralScheduler.solve_problem_process()` checks the returned mapping and falls back to `solve_without_optimization()`, recording `<STATUS>_PASSIVE_FALLBACK` in `schedule_cycles.solver_status`.
2. **True lexicographic CP-SAT objective.**
   - Replaced the fixed weights (`tardiness*1e7 + makespan*1e4 + sum_begin`) with two staged solves: (1) minimize tardiness; (2) constrain tardiness to the stage-1 optimum and minimize `makespan * W + sum(start)` where `W > max_possible_sum(start)`. The stage-1 solution is reused if stage 2 fails. This removes the weight leak where 1 unit of tardiness could be traded for 1000 units of makespan.
3. **Run lifecycle and failure persistence.**
   - Added `runs.status` (`running`/`completed`/`failed`) and `runs.error`.
   - `Shopfloor.run_simulation()` now catches exceptions, calls `Recorder.fail_run()` (persisting the traceback), and re-raises so the CLI exits non-zero. Previously a failure after `create_run` left an orphaned run with NULL KPIs and no child rows.
4. **Non-interactive confirmation guard.**
   - `verify_simulation_setting()` no longer calls `input()` when `sys.stdin.isatty()` is false. It raises `SimulatorError` unless `force=True`; CLI flag `-y/--force` opts in and bypasses the prompt unconditionally (including on a TTY). This fixes the silent `EOFError`-then-exit-0 behavior in CI/piped runs and keeps automated runs deterministic.
5. **Lazy optional imports.**
   - `gurobipy` and `ortools` are imported inside their solver methods; `matplotlib` inside `draw_gantt_chart`. Heuristic-only runs and the analysis CLI no longer require either optimization backend or a GUI stack.
   - Default `-sqc` changed from `GurobiOptimizer` to `ORTools` (no license needed). A missing `matplotlib` now logs a warning and skips the chart instead of failing the run.
6. **Multi-process runner fixed.**
   - `SimulatorMultiThread` now spawns the module-level `_run_shopfloor(kwargs)`; children construct `Shopfloor` themselves (fixes `TypeError: cannot pickle 'generator' object` on macOS spawn).
   - Each worker gets a distinct derived seed (`base_seed + idx`) and the parent initializes/migrates the SQLite schema once; workers pass `skip_db_init=True`, avoiding `database is locked` on `PRAGMA journal_mode = WAL`.
   - Only the parent prunes log directories (`prune_logs=False` for workers); worker exit codes are checked and re-raised.
7. **Strategic-idleness / idle wake fixes.**
   - `CentralScheduler.update_machine_after_optimization()` calls `Machine.notify_schedule_update()` even for an empty schedule, so a strategic-idle machine re-evaluates instead of waiting forever for sentinel Job `-1`.
   - `Machine.process_production()` falls back to FIFO (with a warning) if a schedule rebuild leaves no work for a machine that still has queued jobs; it idles rather than spinning when the queue is empty.
   - `Machine.process_idle()` re-checks the queue after resetting `sufficient_stock`, so a same-timestamp arrival cannot be lost.
8. **CLI and rule fixes.**
   - Added `--no-random_MTBF` (previously `-rnd_mtbf` was `store_true` with `default=True`, so random MTBF could never be disabled).
   - Added `-mt/--multi_thread` and `-tn/--thread_no`, plus `-y/--force`; `main()` is now importable and validates numeric ranges (`-m_no`, `-span`, `-pt_r`, `-utl`, `-dt`, `-pt_cv`, `-mtbf`, `-mttr`, `-tn`).
   - `Slack` now implements true minimum slack (`due - now - total_remaining_pt`) instead of EDD.
   - `Machine.after_decision()` validates that `chosen_pos` is within the queue bounds; `overstay_check()` is guarded against running during processing and triggers a reschedule.
9. **Database/reconstruction correctness.**
   - Added `operations.completed`; the last operation of a job still processing at episode end is stored with `completed=0`. The reconstructor keeps such an operation (and job) `processing` past its expected `end_t`.
   - `_migrate_schema()` adds missing columns (`runs.status/error`, `operations.completed`, DRL decision columns) and drops legacy `decisions.rule_picks_json`.
   - Mean KPIs divide by completed jobs, not all created jobs.
   - `snapshot_at_decision()` recomputes `summary["num_waiting_jobs"]` after replacing the deciding machine's queue with the authoritative `queue_json`; NULL `queue_json`/`candidates_json` in old DBs are tolerated.
   - DRL experience buffers now use one canonical name (`incomplete_rep_memo`) and `Machine.complete_experience()` logs exceptions instead of swallowing them.

**Verification:** `py_compile` clean on all touched files. FIFO/ORTools `span=200` seed 12345: 22/22 jobs, 110 ops each, 0 jobs with ≠5 ops, 0 duplicate machine visits, 0 `chosen_pos` violations; reconstructor 110/110 decisions exact with 0 summary/queue mismatches. ORTools runs report `OPTIMAL` and no fallback; a synthetic `INFEASIBLE` solver produced 6 `INFEASIBLE_PASSIVE_FALLBACK` cycles and completed normally. A synthetic solver exception produced `runs.status='failed'`, a stored traceback, and exit code 1. Non-interactive `-pt_v` without `-y` exited 1 with no `runs` row; with `-y` it completed. `-mt -tn 2` produced two completed runs (FIFO and ORTools) with distinct seeds. Schema migration test dropped/re-added `runs.status/error`, `operations.completed`, and `decisions.candidates_json` and restored all columns.

### 1.9 Automated Verification Suite (`tests/`)

Added a stdlib-`unittest` suite (no pytest dependency) that can be run after every change:

```bash
uv run python -m unittest discover -s tests -t . -v
# equivalents: uv run python test.py   or   make test
```

- `tests/base.py` isolates each test in a temporary directory by monkeypatching `src.utilities.LOG_ROOT_DIR` / `DEFAULT_DB_PATH` and exporting `SIM_LOG_ROOT` (inherited by spawn-based workers). The project's `logs/simulation.db` and retained log folders are never touched.
- `tests/test_project_integrity.py`: compiles every Python source under `src/`, imports all core modules (including the DRL brain), checks the scheduler registry/rules, and verifies required project artifacts exist.
- `tests/test_sequencing_rules.py`: FIFO/LIFO/SPT/LWKR/CR/Slack correctness, including the LIFO `-1` regression and the difference between Slack and EDD.
- `tests/test_scheduler.py`: registry resolution, CP-SAT lexicographic behavior, `SCALE=10` decimal output, and `INFEASIBLE`/`UNKNOWN` → `None` guards (patched `CpSolver.Solve`).
- `tests/test_simulation.py`: FIFO and OR-Tools end-to-end invariants (job/operation counts, unique machine visits, `queue_json[chosen_pos] == j_idx`, candidate-key integrity, FK checks, scheduled-vs-actual alignment), same-seed reproducibility, streaming-flush event preservation, solver fallback, solver-exception run failure, non-interactive guard, and multi-process isolation with distinct seeds.
- `tests/test_database.py`: new-column schema, migration/drop-legacy behavior, `runs.status/error` lifecycle, and completed-status backfill.
- `tests/test_reconstruction.py`: exact per-decision queue replay, summary counts, arbitrary-time snapshots, and incomplete-operation phase semantics.
- `tests/test_cli.py`: parser defaults/flags, numeric validation, subprocess FIFO smoke run, non-interactive abort exit code, and usage-error exit code.

**Result:** 43 tests pass in ~5 s with the project venv (`py_compile`/`git diff --check` also clean).

### 1.10 uv Migration (Python 3.13 + Locked Dependencies)

- Installed **uv 0.13.7** (`brew install uv`) and a managed **CPython 3.13.18** (`uv python install 3.13`).
- Changed `.python-version` from the machine-specific venv name `Py310DRLVenv` to `3.13`.
- Added `pyproject.toml` with `requires-python = ">=3.13,<3.14"` and `[tool.uv] package = false`:
  - core: `numpy`, `simpy`, `tabulate`, `ortools`, `torch`;
  - extras: `gurobi` (`gurobipy`), `plot` (`matplotlib`), `analysis` (`pandas`, `scikit-learn`, `h5py`, `PyYAML`).
- Generated `uv.lock` (58 packages) and created `.venv` with `uv sync --all-extras` (38 installed packages). Key resolved versions: Python 3.13.18, numpy 2.5.4, simpy 4.1.2, tabulate 0.10.0, ortools 9.15.6755, torch 2.15.0, gurobipy 13.0.3, matplotlib 3.11.2, pandas 2.3.3, scikit-learn 1.9.1, h5py 3.16.0, PyYAML 6.0.3.
- Regenerated `requirements.txt` as a pip-compatible export of `uv.lock` (`uv export --format requirements-txt --all-extras --no-hashes`); it is no longer hand-maintained.
- Updated `Makefile` (`make sync`, `make test` via `uv run`), `.vscode/settings.json` (project `.venv`), and `.gitignore` (`.venv/`).
- Fixed Python 3.13 `ResourceWarning: unclosed database` noise by wrapping every `sqlite3.connect` in `contextlib.closing` (production DB/analysis code and tests).
- Verified under Python 3.13: all 43 tests pass (`uv run python -m unittest discover -s tests -t . -v`), and FIFO/ORTools `span=50` smoke runs complete with 5/5 jobs and 25 operations.

Note: if an old pyenv virtualenv is still active, `uv run` prints `VIRTUAL_ENV=... does not match the project environment path .venv`; run `deactivate` once or ignore the warning — `uv run` still uses `.venv`.

### 1.11 Strategic Idleness Survives Machine Breakdown (bug fix)

A machine can be strategically idle (waiting for its scheduled head job) when a breakdown hits. `Narrator.process_machine_breakdown` overwrites `Machine.status` with `"down"` then `"idle"`, so the `"strategic_idle"` marker was lost, and — because the `STR_IDL_END` guards tested `status == "strategic_idle"` — the span was never closed. Offline reconstruction (`ShopfloorStateReconstructor._strategic_idle`) would then see a dangling `STR_IDL_ON` and misreport the machine as strategically idle.

**Fix:** strategic idleness is now tracked by a dedicated `Machine.in_strategic_idle` flag that the breakdown flow never touches. The flag is the source of truth for entering/exiting strategic idleness:

- `check_strategic_idleness()` sets the flag (and logs `STR_IDL_ON`) based on the flag, not `status`.
- A new `Machine._end_strategic_idle()` helper closes the span and logs `STR_IDL_END`, called from `job_arrival()`, `update_status_after_new_schedule()`, and `notify_schedule_update()` (the latter also fixes the empty-schedule FIFO-fallback path, which previously left a dangling `STR_IDL_ON`).

**Verification:** 4 new unit tests in `tests/test_strategic_idle.py` (entry, breakdown-clobber survival, reschedule-close, empty-schedule-close). A real `ORTools` run with breakdowns shows `STR_IDL_ON`/`STR_IDL_END` strictly alternating per machine (0 violations).

### 1.12 HiGHS MILP Solver (open-source alternative to Gurobi)

Added a third centralized backend, **HiGHS** (`highspy`, MIT license), as a free open-source MILP alternative to Gurobi:

- `HiGHSOptimizer` (`src/scheduler/scheduler.py`) formulates the same linear Big-M disjunctive MILP as `GurobiOptimizer`, but solves it with HiGHS — no commercial license required.
- **Continuous time variables**: unlike CP-SAT (integer-only, `SCALE = 10`), HiGHS uses continuous times, so sub-unit `0.1` precision is exact with no truncation risk.
- **Two-stage lexicographic objective** (mirroring `ORTools`): stage 1 minimizes cumulative tardiness; stage 2 constrains tardiness to the stage-1 optimum and minimizes `makespan * W + sum(start)`. Falls back to the stage-1 solution if stage 2 has no incumbent, and returns `None` (→ passive schedule) if stage 1 has no primal feasible solution.
- Registered in `OPTIMIZATION_SOLVERS`, so `-sqc HiGHS` is an automatic CLI choice. `highspy` added to `pyproject.toml` / `uv.lock` / `requirements.txt`.

**Important coexistence limitation (discovered during testing):** OR-Tools bundles its own (older) HiGHS, which is **binary-incompatible** with the standalone `highspy` build — loading both native libraries into one process segfaults on macOS (a `libhighs`/`libortools` symbol clash). `-sqc HiGHS` is unaffected (OR-Tools is imported lazily only when `ORTools` is selected), but **never benchmark OR-Tools and HiGHS in the same process**. The test suite isolates HiGHS in a fresh subprocess (`tests/test_highs.py`) for this reason.

**Verification:** `tests/test_highs.py` (subprocess-isolated) covers tardiness-before-makespan ordering, continuous-time decimal precision (`1.5` not truncated), and a full `span=60` episode with 0 schedule/execution mismatches. Full suite: **50 tests pass**.

---

## 2. Verification Performed

- Smoke-tested schedulers: `GurobiOptimizer` (license dependent), `ORTools`, `HiGHS`, `FIFO`, `LIFO`, `Slack`, `CR`, `SPT`, `LWKR`.
- Reproduced historic seeds (`1910729402`, `6971944126`) under `ORTools`; schedule/execution mismatches that were caused by time truncation are resolved (remaining mismatches only occur from genuine stochastic MTTR/PT cases).
- Stochastic settings (`-pt_v`, `-rnd_mttr`, custom `-pt_r 2 12`) exercised without exceptions.
- Multi-run DB persistence verified (runs, jobs, operations, breakdowns, schedule cycles, events).
- Long-run streaming verified as described in §1.4.
- Offline reconstruction verified as described in §1.6 (440/440 decisions exact, arbitrary-time sampling, stochastic expected-vs-actual).
- P0/P1 robustness checks (§1.8): solver `INFEASIBLE` → passive fallback; synthetic solver exception → `runs.status='failed'` + exit 1; non-interactive `-pt_v` without `-y` → exit 1 and no `runs` row; `-mt -tn 2` → two completed runs with distinct seeds; old-schema migration restores new columns; FIFO/ORTools `span=200` and reconstructor cross-checks remain clean.

---

## 3. Not Yet Implemented / Known Gaps

> Priority: **P0** = needed for the core research goal, **P1** = important, **P2** = cleanup/nice-to-have.

### 3.1 DRL is not wired end-to-end (P0)

The DRL layer is currently **disconnected** from the simulator:

- `Brain` (`src/DRL/brain_sequencing.py`) is never imported anywhere.
- `Machine.complete_experience()` (in `src/simulator/machine.py`) calls `self.event_creator`, `sequencing_data_generation()`, `build_state()`, and `reward_function()`, **none of which exist** on `Machine`. It is never invoked because no SimPy process consumes `sequencing_learning_event`. The exception is now logged (not swallowed), and `Narrator` uses the same `incomplete_rep_memo` buffer name as `Brain`/`Machine`.
- `Narrator.build_sqc_experience_repository()` / `complete_experience()` exist but are never called by the simulation.
- `src/DRL/reward_function.py` only has a stub (`RWD_func.fun1` returns `None`). `Brain` expects `Machine.get_reward{...}` methods that do not exist.
- `src/DRL/brain_routing.py` is **empty** (placeholder for a routing agent).
- There is no `DRL_scheduler` entry in `SequencingMethod` any more (§1.7); the DRL policy must be wired as a callable that sets `Machine.job_sequencing` (as `Brain` intends to do).

**To do:** define the state/action/reward interface on `Machine` (`sequencing_data_generation`, `build_state`, `reward_function`, `get_reward*`), connect `Brain` to `Shopfloor`, and add a run mode that trains/evaluates the DRL agent.

### 3.2 `drl_transitions` table (P1, partially addressed)

Originally deferred ("we'll design the DRL table later"). The new `decisions` table now carries reserved transition columns (`state_json`, `action`, `reward`, `next_state_json`, `episode_done`) alongside the exact queue/candidate snapshot, so a separate table may be unnecessary. **Remaining work:** populate these columns once the DRL state/reward interface is wired (see §3.1), and decide whether to keep a separate `drl_transitions` table for multi-step/offline replay or write directly into `decisions`.

### 3.3 Simulator/DB edge cases (P1)

- **Multi-process `SimulatorMultiThread` + SQLite:** the parent now creates/migrates the schema once and workers pass `skip_db_init=True`, which removes the observed `database is locked` race on WAL setup. Each worker still writes to the same `logs/simulation.db`; WAL serializes writers, but high concurrency (many `thread_no`, large `span`) has not been stress-tested. Consider per-process DB files or a writer queue if used at scale.
- **`sim_events` verbosity toggle:** no CLI switch yet to disable event streaming during large DRL training sweeps (would save disk). Suggested flag: `-ev` / `--log_events {full,summary,off}`.
- **Failed-run persistence:** fixed for exceptions after `start_run` (`runs.status='failed'`, `runs.error` traceback, exit non-zero). Buffered events/child rows written before the failure may still be absent; if full crash diagnostics are needed, flush the recorder in the exception path.
- **`create_logger`** (`src/utilities.py`) is a legacy alternative that reads `config/logger_config.json`; it is unused. Either wire it up or remove it.

### 3.4 Simulator/optimizer leftovers (P2)

- `Machine.overstay_check()` is defined but never called (no job-removal policy is active). It is now guarded against running during an active operation and triggers a reschedule after removals, but it still needs an explicit policy hook if used.
- `src/scheduler/ORToolsJSP.py` is a standalone teaching example, not used by the package.
- `src/DRL/pytorch_padding.py` is a reference snippet (variable-length padding/packing), not integrated.
- `Brain.check_parameter()` references `self.event_creator.due_tightness` / `pt_range` / `E_utliz`; verify attribute names once DRL is wired.
- Gantt chart does not yet overlay `scheduled_begin_t` (`operations` table) for visual scheduled-vs-actual comparison.
- `schedule_cycles` and `operations` tables could expose helper query functions in `src/utilities.py` (e.g., reload a Gantt chart from DB).

### 3.5 Testing (P1 — mostly addressed)

- `tests/` now contains 50 stdlib-`unittest` tests covering project-wide compilation/imports, rules, solver lexicographic/status behavior (CP-SAT + HiGHS), FIFO/OR-Tools simulation invariants, same-seed reproducibility, streaming flush, solver fallback/failure, non-interactive guard, multi-process isolation, DB migration/status lifecycle, reconstruction, strategic-idleness state assertions, and CLI entry points. Run with `uv run python -m unittest discover -s tests -t . -v` (or `uv run python test.py` / `make test`). Note: HiGHS tests run in a subprocess (`tests/test_highs.py`) because OR-Tools and standalone `highspy` are binary-incompatible in one process (§1.12).
- Still worth adding: deterministic reproducibility (same seed → identical KPIs across two runs), a long-run streaming-threshold test, and explicit strategic-idleness state assertions. The Gurobi path cannot be exercised here without a valid license.

---

## 4. Suggested Next Steps (ordered)

1. Wire DRL: implement `Machine` DRL hooks + connect `Brain` to `Shopfloor`; add a `-drl` train/eval entry point.
2. Populate the reserved DRL columns in `decisions` (`state_json`, `action`, `reward`, `next_state_json`) or add a dedicated `drl_transitions` table if multi-step replay is needed.
3. Add `sim_events` verbosity flag and stress-test multi-process DB writes.
4. **Warm starts / incremental re-solving** for the centralized solvers (CP-SAT `AddHint`, HiGHS `setSolution`, Gurobi MIP starts): reusing the previous schedule when a new job arrives can cut per-cycle solve time by an order of magnitude, and would let the hardcoded 10 s / 5 s solve budgets drop to sub-second.
5. **HiGHS polish**: collapse the two staged solves into HiGHS' native lexicographic multi-objective (`addLinearObjective`) for a single solve, and (optionally) resolve the OR-Tools/`highspy` in-process coexistence limitation (§1.12) to allow in-process benchmarking.
6. Clean up unused modules (`create_logger`, `overstay_check`, or `ORToolsJSP.py` if intended to stay as docs).

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
- **Run a demo:** `uv run python main.py -sqc ORTools -seed 42 -draw 0` (or `-sqc HiGHS` for the free open-source MILP backend)
- **Inspect DB:** `sqlite3 logs/simulation.db` (`.tables`, `.schema decisions`).
