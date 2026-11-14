# Project & Architecture Guide for AI Agents (`AGENTS.md`)

> **See also:** [`WORKNOTE.md`](WORKNOTE.md) — chronological handoff log of the latest completed work, verification results, and the prioritized list of changes not yet implemented. Check it before starting new work.

## 1. Nature of the Research Project

- **Project Title**: *Multi-agent Imitational, Chronological, and Asynchronous Reinforcement Learning Framework for Production Scheduling* (`DRLforDynamicScheduling`)
- **Primary Goal**: Develop a Deep Multi-Agent Reinforcement Learning (MARL) framework capable of making decentralized, real-time scheduling decisions for the **Dynamic Job Shop Scheduling Problem (DJSSP)** under stochastic shopfloor disruptions, while leveraging centralized mathematical optimization (MILP / CP-SAT) and heuristic dispatching rules both as benchmarks and for imitation/guided learning.

### 1.1 Problem Domain: Dynamic Job Shop Scheduling Problem (DJSSP)
In a job shop with $M$ machines (`m_no`), jobs arrive dynamically over time rather than being known statically at $t = 0$:
1. **Dynamic Job Arrivals**:
   - Jobs arrive according to a Poisson process over a nominal horizon `span`.
   - Given the nominal operation processing time range `pt_range` $= [t_{\min}, t_{\max}]$ with mean $\mathbb{E}[PT] = \frac{t_{\min} + t_{\max}}{2}$ and target system utilization rate `E_utliz` ($U$), the mean inter-arrival time is $\beta = \frac{\mathbb{E}[PT]}{U}$.
   - Total jobs generated in an episode is $\text{round}(\text{span} / \beta)$, with inter-arrival times sampled from $\text{Exponential}(\beta)$.
2. **Job Routing & Due Dates**:
   - Each job $j$ must visit all $M$ machines once along a randomly shuffled permutation (`trajectory`).
   - Each operation $(j, m)$ has a nominal integer processing time `pt_by_m_idx[m]` $\in [t_{\min}, t_{\max}]$.
   - If `processing_time_variability` (`pt_cv > 0`) is enabled, the *actual* processing time (`actual_remaining_pt`) is drawn from $\mathcal{N}(\mu = pt, \sigma = pt \cdot \text{pt\_cv})$, clipped to `pt_range` and rounded to 1 decimal place (`0.1`). Crucially, **decision makers (schedulers/agents) only observe the expected/nominal processing time (`remaining_pt`)**, whereas the simulator advances by the actual processing time (`actual_remaining_pt`).
   - Each job is assigned a due date upon arrival:
     $$d_j = \text{round}\left(t_{\text{arrival}} + \left(\sum_{m} t_{j,m}\right) \cdot \text{Uniform}(1.2, \text{due\_tightness})\right)$$
3. **Machine Breakdowns (Non-Preemptive)**:
   - When `machine_breakdown` is enabled, each machine experiences periodic failures governed by `MTBF` (Mean Time Between Failures, either deterministic or sampled from $\text{Exponential}(\text{MTBF})$) and `MTTR` (Mean Time To Repair, either deterministic or sampled from $\text{Uniform}(0.5 \cdot \text{MTTR}, 1.5 \cdot \text{MTTR})$).
   - Breakdowns are **non-preemptive**: if a failure event triggers while a machine is processing a job, the breakdown commences immediately *after* the current operation finishes (`m_instance.hidden_release_T`).
   - While broken down, the scheduler observes the *expected* recovery time `release_T = actual_begin + MTTR`, whereas the machine actually recovers at `hidden_release_T = actual_begin + bkd_t`.
4. **Performance Metrics (KPIs)**:
   - **Tardiness**: $T_j = \max(0, C_j - d_j)$, where $C_j$ is job $j$'s completion time. Primary objective is minimizing cumulative/mean tardiness $\sum_j T_j$.
   - **Flowtime**: $F_j = C_j - t_{\text{arrival}, j}$.
   - **Makespan**: $\max_j C_j$.

---

## 2. Architecture & Module Guide

### 2.0 Environment & Dependencies

- **Python**: 3.13, pinned in `.python-version` (`3.13`) and `pyproject.toml` (`>=3.13,<3.14`).
- **Manager**: `uv`. The project environment is `.venv`, created from `pyproject.toml` + `uv.lock` via `uv sync --all-extras`.
- **Commands**: run everything through `uv run python ...`; use `uv add` / `uv lock` when changing dependencies. `requirements.txt` is a generated pip-compatible export, not the source of truth.
- **Extras**: `gurobi` (optional license), `plot` (matplotlib/Gantt), `analysis` (pandas/scikit-learn/h5py/PyYAML). Core includes `numpy`, `simpy`, `tabulate`, `ortools`, and `torch`.

### 2.1 Entry Point & Utilities
- `main.py`: CLI parser that validates numeric ranges and maps flags into `Simulator.run(**kwargs)` (single process) or `SimulatorMultiThread(**kwargs)` (`-mt/--multi_thread`). Default scheduler is `ORTools`; `-y/--force` enables non-interactive stochastic runs, and `--no-random_MTBF` disables exponential MTBF sampling.
- `src/utilities.py`:
  - `setup_logger(stream, keep=3, prune=True)`: Creates a timestamped directory under `logs/<YYYYMMDD-HHMMSSffffff>/sim.log`, prunes older log folders beyond `keep=3` (retaining only the 3 most recent text log folders while preserving `logs/simulation.db`), logs `DEBUG` details to `sim.log`, and streams `INFO` summaries to console. Multi-process workers pass `prune=False`; only the parent prunes. `_prune_log_dirs()` is best-effort and tolerates concurrent deletion.
  - `SimulationDatabase(db_path, init_schema=True)`: Manages the persistent SQLite database (`logs/simulation.db` by default) containing 7 relational tables (`runs`, `jobs`, `operations`, `decisions`, `breakdowns`, `schedule_cycles`, `sim_events`) with WAL mode and foreign key constraints. Uses a streaming persistence strategy: `create_run` inserts the `runs` row upfront with `status='running'`, `append_events` streams `sim_events` in batches (triggered by `Recorder.FLUSH_EVENTS_THRESHOLD`), `append_children` flushes the smaller tables once at the end, and `finalize_run` marks the run `completed` and updates final KPIs. `fail_run()` marks a partially executed run `failed` and stores the traceback in `runs.error`. `_migrate_schema()` adds new columns (`runs.status/error`, `operations.completed`, DRL decision columns) and drops the legacy `decisions.rule_picks_json`. The `decisions` table stores the authoritative per-decision replay snapshot (`queue_json`, `chosen_pos`, `candidates_json` keyed by `j_idx`) plus reserved DRL columns.
  - `draw_gantt_chart(logger, recorder, **kwargs)`: Plots each job's operations (`recorder.j_operation_dict`) as color-coded horizontal bars and overlays machine breakdowns (`recorder.m_bkd_dict`) as hatched white bars, saving `gantt_chart.png` inside the active log folder. `matplotlib` is imported lazily and a missing install only skips the chart.

### 2.2 Discrete-Event Simulator (`src/simulator/`)
Built on **SimPy** (`simpy.Environment`):
- `src/simulator/simulator.py`:
  - `Simulator` & `SimulatorMultiThread`: Single-process and multi-process wrappers around `Shopfloor`. `SimulatorMultiThread` spawns a module-level `_run_shopfloor(kwargs)` worker (the `Shopfloor` itself is not picklable on macOS/Windows spawn), derives a distinct seed per worker, initializes the SQLite schema once in the parent (avoiding `database is locked` races), and re-raises if any worker exits non-zero.
  - `Shopfloor`: Instantiates `simpy.Environment`, `Recorder`, `Machine` list (`m_list`), and `Narrator`. Runs `env.run(until=span + 1000)` so all jobs arriving within `[0, span]` complete their remaining operations. `run_simulation()` wraps the episode in `try/except`: on failure it calls `Recorder.fail_run()` (marking the `runs` row `failed` with the traceback) and re-raises so the CLI exits non-zero. `verify_simulation_setting()` refuses to call `input()` when stdin is not a TTY unless `force=True` (`-y/--force`).
- `src/simulator/event.py`:
  - `Narrator`:
    - Spawns `process_job_creation()`: Generates jobs at pre-sampled exponential intervals, registers them in `recorder.in_system_jobs`, triggers `central_scheduler.build_schedule_event` (if in optimization mode), and routes each job to its first machine.
    - Spawns `process_machine_breakdown(m, random_MTBF, random_MTTR)` for each machine: Waits for `MTBF`, waits for any in-progress operation on `m` to complete, sets `m.working_event = env.event()` and `m.status = "down"`, triggers `central_scheduler.build_schedule_event`, waits for the repair duration `bkd_t`, records `[actual_begin, actual_end]` in `recorder.m_bkd_dict[m.m_idx]`, and triggers `m.working_event.succeed()`.
    - `post_simulation()`: Verifies all jobs completed, checks for schedule-execution start-time mismatches (when `pt_cv == 0` and `random_MTTR == False`), and prints tabulated configuration and KPI summaries. Mean tardiness/flowtime divide by the number of *completed* jobs, not by all created jobs.
  - `Recorder`: Central state & statistics store shared across all entities (`in_system_jobs`, `j_operation_dict`, `j_tardiness_dict`, `j_flowtime_dict`, `m_bkd_dict`, decision counters `sqc_cnt_opt`, `sqc_cnt_SI`, `sqc_cnt_reactive`, `sqc_cnt_passive`). Buffers structured rows for `operations`, `decisions`, `breakdowns`, `schedule_cycles`, and `sim_events`; `record_decision(...)` captures the exact pre-decision queue and candidate features. `finalize_episode()` marks the last operation of a job still processing at episode end with `completed=0` (its `end_t` is only the expected end). `start_run()` accepts `skip_db_init` so multi-process workers reuse the parent-created schema. `fail_run()` forwards a failure traceback to `SimulationDatabase.fail_run()`.
- `src/simulator/job.py`:
  - `Job`: Dataclass holding `j_idx`, `trajectory`, `pt_by_m_idx`, `remaining_machines`, `remaining_pt` (nominal), `actual_remaining_pt` (actual), `due`, `status` (`"queuing"`, `"processing"`, `"completed"`), `available_T` (expected completion time of current operation), and `operation_record`.
  - Lifecycle hooks:
    - `after_arrival()`: Updates `arrival_T = env.now`.
    - `after_decision(m_idx, wait)`: Called when a machine starts processing the job; records `[m_idx, env.now, actual_pt, wait]`, appends a structured `operations` row with `completed=1` (downgraded to `0` at episode end if still running), sets `status = "processing"`, and sets `available_T = env.now + remaining_pt[0]`.
    - `after_operation()`: Called when the operation finishes; pops the completed operation from `remaining_operations`, `remaining_machines`, `remaining_pt`, `actual_remaining_pt`, refreshes `available_T = env.now`, and returns the next machine index (or `-1` after calling `completion()`).
- `src/simulator/machine.py`:
  - `Machine`: Models a workstation with `m_idx`, `queue`, `status` (`"idle"`, `"processing"`, `"strategic_idle"`, `"down"`), `release_T` (expected release time visible to scheduler), and `hidden_release_T` (actual release time).
  - Main loop `process_production()` has three phases per operation:
    - **Part I (Sequencing Decision)**:
      - **Scheduled Mode (`self.schedule_mode == True`)**: Follows the centralized schedule (`CentralScheduler`). If the next scheduled job (`self.next_job_in_schedule`) has not arrived in `self.queue` yet, the machine enters **Strategic Idleness** (`status = "strategic_idle"`) and waits on `self.required_job_in_queue_event` even if other jobs are already waiting in its queue. Once the required job is in `self.queue` and the machine is operational, it calls `draw_from_schedule(m_idx)` to pop the job from the schedule and starts processing. If a schedule rebuild leaves `next_job_in_schedule == -1` while jobs are queued (e.g. a removal/reschedule race), the machine logs a warning and falls back to FIFO for that decision instead of waiting forever; `CentralScheduler.update_machine_after_optimization()` also calls `notify_schedule_update()` for empty schedules so a strategic-idle machine wakes and re-evaluates. `process_idle()` re-checks the queue after resetting `sufficient_stock` so a same-timestamp arrival cannot be lost.
      - **Reactive Mode (`len(self.queue) > 1`)**: Calls heuristic rule or DRL policy (`self.job_sequencing(jobs=self.queue)`) to pick a queue position.
      - **Passive Mode (`len(self.queue) == 1`)**: Picks the sole job at index `0`.
    - **Part II (Execution)**: Calls `after_decision()`, sets `status = "processing"`, and yields `env.timeout(actual_pt)`.
    - **Part III (Post-Operation)**: Calls `after_operation()` (which pops the job from `self.queue` and routes it to its next machine), yields `env.timeout(0)` so concurrent events at the same timestamp settle, and checks for breakdown (`working_event`) or empty queue (`process_idle()`).

### 2.3 Centralized Scheduler & Heuristics (`src/scheduler/`)
- `src/scheduler/sequencing_rule.py`:
  - `SequencingMethod`: Provides classmethods for pure reactive dispatching rules (`FIFO`, `LIFO`, `Slack`, `CR`, `SPT`, `LWKR`). `Slack` is the true minimum-slack rule (`due - now - total_remaining_processing_time`), not EDD. Centralized solvers and the DRL policy are intentionally not defined here (no dummy placeholders).
- `src/scheduler/scheduler.py`:
  - `OPTIMIZATION_SOLVERS = {"GurobiOptimizer": GurobiOptimizer, "ORTools": ORTools, "HiGHS": HiGHSOptimizer}` — the single registry of centralized backends. `resolve_optimization_solver(sqc_method)` returns the backing class (or `None` for heuristics), and `is_optimization_scheduler(sqc_method)` is the predicate used by `Narrator` / `Shopfloor`. `sqc_method` may be a registry name string, a solver class, or a heuristic callable.
  - `CentralScheduler` accepts an optional injected `solver_cls` for one-off experiments; otherwise it resolves the backend from `OPTIMIZATION_SOLVERS`. Overriding `OPTIMIZATION_SOLVERS["ORTools"]` replaces the backend while keeping the CLI name `-sqc ORTools`.
  - `CentralScheduler`:
    - Runs `solve_problem_process()` which waits on `self.build_schedule_event` (triggered upon any new job arrival or machine breakdown).
    - Extracts remaining unstarted operations for all `in_system_jobs`:
      - If `job.status == "queuing"`: all `job.remaining_machines` are unstarted.
      - If `job.status == "processing"`: `job.remaining_machines[0]` is already executing, so only `job.remaining_machines[1:]` are unstarted.
    - Checks pairwise machine intersections (`self.job_intersections`). If $\le 1$ job is in the system or no remaining trajectories share a machine, calls `solve_without_optimization()` (passive schedule). Otherwise invokes the backend resolved from `OPTIMIZATION_SOLVERS` (`GurobiOptimizer`, `ORTools`, or `HiGHS`, or an injected `solver_cls`). If the backend returns no usable mapping (infeasible/unknown/no incumbent), it logs a warning, calls `solve_without_optimization()`, and records `<STATUS>_PASSIVE_FALLBACK` in `schedule_cycles.solver_status`.
    - `convert_to_schedule(varOpBeginT)`: Sorts all remaining operations $(j, m)$ chronologically by optimized start time `varOpBeginT[j, m]` to construct `self.schedule[m_idx]` (ordered list of `j_idx`) and `self.j_op_by_schedule[j_idx]` (list of `(m_idx, expected_begin_T)`), then calls `update_machine_after_optimization()`.
  - `GurobiOptimizer`:
    - Formulates a Mixed-Integer Linear Program (MILP) using linear Big-M disjunctive constraints on binary precedence variables `varJobPrec[j1, j2, m]` and continuous start times `varOpBeginT[j, m]`.
    - Uses Gurobi's hierarchical multi-objective optimization (`setObjective` / `setObjectiveN`):
      1. **Tier 1**: Minimize cumulative tardiness $\sum_j T_j$;
      2. **Tier 2**: Minimize schedule makespan $C_{\max}$;
      3. **Tier 3**: Minimize sum of operation start times $\sum_{j, m} b_{j, m}$ (eliminating lazy delays).
    - Guards `model.SolCount == 0` (infeasible/unbounded/no incumbent) and returns `None` so `CentralScheduler` can fall back to the passive schedule instead of reading undefined `.X` values.
  - `ORTools`:
    - Formulates a Constraint Programming (CP-SAT) model using `NewIntervalVar` and `AddNoOverlap`.
    - **Time Scaling Invariant (`SCALE = 10`)**: Because SimPy timestamps and durations can have `0.1` decimal resolution (from `random_MTBF`, `random_MTTR`, or `pt_cv`), `ORTools` scales all time values by `10` into integer tenths of a time unit inside CP-SAT and divides solved start times by `10.0`.
    - Optimizes the 3-tier hierarchy **lexicographically with two staged solves** (not fixed weights): stage 1 minimizes cumulative tardiness; stage 2 constrains tardiness to the stage-1 optimum and minimizes `makespan * W + sum(start)` with `W > max_possible_sum(start)`. If stage 2 fails, the stage-1 feasible solution is reused. If stage 1 is infeasible/unknown, returns `None` so the scheduler falls back to the passive schedule. This avoids the previous weight leak where 1 unit of tardiness could be traded for 1000 units of makespan.
  - `HiGHSOptimizer`:
    - Open-source (MIT) MILP alternative to Gurobi, using the standalone `highspy` binding. Formulates the same linear Big-M disjunctive MILP as `GurobiOptimizer` (binary `prec[j1,j2,m]` + continuous `b[j,m]`).
    - Uses **continuous time variables** (no `SCALE = 10`, unlike CP-SAT), so sub-unit `0.1` precision is exact with no truncation risk.
    - Enforces the 3-tier hierarchy **lexicographically with two staged solves** (identical structure to `ORTools`): stage 1 minimizes cumulative tardiness; stage 2 constrains tardiness to the stage-1 optimum and minimizes `makespan * W + sum(start)`. Falls back to the stage-1 solution if stage 2 has no incumbent, and returns `None` if stage 1 has no primal feasible/optimal solution (`HighsInfo.primal_solution_status`).
    - **Coexistence warning:** OR-Tools bundles its own (older) HiGHS, which is binary-incompatible with `highspy`. Importing/solving both in one process can segfault on macOS. `-sqc HiGHS` is unaffected (OR-Tools is only imported lazily when `ORTools` is selected), but never benchmark OR-Tools and HiGHS in the same process — the test suite runs HiGHS in a subprocess (`tests/test_highs.py`) for this reason.
  - All three backends are imported lazily inside `solve_scheduling_problem`, so heuristic-only runs do not require `gurobipy`, `ortools`, or `highspy`.

### 2.4 Deep Multi-Agent Reinforcement Learning (`src/DRL/`)
- `src/DRL/brain_sequencing.py`:
  - `Brain`: Implements a **Parameter-Sharing Double Deep Q-Network (DDQN)** across all workstation agents:
    - **Warm-up Phase (`warm_up_process`)**: Machines use `random_exploration` to populate per-machine experience buffers.
    - **State Representation (`state_direct`)**: Constructs a fixed-size $5 \times 5$ state tensor for the deciding machine: 4 rows representing candidate queuing jobs chosen by distinct dispatching criteria (`SPT`, `LWKR`, `MS`, `AVLM`) without duplication when possible, plus 1 row representing the next arriving job (enabling strategic idleness evaluation).
    - **Action Space (`action_direct`, `action_conversion`)**: Discrete action $a_t \in \{0, 1, 2, 3\}$ selecting one of the 4 candidate slots via an $\epsilon$-greedy policy.
    - **Asynchronous Experience Completion**: Because an operation takes $pt > 0$ time units to finish, transition recording is split chronologically:
      1. At decision time $t = \text{decision\_T}$, `[s_t, a_t]` is stored in `incomplete_rep_memo[m_idx][decision_T]`.
      2. Upon operation completion, `Machine.complete_experience()` observes $s_{t+1}$ and reward $r_t$, completes the tuple `[s_t, a_t, s_{t+1}, r_t]`, and appends it to `rep_memo[m_idx]`.
    - **Training (`train_Double_DQN`)**: Periodically aggregates experiences from all machines into a shared replay buffer, samples minibatches, computes DDQN targets using `action_NN` for action selection and `target_NN` for value evaluation, and updates `action_NN` with Smooth L1 (Huber) loss.

### 2.5 Offline Analysis & State Reconstruction (`src/analysis/`)
- `src/analysis/state_reconstruction.py`:
  - `ShopfloorStateReconstructor(db_path, run_id)`: Loads one run's `jobs`, `operations`, `breakdowns`, `decisions`, `schedule_cycles`, and relevant `sim_events` into memory, then replays the timeline to rebuild the **complete shopfloor condition at any time** without re-running SimPy.
  - `snapshot(t, tie="post")`: Pure-timeline reconstruction. Returns per-machine `status`, `current_job`, ordered waiting `queue` (candidate features), `breakdown`, `next_job_in_schedule`, `release_T`/`hidden_release_T`; per-in-system-job remaining route/work and accrued KPIs; plus `summary` and any `decisions` at `t`. `tie="post"` (default) treats a job starting exactly at `t` as processing; `tie="pre"` treats it as still queued.
  - `snapshot_at_decision(decision_id)` / `iter_snapshots()`: Exact pre-decision states. The deciding machine's queue is taken from the authoritative `decisions.queue_json`; every other machine is reconstructed from the timeline. Operations starting at the same timestamp `t` are classified by decision order (`decision_id`) — a machine whose decision preceded this one is already processing, one whose decision follows is still queuing — which resolves the same-timestamp ambiguity a pure time query cannot.
  - `reconstruct_state(...)`: One-shot convenience wrapper.
  - CLI: `python -m src.analysis --run_id 1 --time 77.0` / `--decision_id 42`.
  - Queue arrival order is recovered exactly from `sim_events.JOB_IN.event_id` (monotonic insertion order). Expected vs. actual values are both exposed.

---

## 3. Key Invariants to Preserve When Modifying the Code

1. **Unstarted Operations Alignment (`CentralScheduler` vs. `Machine`)**:
   - `CentralScheduler` schedules all operations that have *not yet started processing* (`job.status == "queuing"` -> `remaining_machines[:]`; `job.status == "processing"` -> `remaining_machines[1:]`).
   - Therefore, in `schedule_mode`, a `Machine` must **only pop** from `CentralScheduler.schedule[m_idx]` (`draw_from_schedule`) at the exact moment it commits to processing the job (immediately before `after_decision()` transitions `job.status` to `"processing"` with no SimPy `yield` in between). While waiting in `strategic_idle`, the machine inspects `self.next_job_in_schedule` (`schedule[m_idx][0]`) without popping.
2. **Expected vs. Actual Release/Available Times**:
   - `Machine.release_T` and `Job.available_T` represent the *expected* times visible to schedulers/agents.
   - `Machine.hidden_release_T` and `Job.actual_remaining_pt` represent the *actual* simulation times used by SimPy timeouts and non-preemptive breakdown triggers.
   - Upon operation completion (`after_operation`) or breakdown repair (`process_machine_breakdown`), both `release_T` and `hidden_release_T` (and `Job.available_T`) must be synchronized to `env.now` so early completions under stochastic settings do not leave stale future timestamps.
3. **Sub-unit Time Precision in Solvers**:
   - The simulator operates at `0.1` time-unit resolution when `random_MTBF`, `random_MTTR`, or `processing_time_variability` is active. Any integer-only solver (such as OR-Tools CP-SAT) must scale time values by `10` rather than truncating floats with `int()`.
4. **Decision Snapshot Timing (`decisions` table)**:
   - The exact queue snapshot (`queue_json`, `candidates_json`, `chosen_pos`) must be captured inside `Machine.after_decision()` **before** `Job.after_decision()` and before any later queue mutation (`after_operation` pops the chosen job). `self.queue` is in arrival order and still contains the chosen job at that point, so `queue_json[chosen_pos] == j_idx` must always hold.
5. **Offline Reconstruction Precedence (`src/analysis/`)**:
   - At a recorded decision time, the deciding machine's queue must come from the authoritative `decisions.queue_json`, not from re-deriving it from `operations`. Same-timestamp arrivals make timeline-only reconstruction ambiguous.
   - Simultaneous (same-timestamp) decisions must be ordered by `decision_id`: a job starting exactly at `t` is "processing" iff its own decision id precedes the decision being reconstructed (`snapshot_at_decision`); a pure `snapshot(t)` uses `tie` instead.
   - Queue arrival order for other machines is recovered from `sim_events.JOB_IN.event_id` (insertion order), falling back to `(arrival_t, j_idx)`. Do not reorder by `j_idx` alone.
   - Expected (`release_T`, `remaining_expected_pt`, `breakdown.expected_remaining`) and actual (`hidden_release_T`, `remaining_actual_pt`, `breakdown.actual_remaining`) values must remain distinct in reconstructed states, mirroring invariant 2.
   - An operation with `completed=0` started but never finished before the episode ended; it remains `processing` for every reconstructed time after `begin_t`, even past its recorded (expected) `end_t`.
6. **Solver Failure Is Never Dispatched**:
   - A solver must return `None` (not a partial/undefined solution) when it has no feasible incumbent (`INFEASIBLE`, `UNKNOWN`, Gurobi `SolCount == 0`, etc.).
   - `CentralScheduler` must check the returned mapping and fall back to `solve_without_optimization()` (passive schedule), recording `<STATUS>_PASSIVE_FALLBACK` in `schedule_cycles.solver_status`. Never call `convert_to_schedule()` on an empty/undefined solution.
7. **Run Lifecycle (`runs.status` / `runs.error`)**:
   - `create_run` inserts `status='running'`; `finalize_run` sets `status='completed'` and clears `error`; `fail_run` sets `status='failed'` and stores the traceback.
   - `Shopfloor.run_simulation()` must mark the run failed and re-raise on any exception after `start_run`, so a failed episode is distinguishable from an in-progress or successfully completed one.
8. **Multi-Process Safety**:
   - The parent of `SimulatorMultiThread` initializes/migrates the SQLite schema once; workers pass `skip_db_init=True`. Letting workers run `PRAGMA journal_mode = WAL` concurrently can fail with `database is locked`.
   - Workers must construct `Shopfloor` inside the child via the module-level `_run_shopfloor(kwargs)`; never pass a live `Shopfloor`/SimPy generator through `multiprocessing`.
   - Only the parent prunes log directories; workers use `prune_logs=False`.
9. **Non-Interactive Execution**:
   - `verify_simulation_setting()` must not call `input()` when `sys.stdin.isatty()` is false. Require explicit `force=True` (`-y/--force`); `force=True` bypasses the prompt unconditionally (even on a TTY). Without `force`, non-TTY stdin raises `SimulatorError` and exits non-zero.

---

## 4. Verification Before Handing Off

Run the test suite after **any** change to the simulator, schedulers, database, or reconstruction code:

```bash
uv run python -m unittest discover -s tests -t . -v
# equivalents: uv run python test.py   or   make test
```

The suite is stdlib-`unittest` only (pytest is not required). `tests/base.py` monkeypatches `src.utilities.LOG_ROOT_DIR` / `DEFAULT_DB_PATH` and exports `SIM_LOG_ROOT`, and every test uses a temporary database, so tests never pollute the project's `logs/simulation.db` or retained log folders.

What the suite enforces:

- **Project integrity:** every `src/`/`main.py`/`test.py` source compiles, core modules import, the scheduler registry/rules are complete, and required artifacts exist.
- **Rules:** concrete in-range indices; `LIFO` never returns `-1`; `Slack` subtracts remaining work.
- **Solvers:** CP-SAT lexicographic stage 1/2, `SCALE=10` precision, and `None` on `INFEASIBLE`/`UNKNOWN`; registry resolution.
- **Simulation:** all jobs complete with exactly `m_no` operations and one visit per machine; `queue_json[chosen_pos] == j_idx`; `chosen_pos` in range; candidate keys match the queue; FK integrity; scheduled starts align with actual starts under deterministic settings.
- **Failure paths:** solver fallback to passive schedule, solver exception → `runs.status='failed'` + re-raise, non-interactive stochastic guard, multi-process workers complete with distinct seeds.
- **Reproducibility/streaming:** identical seeds produce identical KPIs/operations; forcing a tiny `FLUSH_EVENTS_THRESHOLD` preserves the full `sim_events` count.
- **DB/reconstruction:** schema migration, `runs.status/error` lifecycle, `operations.completed`, and exact per-decision replay/summary counts.
- **CLI:** defaults, `--no-random_MTBF`, `-y`, `-mt/-tn`, numeric validation, and subprocess exit codes.
