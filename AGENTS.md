# Project & Architecture Guide for AI Agents (`AGENTS.md`)

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

### 2.1 Entry Point & Utilities
- `main.py`: CLI parser that maps user flags into `Simulator.run(**kwargs)`.
- `src/utilities.py`:
  - `setup_logger(stream, keep)`: Creates a timestamped directory under `logs/<YYYYMMDD-HHMMSSffffff>/sim.log`, prunes older log folders beyond `keep`, logs `DEBUG` details to `sim.log`, and streams `INFO` summaries to console.
  - `draw_gantt_chart(logger, recorder, **kwargs)`: Plots each job's operations (`recorder.j_operation_dict`) as color-coded horizontal bars and overlays machine breakdowns (`recorder.m_bkd_dict`) as hatched white bars, saving `gantt_chart.png` inside the active log folder.

### 2.2 Discrete-Event Simulator (`src/simulator/`)
Built on **SimPy** (`simpy.Environment`):
- `src/simulator/simulator.py`:
  - `Simulator` & `SimulatorMultiThread`: Single-process and multi-process wrappers around `Shopfloor`.
  - `Shopfloor`: Instantiates `simpy.Environment`, `Recorder`, `Machine` list (`m_list`), and `Narrator`. Runs `env.run(until=span + 1000)` so all jobs arriving within `[0, span]` complete their remaining operations.
- `src/simulator/event.py`:
  - `Narrator`:
    - Spawns `process_job_creation()`: Generates jobs at pre-sampled exponential intervals, registers them in `recorder.in_system_jobs`, triggers `central_scheduler.build_schedule_event` (if in optimization mode), and routes each job to its first machine.
    - Spawns `process_machine_breakdown(m, random_MTBF, random_MTTR)` for each machine: Waits for `MTBF`, waits for any in-progress operation on `m` to complete, sets `m.working_event = env.event()` and `m.status = "down"`, triggers `central_scheduler.build_schedule_event`, waits for the repair duration `bkd_t`, records `[actual_begin, actual_end]` in `recorder.m_bkd_dict[m.m_idx]`, and triggers `m.working_event.succeed()`.
    - `post_simulation()`: Verifies all jobs completed, checks for schedule-execution start-time mismatches (when `pt_cv == 0` and `random_MTTR == False`), and prints tabulated configuration and KPI summaries.
  - `Recorder`: Central state & statistics store shared across all entities (`in_system_jobs`, `j_operation_dict`, `j_tardiness_dict`, `j_flowtime_dict`, `m_bkd_dict`, decision counters `sqc_cnt_opt`, `sqc_cnt_SI`, `sqc_cnt_reactive`, `sqc_cnt_passive`).
- `src/simulator/job.py`:
  - `Job`: Dataclass holding `j_idx`, `trajectory`, `pt_by_m_idx`, `remaining_machines`, `remaining_pt` (nominal), `actual_remaining_pt` (actual), `due`, `status` (`"queuing"`, `"processing"`, `"completed"`), `available_T` (expected completion time of current operation), and `operation_record`.
  - Lifecycle hooks:
    - `after_arrival()`: Updates `arrival_T = env.now`.
    - `after_decision(m_idx, wait)`: Called when a machine starts processing the job; records `[m_idx, env.now, actual_pt, wait]`, sets `status = "processing"`, and sets `available_T = env.now + remaining_pt[0]`.
    - `after_operation()`: Called when the operation finishes; pops the completed operation from `remaining_operations`, `remaining_machines`, `remaining_pt`, `actual_remaining_pt`, refreshes `available_T = env.now`, and returns the next machine index (or `-1` after calling `completion()`).
- `src/simulator/machine.py`:
  - `Machine`: Models a workstation with `m_idx`, `queue`, `status` (`"idle"`, `"processing"`, `"strategic_idle"`, `"down"`), `release_T` (expected release time visible to scheduler), and `hidden_release_T` (actual release time).
  - Main loop `process_production()` has three phases per operation:
    - **Part I (Sequencing Decision)**:
      - **Scheduled Mode (`self.schedule_mode == True`)**: Follows the centralized schedule (`CentralScheduler`). If the next scheduled job (`self.next_job_in_schedule`) has not arrived in `self.queue` yet, the machine enters **Strategic Idleness** (`status = "strategic_idle"`) and waits on `self.required_job_in_queue_event` even if other jobs are already waiting in its queue. Once the required job is in `self.queue` and the machine is operational, it calls `draw_from_schedule(m_idx)` to pop the job from the schedule and starts processing.
      - **Reactive Mode (`len(self.queue) > 1`)**: Calls heuristic rule or DRL policy (`self.job_sequencing(jobs=self.queue)`) to pick a queue position.
      - **Passive Mode (`len(self.queue) == 1`)**: Picks the sole job at index `0`.
    - **Part II (Execution)**: Calls `after_decision()`, sets `status = "processing"`, and yields `env.timeout(actual_pt)`.
    - **Part III (Post-Operation)**: Calls `after_operation()` (which pops the job from `self.queue` and routes it to its next machine), yields `env.timeout(0)` so concurrent events at the same timestamp settle, and checks for breakdown (`working_event`) or empty queue (`process_idle()`).

### 2.3 Centralized Scheduler & Heuristics (`src/scheduler/`)
- `src/scheduler/sequencing_rule.py`:
  - `SequencingMethod`: Provides classmethods for reactive dispatching rules (`FIFO`, `LIFO`, `Slack`, `CR`, `SPT`, `LWKR`) and placeholders (`GurobiOptimizer`, `ORTools`, `DRL_scheduler`).
- `src/scheduler/scheduler.py`:
  - `CentralScheduler`:
    - Runs `solve_problem_process()` which waits on `self.build_schedule_event` (triggered upon any new job arrival or machine breakdown).
    - Extracts remaining unstarted operations for all `in_system_jobs`:
      - If `job.status == "queuing"`: all `job.remaining_machines` are unstarted.
      - If `job.status == "processing"`: `job.remaining_machines[0]` is already executing, so only `job.remaining_machines[1:]` are unstarted.
    - Checks pairwise machine intersections (`self.job_intersections`). If $\le 1$ job is in the system or no remaining trajectories share a machine, calls `solve_without_optimization()` (passive schedule). Otherwise invokes `GurobiOptimizer` or `ORTools`.
    - `convert_to_schedule(varOpBeginT)`: Sorts all remaining operations $(j, m)$ chronologically by optimized start time `varOpBeginT[j, m]` to construct `self.schedule[m_idx]` (ordered list of `j_idx`) and `self.j_op_by_schedule[j_idx]` (list of `(m_idx, expected_begin_T)`), then calls `update_machine_after_optimization()`.
  - `GurobiOptimizer`:
    - Formulates a Mixed-Integer Linear Program (MILP) using linear Big-M disjunctive constraints on binary precedence variables `varJobPrec[j1, j2, m]` and continuous start times `varOpBeginT[j, m]`.
    - Uses Gurobi's hierarchical multi-objective optimization (`setObjective` / `setObjectiveN`):
      1. **Tier 1**: Minimize cumulative tardiness $\sum_j T_j$;
      2. **Tier 2**: Minimize schedule makespan $C_{\max}$;
      3. **Tier 3**: Minimize sum of operation start times $\sum_{j, m} b_{j, m}$ (eliminating lazy delays).
  - `ORTools`:
    - Formulates a Constraint Programming (CP-SAT) model using `NewIntervalVar` and `AddNoOverlap`.
    - **Time Scaling Invariant (`SCALE = 10`)**: Because SimPy timestamps and durations can have `0.1` decimal resolution (from `random_MTBF`, `random_MTTR`, or `pt_cv`), `ORTools` scales all time values by `10` into integer tenths of a time unit inside CP-SAT and divides solved start times by `10.0`.
    - Minimizes the combined 3-tier hierarchical objective (`var_cum_tardiness * 10_000_000 + var_makespan * 10_000 + var_sum_op_begin_t`).

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
