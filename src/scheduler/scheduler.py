"""
Centralized production scheduler module.
Implements dynamic rescheduling via mathematical optimization (Gurobi MILP and Google OR-Tools CP-SAT)
Whenever a dynamic event (job arrival or machine breakdown) triggers `build_schedule_event`,
`CentralScheduler` formulates and solves the remaining Job Shop Scheduling Problem (JSSP) instance
and dispatches the resulting operation sequences to all machines on the shopfloor.
"""

# standard imports
import collections
import itertools
import json
import logging
from pathlib import Path
import time
from typing import Dict, List

from tabulate import tabulate

# NOTE: `gurobipy` and `ortools` are imported lazily inside their solver classes.
# This keeps reactive/heuristic simulations runnable in environments where only
# one (or neither) optimization backend is installed.

# project modules
from ..simulator.job import Job
from ..simulator.machine import Machine


class CentralScheduler:
    """
    Coordinates centralized mathematical optimization for dynamic scheduling.
    Maintains per-machine operation queues (`self.schedule`) and synchronizes with `Machine` instances.
    """

    def __init__(self, *args, **kwargs):
        # Map keyword arguments to instance attributes
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.schedule = {m.m_idx: [] for m in self.m_list}
        self.j_op_by_schedule = {}
        self.logger: logging.Logger

        # Reference the active jobs dictionary in the shared Recorder
        self.in_system_jobs: Dict[int, Job] = self.recorder.in_system_jobs

        # Path to log complex / over-extended scheduling problem instances (>1s solve time)
        self.ext_prob_log = {}
        self.ext_prob_log_path = Path(self.logger.handlers[0].baseFilename).parent / "over_extended_problems.json"

        # SimPy event triggered by Narrator whenever a new job arrives or a machine breaks down
        self.build_schedule_event = self.env.event()
        self.last_trigger_type = "INITIAL"

        # Select the mathematical optimization backend solver.
        # `solver_cls` can be injected explicitly (useful for tests or experiments);
        # otherwise it is resolved from the scheduling-method name via the registry below.
        self.scheduler = kwargs.get("solver_cls") or resolve_optimization_solver(self.sqc_method)
        if self.scheduler is None:
            raise ValueError(
                f"No optimization solver registered for sqc_method={self.sqc_method!r}"
            )

        # Register the continuous schedule-building SimPy process
        self.env.process(self.solve_problem_process())

    def trigger_reschedule(self, trigger_type: str) -> None:
        """Trigger `build_schedule_event` and record the cause ('JOB_ARRIVAL' or 'MACHINE_BREAKDOWN')."""
        self.last_trigger_type = trigger_type
        if not self.build_schedule_event.triggered:
            self.build_schedule_event.succeed()

    def solve_problem_process(self):
        """
        SimPy process that waits for `self.build_schedule_event`, extracts the remaining operations
        of all in-system jobs, checks for machine contention (intersections), and computes an updated schedule.
        """
        while True:
            yield self.build_schedule_event
            begin_t = time.time()
            solver_status = "PASSIVE"

            # Case 1: 0 or 1 job currently in the system -> no inter-job contention exists
            if len(self.in_system_jobs) <= 1:
                self.solve_without_optimization()
            # Case 2: Multiple jobs in the system -> extract remaining trajectories and check for contention
            else:
                self.remaining_trajectories = {}
                self.remaining_pts = {}
                self.job_intersections = {}

                # Extract remaining machine trajectories and processing times for all unfinished operations
                for j_idx, job in self.in_system_jobs.items():
                    if job.status == 'queuing':
                        self.remaining_trajectories[j_idx] = job.remaining_machines
                        self.remaining_pts[j_idx] = job.remaining_pt
                    elif len(job.remaining_machines) > 1:
                        # Exclude the current machine if the job is already undergoing processing on it
                        self.remaining_trajectories[j_idx] = job.remaining_machines[1:]
                        self.remaining_pts[j_idx] = job.remaining_pt[1:]

                # Identify pairs of jobs whose remaining trajectories share at least one common machine
                for pair in itertools.combinations(self.remaining_trajectories.keys(), 2):
                    rem_traj_1 = self.remaining_trajectories[pair[0]]
                    rem_traj_2 = self.remaining_trajectories[pair[1]]
                    intersec = list(set(rem_traj_1).intersection(rem_traj_2))
                    if intersec:
                        self.job_intersections[pair] = intersec

                # If multiple jobs exist but their remaining trajectories do not intersect, no solver is needed
                if len(self.job_intersections) == 0:
                    self.solve_without_optimization()
                else:
                    # Call the optimization solver (Gurobi or OR-Tools) to resolve machine contention
                    var_op_begin_t, over_extended_problem, solver_status = self.scheduler.solve_scheduling_problem(
                        self.logger,
                        self.env,
                        self.m_list,
                        self.job_intersections,
                        self.remaining_trajectories,
                        self.in_system_jobs,
                    )
                    if not var_op_begin_t:
                        # The solver reported infeasible / unknown / timed out without a feasible
                        # solution. NEVER feed partial or undefined variable values into the
                        # schedule: fall back to the deterministic passive schedule instead.
                        self.logger.warning(
                            "Solver {} returned no usable schedule (status={}); "
                            "falling back to the passive schedule.".format(
                                getattr(self.scheduler, "__name__", self.scheduler),
                                solver_status,
                            )
                        )
                        self.solve_without_optimization()
                        solver_status = f"{solver_status}_PASSIVE_FALLBACK"
                    else:
                        self.convert_to_schedule(var_op_begin_t)
                        # Record over-extended problem instance if solver took > 1s
                        if over_extended_problem is not None:
                            self.ext_prob_log[int(self.env.now)] = over_extended_problem

            cycle_expense = time.time() - begin_t
            if solver_status != "PASSIVE":
                self.recorder.opt_time_expense += cycle_expense

            # Record this rescheduling snapshot for the `schedule_cycles` SQLite table
            num_ops = sum(len(ops) for ops in self.schedule.values())
            m_sch_serializable = {int(k): [int(x) for x in v] for k, v in self.schedule.items()}
            j_sch_serializable = {
                int(k): [[int(m_id), float(t)] for m_id, t in v]
                for k, v in self.j_op_by_schedule.items()
            }
            self.recorder.schedule_cycles_list.append(
                (
                    round(float(self.env.now), 2),
                    str(self.last_trigger_type),
                    int(len(self.in_system_jobs)),
                    int(num_ops),
                    str(solver_status),
                    round(float(cycle_expense), 5),
                    json.dumps(m_sch_serializable),
                    json.dumps(j_sch_serializable),
                )
            )
            self.recorder.record_event(
                self.env.now,
                "DEBUG",
                "RESCHEDULE",
                message=f"trigger={self.last_trigger_type}, status={solver_status}, jobs={len(self.in_system_jobs)}, ops={num_ops}",
            )

            # Reset `build_schedule_event` so future dynamic events can trigger the next rescheduling cycle
            self.build_schedule_event = self.env.event()

    def solve_without_optimization(self) -> None:
        """
        Build a direct (passive) schedule when there is no machine contention between remaining jobs.
        """
        # Reset all machines' schedules first so stale entries from completed jobs are cleared
        self.schedule = {m.m_idx: [] for m in self.m_list}
        self.j_op_by_schedule = {}

        for j_idx, j_object in self.in_system_jobs.items():
            # Exclude the currently executing operation if the job is already being processed
            if j_object.status == "queuing":
                j_traj = j_object.remaining_machines
                j_pts = j_object.remaining_pt
            elif len(j_object.remaining_machines) > 1:
                j_traj = j_object.remaining_machines[1:]
                j_pts = j_object.remaining_pt[1:]
            else:
                continue

            self.j_op_by_schedule[j_idx] = []
            cur_t = max(j_object.available_T, self.env.now)
            for m_idx, pt in zip(j_traj, j_pts):
                self.schedule[m_idx].append(j_idx)
                op_begin_t = max(cur_t, self.m_list[m_idx].release_T, self.env.now)
                self.j_op_by_schedule[j_idx].append((int(m_idx), round(float(op_begin_t), 1)))
                cur_t = op_begin_t + pt

        # Log the passive schedule if any machine has scheduled operations
        if any(sch for sch in self.schedule.values()):
            self.logger.info(
                "{} > Passive schedule: \n{}".format(
                    self.env.now,
                    tabulate(
                        [["Machine"] + list(self.schedule.keys()), ["Schedule"] + list(self.schedule.values())],
                        headers="firstrow",
                        tablefmt="psql",
                    ),
                )
            )

        self.update_machine_after_optimization()

    def convert_to_schedule(self, varOpBeginT: dict) -> None:
        """
        Convert optimized operation start times `varOpBeginT[(j_idx, m_idx)]` into ordered
        job sequences per machine and expected operation start times per job.
        """
        # Reset per-machine schedule and per-job expected operation start times
        self.schedule = {m.m_idx: [] for m in self.m_list}
        self.j_op_by_schedule = {j_idx: [] for j_idx in self.remaining_trajectories.keys()}

        # Sort operations chronologically by their optimized begin time
        reordered_var_op_begin_t = sorted(varOpBeginT.items(), key=lambda item: (item[1], item[0][0]))
        for (j_idx, m_idx), begin_t in reordered_var_op_begin_t:
            # Append job index to the corresponding machine's schedule sequence
            self.schedule[m_idx].append(j_idx)
            # Record expected operation start time for schedule-execution verification
            self.j_op_by_schedule[j_idx].append((int(m_idx), round(float(begin_t), 1)))

        # Log the machines' job sequences
        self.logger.debug(
            "Machines' sequence in new schedule: \n{}".format(
                tabulate(
                    [["M.idx", "Job sequence [j_idx]"], *self.schedule.items()],
                    headers="firstrow",
                    tablefmt="psql",
                )
            )
        )
        # Log the jobs' scheduled operations and expected start times
        self.logger.debug(
            "Jobs' operations in new schedule: \n{}".format(
                tabulate(
                    [
                        ["J.idx", "Remaining Operations (m_idx, opBeginT)"],
                        *[[j_idx, op] for j_idx, op in self.j_op_by_schedule.items()],
                    ],
                    headers="firstrow",
                    tablefmt="psql",
                )
            )
        )
        self.update_machine_after_optimization()

    def update_machine_after_optimization(self) -> None:
        """
        Synchronize each machine's `next_job_in_schedule` pointer with the newly built `self.schedule`
        and wake up any machine in strategic idleness whose newly scheduled job is already in its queue.
        """
        for m in self.m_list:
            if not self.schedule[m.m_idx]:
                m.next_job_in_schedule = -1
                # Wake any machine blocked in strategic idleness: with an empty schedule it
                # must re-evaluate (and fall back to its queue) instead of waiting forever
                # for a job index of -1 that can never arrive.
                m.notify_schedule_update()
                continue
            m.next_job_in_schedule = self.schedule[m.m_idx][0]
            m.update_status_after_new_schedule()

    def draw_from_schedule(self, m_idx: int) -> int:
        """
        Called by Machine `m_idx` at the moment it commits to processing its next scheduled job.
        Pops and returns the head job index from `self.schedule[m_idx]` and advances `next_job_in_schedule`.
        """
        self.logger.debug(
            "Draw from schedule, Machine {}, current schedule {}, queue {}".format(
                m_idx, self.schedule[m_idx], [j.j_idx for j in self.m_list[m_idx].queue]
            )
        )
        next_job_in_schedule = self.schedule[m_idx].pop(0)
        self.m_list[m_idx].next_job_in_schedule = (
            self.schedule[m_idx][0] if self.schedule[m_idx] else -1
        )
        return next_job_in_schedule

    def post_simulation(self) -> None:
        """Write any over-extended scheduling problem instances (>1s solve time) to a JSON file."""
        if self.ext_prob_log:
            print(
                "{} over-extended scheduling problem is recorded, saved to {}".format(
                    len(self.ext_prob_log), self.ext_prob_log_path
                )
            )
        with open(self.ext_prob_log_path, "w") as f:
            json.dump(self.ext_prob_log, f)


class ORTools:
    """
    Google OR-Tools CP-SAT Constraint Programming solver for the dynamic Job Shop Scheduling Problem.
    Note: Since CP-SAT requires integer variables while the simulator uses 0.1-unit time precision
    (e.g., for random MTBF/MTTR or PT variability), all time parameters are scaled by `SCALE = 10`
    inside the CP model and scaled back by `1 / SCALE` when extracting the solution.
    """

    SCALE = 10

    @classmethod
    def solve_scheduling_problem(
        cls,
        logger: logging.Logger,
        env,
        m_list: List[Machine],
        job_intersections: dict,
        remaining_trajectories: Dict[int, list],
        in_system_jobs: Dict[int, Job],
    ):
        # Lazy import: heuristic-only simulations must not require OR-Tools to be installed.
        from ortools.sat.python import cp_model

        start_t = time.time()
        scale = cls.SCALE

        # Extract machines' expected release times and jobs' expected available times (scaled to integer tenths)
        machine_release_t = {
            m.m_idx: int(round(max(m.release_T, env.now) * scale)) for m in m_list
        }
        job_available_t = {
            j_idx: int(round(max(in_system_jobs[j_idx].available_T, env.now) * scale))
            for j_idx in remaining_trajectories.keys()
        }

        # Build the OR-Tools CP-SAT model
        model = cp_model.CpModel()

        """
        PART I: Create decision variables and interval pairings
        """
        now_scaled = int(round(env.now * scale))
        total_rem_pt_scaled = int(
            round(sum(sum(j.remaining_pt) for j in in_system_jobs.values()) * scale)
        )
        max_release_scaled = max(
            list(machine_release_t.values()) + list(job_available_t.values()) + [now_scaled]
        )
        ul = max_release_scaled + total_rem_pt_scaled + 100 * scale

        all_jobs, all_ops = {}, {}
        m_to_ops = {m.m_idx: [] for m in m_list}
        job_tuple = collections.namedtuple("job", ['completion', 'discrepancy', 'tardiness'])
        op_tuple = collections.namedtuple("operation", ['begin', 'end', 'interval'])

        # 1. Job / Operation / Machine interval decision variables
        for j_idx, traj in remaining_trajectories.items():
            for m_idx in traj:
                suffix = f"_j{j_idx}_m{m_idx}"
                duration_scaled = int(round(in_system_jobs[j_idx].pt_by_m_idx[m_idx] * scale))
                var_op_begin_t = model.NewIntVar(now_scaled, ul, "varOpBeginT" + suffix)
                var_op_end_t = model.NewIntVar(now_scaled, ul, "varOpEndT" + suffix)
                var_op_interval = model.NewIntervalVar(
                    var_op_begin_t, duration_scaled, var_op_end_t, "varOpInterval" + suffix
                )
                all_ops[j_idx, m_idx] = op_tuple(
                    begin=var_op_begin_t, end=var_op_end_t, interval=var_op_interval
                )
                m_to_ops[m_idx].append(var_op_interval)

            # 2. Job completion discrepancy (earliness/tardiness) and non-negative tardiness variables
            var_job_discrepancy = model.NewIntVar(-ul, ul, f"varJobDiscrepancy_{j_idx}")
            var_job_tardiness = model.NewIntVar(0, ul, f"varJobTardiness_{j_idx}")
            all_jobs[j_idx] = job_tuple(
                completion=var_op_end_t,
                discrepancy=var_job_discrepancy,
                tardiness=var_job_tardiness,
            )

        # 3. System-level performance variables: schedule makespan and cumulative tardiness
        var_makespan = model.NewIntVar(now_scaled, ul, "varMakespan")
        var_cum_tardiness = cp_model.LinearExpr.Sum([j.tardiness for j in all_jobs.values()])
        var_sum_op_begin_t = cp_model.LinearExpr.Sum([op.begin for op in all_ops.values()])

        """
        PART II: Specify scheduling constraints
        """
        model_spec = {
            'op_sqc': 0,
            'op_overlap': 0,
            'J_release': 0,
            'M_release': 0,
            'J_discrepency': 0,
            'J_tardiness': 0,
        }
        for j_idx, traj in remaining_trajectories.items():
            # 1. Precedence constraints: a job's operations must follow its routing trajectory order
            for op_sqc in range(len(traj) - 1):
                model.Add(all_ops[j_idx, traj[op_sqc + 1]].begin >= all_ops[j_idx, traj[op_sqc]].end)
                model_spec['op_sqc'] += 1

            # 2. Release time constraints: operations cannot start before the machine or job is available
            # 2.1 Machine release constraint
            for m_idx in traj:
                model.Add(all_ops[j_idx, m_idx].begin >= machine_release_t[m_idx])
                model_spec['M_release'] += 1
            # 2.2 Job availability constraint on its first remaining operation
            model.Add(all_ops[j_idx, traj[0]].begin >= job_available_t[j_idx])
            model_spec['J_release'] += 1

            # 2.3 Completion time discrepancy relative to due date
            due_scaled = int(round(in_system_jobs[j_idx].due * scale))
            model.Add(all_jobs[j_idx].discrepancy == all_jobs[j_idx].completion - due_scaled)
            # 2.4 Tardiness = max(0, discrepancy)
            model.AddMaxEquality(all_jobs[j_idx].tardiness, [0, all_jobs[j_idx].discrepancy])

        # 3. Disjunctive (no-overlap) constraints: each machine processes at most one job at a time
        for machine in m_to_ops:
            model.AddNoOverlap(m_to_ops[machine])
            model_spec['op_overlap'] += len(m_to_ops[machine])

        # 4. Makespan definition: maximum completion time across all scheduled jobs
        model.AddMaxEquality(
            var_makespan,
            [all_ops[j_idx, traj[-1]].end for j_idx, traj in remaining_trajectories.items()],
        )

        """
        PART III: Hierarchical optimization with a TRUE lexicographic order.

        Tier 1 (Primary):   minimize cumulative tardiness (`var_cum_tardiness`).
        Tier 2 (Secondary): minimize schedule makespan (`var_makespan`).
        Tier 3 (Tertiary):  minimize sum of operation start times (`var_sum_op_begin_t`).

        CP-SAT exposes a single scalar objective, so the previous implementation used
        fixed weights (1e7 / 1e4 / 1). Those weights do NOT enforce a hierarchy: with
        scaled time, one unit of tardiness could be traded for 1000 units of makespan,
        and the tertiary term could outweigh several units of tardiness. Instead, solve
        in two stages:

          1. minimize tardiness and keep its optimal (or best feasible) value;
          2. constrain tardiness to that value and minimize
             `makespan * W + sum(start)`, where `W > max_possible_sum(start)` makes
             makespan strictly dominate the tertiary term.
        """
        logger.debug(
            'Problem spec.: [{} Jobs], [{} Ops]. Constraints: [{} op_sqc]; [{} op_overlap]; '
            '[{} M_release]; [{} J_release/discrepency/tardiness]'.format(
                len(in_system_jobs),
                sum(len(x) for x in remaining_trajectories.values()),
                model_spec['op_sqc'],
                model_spec['op_overlap'],
                model_spec['M_release'],
                model_spec['J_release'],
            )
        )

        # ---- Stage 1: minimize cumulative tardiness -------------------------
        model.Minimize(var_cum_tardiness)
        solver_t1 = cp_model.CpSolver()
        solver_t1.parameters.max_time_in_seconds = 10.0
        status_t1 = solver_t1.Solve(model)
        status_t1_name = solver_t1.StatusName(status_t1)

        if status_t1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            time_expense = round(time.time() - start_t, 3)
            logger.warning(
                "OR-Tools found no feasible schedule (status={}, {:.3f}s); "
                "the passive schedule will be used.".format(status_t1_name, time_expense)
            )
            return None, None, status_t1_name

        opt_tardiness = int(round(solver_t1.ObjectiveValue()))

        # ---- Stage 2: keep Tier 1 optimal, then optimize makespan + sum(start)
        # `max_sum_op_begin` bounds the largest possible value of `var_sum_op_begin_t`
        # (each of the `len(all_ops)` start variables is in [now_scaled, ul]).
        max_sum_op_begin = max(1, len(all_ops) * ul)
        model.Add(var_cum_tardiness <= opt_tardiness)
        model.ClearObjective()
        model.Minimize(var_makespan * (max_sum_op_begin + 1) + var_sum_op_begin_t)

        solver_t2 = cp_model.CpSolver()
        solver_t2.parameters.max_time_in_seconds = 5.0
        status_t2 = solver_t2.Solve(model)
        status_t2_name = solver_t2.StatusName(status_t2)

        if status_t2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            solver = solver_t2
            status_name = status_t2_name
        else:
            # The Stage-1 solution is still feasible and satisfies the Tier-1 bound;
            # reuse it rather than dispatching an undefined / partial solution.
            solver = solver_t1
            status_name = f"{status_t1_name}_T2_{status_t2_name}"

        """
        PART IV: Extract solution and convert scaled integers back to simulation time units
        """
        time_expense = round(time.time() - start_t, 3)
        logger.debug(
            "OR_Tools CP Model solving process elapsed, model status: {}, time expense: {}s".format(
                status_name, time_expense
            )
        )

        # Record problem details if solving took longer than 1 second
        if time_expense > 1:
            over_extended_problem = {
                'Jobs': {
                    j_idx: {
                        "ops": [
                            (float(m_idx), float(in_system_jobs[j_idx].pt_by_m_idx[m_idx]))
                            for m_idx in traj
                        ],
                        "avail": job_available_t[j_idx] / scale,
                    }
                    for j_idx, traj in remaining_trajectories.items()
                },
                'Machines': {m_idx: release_t / scale for m_idx, release_t in machine_release_t.items()},
                'Expense': time_expense,
            }
        else:
            over_extended_problem = None

        converted_var_op_begin_t = {
            key: round(solver.Value(op.begin) / scale, 1) for key, op in all_ops.items()
        }
        return converted_var_op_begin_t, over_extended_problem, status_name


class GurobiOptimizer:
    """
    Gurobi Mixed-Integer Linear Programming (MILP) solver for the dynamic Job Shop Scheduling Problem.
    Uses hierarchical multi-objective optimization:
      1. Minimize total tardiness;
      2. Minimize schedule makespan;
      3. Minimize sum of operation start times.
    """

    @classmethod
    def solve_scheduling_problem(
        cls,
        logger: logging.Logger,
        env,
        m_list: List[Machine],
        job_intersections: dict,
        remaining_trajectories: Dict[int, list],
        in_system_jobs: Dict[int, Job],
    ):
        # Lazy import: heuristic-only simulations must not require Gurobi to be installed.
        import gurobipy as gp
        from gurobipy import GRB

        grb_msg = {
            2: 'optimal',
            3: 'infeasible',
            4: 'infeasible or unbounded',
            5: 'unbounded',
            9: 'time limit',
            11: 'interrupted',
            12: 'numeric',
            13: 'suboptimal',
        }
        start_t = time.time()

        # Extract machines' expected release times and jobs' expected available times
        machine_release_t = {m.m_idx: max(m.release_T, env.now) for m in m_list}
        job_available_t = {
            j_idx: max(in_system_jobs[j_idx].available_T, env.now)
            for j_idx in remaining_trajectories.keys()
        }

        # Compute a valid Big-M constant for disjunctive machine precedence constraints
        total_rem_pt = sum(sum(j.remaining_pt) for j in in_system_jobs.values())
        max_release = max(
            list(machine_release_t.values()) + list(job_available_t.values()) + [env.now]
        )
        big_m = float(max_release + total_rem_pt + 1000.0)

        # Build the Gurobi optimization model
        with gp.Env(empty=True) as grb_env:
            grb_env.setParam('LogToConsole', 0)
            grb_env.setParam('LogFile', str(Path(logger.handlers[0].baseFilename).parent / "gurobi.log"))
            grb_env.start()
            with gp.Model(name="jsp_scheduler", env=grb_env) as model:
                """
                PART I: Create decision variables and index pairings
                """
                pair_op_begin_t = []
                pair_op_sqc = []
                pair_job_first_op = []
                pair_job_last_op = []
                for j_idx, traj in remaining_trajectories.items():
                    # (j_idx, m_idx) pairs for operation start time variables
                    pair_op_begin_t += list(itertools.product([j_idx], traj))
                    # Pairs of consecutive machines in a job's trajectory
                    pair_consec_m = list(zip(traj, traj[1:]))
                    # [(j_idx, m1_idx), (j_idx, m2_idx)] pairs for intra-job operation sequence constraints
                    pair_op_sqc += [list(itertools.product([j_idx], mp)) for mp in pair_consec_m]
                    # First and last remaining operations of each job
                    pair_job_first_op += [(j_idx, traj[0])]
                    pair_job_last_op += [(j_idx, traj[-1])]

                # 1. Continuous variables indicating the start time of each remaining operation
                var_op_begin_t = model.addVars(pair_op_begin_t, vtype=GRB.CONTINUOUS, name="varOpBeginT")

                # 2. Job-level performance variables: completion time, due-date discrepancy, and tardiness
                job_keys = list(remaining_trajectories.keys())
                var_job_comp_t = model.addVars(job_keys, vtype=GRB.CONTINUOUS, name="varJobCompT")
                var_job_comp_discr = model.addVars(
                    job_keys, lb=-big_m, vtype=GRB.CONTINUOUS, name="varJobCompDiscr"
                )
                var_job_tardiness = model.addVars(job_keys, vtype=GRB.CONTINUOUS, name="varJobTardiness")
                var_makespan = model.addVar(vtype=GRB.CONTINUOUS, name="varMakespan")

                # 3. Binary precedence variables for each pair of jobs sharing a machine:
                # varJobPrec[j1, j2, m] == 0 if job j1 precedes job j2 on machine m; 1 otherwise
                pair_job_prec = []
                for (j1, j2), intersec in job_intersections.items():
                    pair_job_prec += list(itertools.product([j1], [j2], intersec))
                var_job_prec = model.addVars(pair_job_prec, vtype=GRB.BINARY, name='varJobPrec')

                logger.debug(
                    'Job in system: {}, Operation begin time pairs: {}, '
                    'Job operations sequence pairs: {}, Job precedence pairs: {}'.format(
                        len(in_system_jobs),
                        len(pair_op_begin_t),
                        len(pair_op_sqc),
                        len(pair_job_prec),
                    )
                )

                """
                PART II: Create scheduling constraints
                """
                # 1. Intra-job precedence: operations of a job must follow its routing trajectory
                model.addConstrs(
                    (
                        var_op_begin_t[j, m1] + in_system_jobs[j].pt_by_m_idx[m1] <= var_op_begin_t[j, m2]
                        for (j, m1), (j, m2) in pair_op_sqc
                    ),
                    name='constrOpSqc',
                )

                # 2. Release time constraints
                # 2.1 Machine release time constraint
                model.addConstrs(
                    (var_op_begin_t[j, m] >= machine_release_t[m] for j, m in pair_op_begin_t),
                    name='constrMachineRelease',
                )
                # 2.2 Job availability constraint on its first remaining operation
                model.addConstrs(
                    (var_op_begin_t[j, m] >= job_available_t[j] for j, m in pair_job_first_op),
                    name='constrJobAvailable',
                )

                # 3. Disjunctive machine precedence constraints (linear Big-M formulation):
                # Note: Using linear Big-M instead of bilinear `(varOpBeginT + pt) * (1 - varJobPrec)`
                # keeps the model a pure MILP rather than a non-convex MIQCP.
                # 3.1 If job 1 precedes job 2 (varJobPrec == 0): b[j1, m] + pt[j1, m] <= b[j2, m]
                model.addConstrs(
                    (
                        var_op_begin_t[j1, m]
                        + in_system_jobs[j1].pt_by_m_idx[m]
                        - big_m * var_job_prec[j1, j2, m]
                        <= var_op_begin_t[j2, m]
                        for j1, j2, m in pair_job_prec
                    ),
                    name='constrJobPrec_0',
                )
                # 3.2 If job 2 precedes job 1 (varJobPrec == 1): b[j2, m] + pt[j2, m] <= b[j1, m]
                model.addConstrs(
                    (
                        var_op_begin_t[j2, m]
                        + in_system_jobs[j2].pt_by_m_idx[m]
                        - big_m * (1 - var_job_prec[j1, j2, m])
                        <= var_op_begin_t[j1, m]
                        for j1, j2, m in pair_job_prec
                    ),
                    name='constrJobPrec_1',
                )

                # 4. Performance metric definitions
                # 4.1 Job completion time
                model.addConstrs(
                    (
                        var_job_comp_t[j] == var_op_begin_t[j, m] + in_system_jobs[j].pt_by_m_idx[m]
                        for j, m in pair_job_last_op
                    ),
                    name='constrJobCompT',
                )
                # 4.2 Job completion discrepancy (earliness / tardiness)
                model.addConstrs(
                    (
                        var_job_comp_discr[j] == var_job_comp_t[j] - in_system_jobs[j].due
                        for j, m in pair_job_last_op
                    ),
                    name='constrJobCompDiscr',
                )
                # 4.3 Job tardiness = max(0, discrepancy)
                model.addConstrs(
                    (
                        var_job_tardiness[j] == gp.max_(var_job_comp_discr[j], constant=0)
                        for j, m in pair_job_last_op
                    ),
                    name='constrJobTardiness',
                )
                # 4.4 Schedule makespan in this rescheduling cycle
                model.addConstr(
                    var_makespan == gp.max_([var_job_comp_t[j] for j, m in pair_job_last_op]),
                    name='constrMakespan',
                )

                """
                PART III: Configure hierarchical objectives and run optimization
                """
                model.update()
                model.setParam('TimeLimit', 10)
                # Tier 1 (Primary) objective: minimize cumulative tardiness
                model.setObjective(var_job_tardiness.sum(), GRB.MINIMIZE)
                # Tier 2 (Secondary) objective: minimize schedule makespan without degrading Tier 1
                model.setObjectiveN(expr=var_makespan, index=1, priority=-1)
                # Tier 3 (Tertiary) objective: start each operation as early as possible
                model.setObjectiveN(expr=var_op_begin_t.sum(), index=2, priority=-2)

                model.optimize()

                """
                PART IV: Extract solution into standard Python dictionary
                """
                time_expense = round(time.time() - start_t, 3)
                status_name = grb_msg.get(model.status, f"status_{model.status}").upper()
                logger.debug(
                    "Optimization elapsed, model status: {}, time expense: {}s".format(
                        status_name, time_expense
                    )
                )

                # `var.X` is undefined when Gurobi has no feasible solution (infeasible,
                # unbounded, or a time limit reached before the first incumbent). Guard
                # before extraction so the scheduler can fall back to the passive schedule.
                if model.SolCount == 0:
                    logger.warning(
                        "Gurobi found no feasible schedule (status={}, {:.3f}s); "
                        "the passive schedule will be used.".format(status_name, time_expense)
                    )
                    return None, None, status_name

                converted_var_op_begin_t = {key: round(var.X, 1) for key, var in var_op_begin_t.items()}

        # Record problem instance if solve time exceeded 1 second
        if time_expense > 1:
            over_extended_problem = [
                [(float(m_idx), float(in_system_jobs[j_idx].pt_by_m_idx[m_idx])) for m_idx in traj]
                for j_idx, traj in remaining_trajectories.items()
            ]
        else:
            over_extended_problem = None

        return converted_var_op_begin_t, over_extended_problem, status_name


class HiGHSOptimizer:
    """
    HiGHS Mixed-Integer Linear Programming (MILP) solver (open-source, MIT license)
    for the dynamic Job Shop Scheduling Problem. Formulates the same linear Big-M
    disjunctive MILP as `GurobiOptimizer`, but solves it with the free HiGHS MIP solver
    so no commercial license is required.

    .. note:: OR-Tools bundles its own (older) HiGHS; the standalone ``highspy`` build
        is binary-incompatible with it, and importing/solving both in one process can
        segfault on macOS. A ``-sqc HiGHS`` run is unaffected (OR-Tools is only imported
        lazily when ``ORTools`` is selected), but do not benchmark OR-Tools and HiGHS in
        the same process (the test suite runs HiGHS in a subprocess for this reason).

    Unlike `ORTools` (integer-only CP-SAT, which scales time by `SCALE = 10`), HiGHS
    uses continuous time variables directly, so sub-unit (0.1) time precision needs no
    scaling and there is no truncation risk.

    The 3-tier hierarchical objective is enforced lexicographically with two staged
    solves (mirroring `ORTools`):
      1. minimize cumulative tardiness;
      2. constrain tardiness to the stage-1 optimum and minimize
         ``makespan * W + sum(start)`` where ``W > max_possible_sum(start)`` makes
         makespan strictly dominate the tertiary term.
    """

    @classmethod
    def solve_scheduling_problem(
        cls,
        logger: logging.Logger,
        env,
        m_list: List[Machine],
        job_intersections: dict,
        remaining_trajectories: Dict[int, list],
        in_system_jobs: Dict[int, Job],
    ):
        # Lazy import: heuristic/CP-SAT-only simulations must not require HiGHS.
        import highspy

        start_t = time.time()
        inf = highspy.kHighsInf
        feasible_status = int(highspy.SolutionStatus.kSolutionStatusFeasible)

        # Expected machine release / job availability times (visible to the scheduler).
        machine_release_t = {m.m_idx: float(max(m.release_T, env.now)) for m in m_list}
        job_available_t = {
            j_idx: float(max(in_system_jobs[j_idx].available_T, env.now))
            for j_idx in remaining_trajectories.keys()
        }

        # Valid Big-M constant for the disjunctive machine precedence constraints.
        total_rem_pt = sum(sum(j.remaining_pt) for j in in_system_jobs.values())
        max_release = max(
            list(machine_release_t.values()) + list(job_available_t.values()) + [env.now]
        )
        big_m = float(max_release + total_rem_pt + 1000.0)

        # Disjunctive precedence pairs: (j1, j2, m) for each machine shared by two jobs.
        pair_job_prec = []
        for (j1, j2), intersec in job_intersections.items():
            pair_job_prec += list(itertools.product([j1], [j2], intersec))

        # ----------------------------------------------------------------- columns
        # Register each decision variable as a column index and record its bounds +
        # integrality. Helper closures keep the index bookkeeping in one place.
        start_col, comp_col, discr_col, tard_col, prec_col = {}, {}, {}, {}, {}
        col_lb, col_ub, col_integrality = [], [], []

        def add_col(lb, ub, integrality):
            idx = len(col_lb)
            col_lb.append(lb)
            col_ub.append(ub)
            col_integrality.append(integrality)
            return idx

        # Start time of every remaining operation b[j, m] (continuous, >= earliest
        # possible start, <= big_m which is a safe upper bound on any schedule length).
        for j_idx, traj in remaining_trajectories.items():
            for m_idx in traj:
                lb = max(machine_release_t[m_idx], job_available_t[j_idx], env.now)
                start_col[(j_idx, m_idx)] = add_col(
                    float(lb), big_m, highspy.HighsVarType.kContinuous
                )
        # Job-level timing/objective variables: completion C[j], due-date discrepancy
        # D[j] = C[j] - due[j], and non-negative tardiness T[j] = max(0, D[j]).
        for j_idx in remaining_trajectories.keys():
            comp_col[j_idx] = add_col(0.0, big_m, highspy.HighsVarType.kContinuous)
            discr_col[j_idx] = add_col(-big_m, big_m, highspy.HighsVarType.kContinuous)
            tard_col[j_idx] = add_col(0.0, big_m, highspy.HighsVarType.kContinuous)
        # System-level makespan and the binary job-ordering precedence variables.
        cmax_col = add_col(0.0, big_m, highspy.HighsVarType.kContinuous)
        for (j1, j2, m) in pair_job_prec:
            prec_col[(j1, j2, m)] = add_col(0.0, 1.0, highspy.HighsVarType.kInteger)

        # -------------------------------------------------------------------- rows
        # Each row is a linear constraint (lower_bound, upper_bound, {col: coeff});
        # an equality is just lower == upper.
        rows = []

        def add_row(lb, ub, coeffs):
            rows.append((lb, ub, coeffs))

        # 1. Intra-job precedence: b[j, m2] - b[j, m1] >= pt[j, m1]
        for j_idx, traj in remaining_trajectories.items():
            for m1, m2 in zip(traj, traj[1:]):
                pt = float(in_system_jobs[j_idx].pt_by_m_idx[m1])
                add_row(pt, inf, {start_col[(j_idx, m2)]: 1.0, start_col[(j_idx, m1)]: -1.0})
        # 2. Machine release: b[j, m] >= machine_release_t[m]
        for j_idx, traj in remaining_trajectories.items():
            for m_idx in traj:
                add_row(machine_release_t[m_idx], inf, {start_col[(j_idx, m_idx)]: 1.0})
        # 3. Job availability on its first remaining operation
        for j_idx, traj in remaining_trajectories.items():
            add_row(job_available_t[j_idx], inf, {start_col[(j_idx, traj[0])]: 1.0})
        # 4. Disjunctive Big-M constraints (linear, no bilinear terms):
        #    - prec == 0 -> job j1 precedes j2 on machine m
        #    - prec == 1 -> job j2 precedes j1 on machine m
        for (j1, j2, m) in pair_job_prec:
            p = prec_col[(j1, j2, m)]
            pt1 = float(in_system_jobs[j1].pt_by_m_idx[m])
            pt2 = float(in_system_jobs[j2].pt_by_m_idx[m])
            # b[j1,m] - b[j2,m] - big_m * prec <= -pt1
            add_row(-inf, -pt1, {start_col[(j1, m)]: 1.0, start_col[(j2, m)]: -1.0, p: -big_m})
            # b[j2,m] - b[j1,m] + big_m * prec <= big_m - pt2
            add_row(-inf, big_m - pt2, {start_col[(j2, m)]: 1.0, start_col[(j1, m)]: -1.0, p: big_m})
        # 5. Job completion: C[j] - b[j, m_last] == pt[j, m_last]
        for j_idx, traj in remaining_trajectories.items():
            m_last = traj[-1]
            pt = float(in_system_jobs[j_idx].pt_by_m_idx[m_last])
            add_row(pt, pt, {comp_col[j_idx]: 1.0, start_col[(j_idx, m_last)]: -1.0})
        # 6. Completion discrepancy: D[j] - C[j] == -due[j]
        for j_idx in remaining_trajectories.keys():
            add_row(-float(in_system_jobs[j_idx].due), -float(in_system_jobs[j_idx].due),
                    {discr_col[j_idx]: 1.0, comp_col[j_idx]: -1.0})
        # 7. Tardiness >= discrepancy: T[j] - D[j] >= 0 (T[j] >= 0 via its lower bound)
        for j_idx in remaining_trajectories.keys():
            add_row(0.0, inf, {tard_col[j_idx]: 1.0, discr_col[j_idx]: -1.0})
        # 8. Makespan >= every job completion time
        for j_idx in remaining_trajectories.keys():
            add_row(0.0, inf, {cmax_col: 1.0, comp_col[j_idx]: -1.0})

        # ---------------------------------------------------- LP assembly + solve
        # `solve` builds the LP, runs HiGHS, and returns (solution, info, status_name).
        # `solution` is None when HiGHS has no primal feasible/optimal solution (so the
        # caller can fall back to the passive schedule instead of reading garbage).
        def solve(rows_used, obj_cost, time_limit):
            lp = highspy.HighsLp()
            lp.num_col_ = len(col_lb)
            lp.num_row_ = len(rows_used)
            lp.col_cost_ = [obj_cost.get(c, 0.0) for c in range(len(col_lb))]
            lp.col_lower_ = col_lb
            lp.col_upper_ = col_ub
            lp.row_lower_ = [r[0] for r in rows_used]
            lp.row_upper_ = [r[1] for r in rows_used]
            # Assemble the column-major sparse coefficient matrix. `rows` is a list of
            # {col_index: coefficient} dicts (row-major); HiGHS wants a CSC-style
            # (start_/index_/value_) layout, so invert to {column: [(row, coeff)]}.
            col_entries = [[] for _ in range(len(col_lb))]
            for r_i, (_, _, coeffs) in enumerate(rows_used):
                for c, coef in coeffs.items():
                    if coef:
                        col_entries[c].append((r_i, coef))
            starts, index, value = [0], [], []
            for c in range(len(col_lb)):
                for r_i, coef in col_entries[c]:
                    index.append(r_i)
                    value.append(coef)
                starts.append(starts[-1] + len(col_entries[c]))
            lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
            lp.a_matrix_.start_ = starts
            lp.a_matrix_.index_ = index
            lp.a_matrix_.value_ = value
            lp.sense_ = highspy.ObjSense.kMinimize
            lp.integrality_ = col_integrality

            h = highspy.Highs()
            h.setOptionValue("output_flag", False)
            h.setOptionValue("time_limit", time_limit)
            h.passModel(lp)
            h.run()
            info = h.getInfo()
            status_name = h.modelStatusToString(h.getModelStatus())
            if int(info.primal_solution_status) != feasible_status:
                return None, None, status_name
            return h.getSolution(), info, status_name

        # ---- Stage 1: minimize cumulative tardiness ---------------------------
        stage1_obj = {tard_col[j_idx]: 1.0 for j_idx in remaining_trajectories.keys()}
        sol1, info1, status1 = solve(rows, stage1_obj, 10.0)
        if sol1 is None:
            time_expense = round(time.time() - start_t, 3)
            logger.warning(
                "HiGHS found no feasible schedule (status={}, {:.3f}s); "
                "the passive schedule will be used.".format(status1, time_expense)
            )
            return None, None, status1

        opt_tardiness = float(info1.objective_function_value)

        # ---- Stage 2: keep Tier-1 optimal, minimize makespan*W + sum(start) ---
        # `W` strictly dominates the tertiary term so makespan is minimized first:
        # any makespan change of 1 outweighs the entire possible range of sum(start).
        max_sum_start = max(1.0, float(len(start_col) * big_m))
        w = max_sum_start + 1.0
        # Constrain cumulative tardiness to the stage-1 optimum. The `+1e-4` slack
        # absorbs the solver's floating-point tolerances so the bound can never make
        # a previously-optimal stage-1 solution appear infeasible in stage 2.
        tard_sum = {tard_col[j_idx]: 1.0 for j_idx in remaining_trajectories.keys()}
        tard_bound_row = (-inf, opt_tardiness + 1e-4, tard_sum)
        stage2_obj = {cmax_col: w}
        for col in start_col.values():
            stage2_obj[col] = 1.0

        sol2, info2, status2 = solve(rows + [tard_bound_row], stage2_obj, 5.0)
        if sol2 is None:
            # The stage-1 solution is still feasible and Tier-1 optimal; reuse it
            # rather than dispatching an undefined / partial solution.
            solution = sol1
            status_name = "{}_T2_{}".format(status1, status2)
        else:
            solution = sol2
            status_name = status2

        time_expense = round(time.time() - start_t, 3)
        logger.debug(
            "HiGHS solving elapsed, model status: {}, time expense: {}s".format(
                status_name, time_expense
            )
        )

        # Record the problem instance if solving took longer than 1 second.
        if time_expense > 1:
            over_extended_problem = {
                "Jobs": {
                    j_idx: {
                        "ops": [
                            (float(m_idx), float(in_system_jobs[j_idx].pt_by_m_idx[m_idx]))
                            for m_idx in traj
                        ],
                        "avail": round(float(job_available_t[j_idx]), 3),
                    }
                    for j_idx, traj in remaining_trajectories.items()
                },
                "Machines": {m: round(rt, 3) for m, rt in machine_release_t.items()},
                "Expense": time_expense,
            }
        else:
            over_extended_problem = None

        converted_var_op_begin_t = {
            key: round(float(solution.col_value[col]), 1)
            for key, col in start_col.items()
        }
        return converted_var_op_begin_t, over_extended_problem, status_name


# ---------------------------------------------------------------------------
# Centralized-solver registry
# ---------------------------------------------------------------------------
# Single source of truth mapping a scheduling-method name to the class that
# implements the centralized optimizer. `Machine`s never call these directly;
# `CentralScheduler` resolves the backend here and dispatches to it, and both the
# simulator and the narrator use `is_optimization_scheduler` to decide whether a
# centralized schedule is needed at all.
#
# Every registered solver must expose the same classmethod contract:
#
#     solve_scheduling_problem(logger, env, m_list, job_intersections,
#                              remaining_trajectories, in_system_jobs)
#         -> (var_op_begin_t, over_extended_problem, status_name)
#
# To override or extend a solver (custom objective, a mock in tests, a new
# CP/MILP backend, ...), subclass an existing class and replace the registry
# entry, or register a new name:
#
#     from src.scheduler.scheduler import OPTIMIZATION_SOLVERS, ORTools
#     class MyORTools(ORTools):
#         @classmethod
#         def solve_scheduling_problem(cls, *args, **kwargs):
#             ...  # custom logic
#     OPTIMIZATION_SOLVERS["ORTools"] = MyORTools
#
# Alternatively, inject a one-off backend when constructing the shopfloor:
#     Shopfloor(..., sqc_method="ORTools", solver_cls=MyORTools)
OPTIMIZATION_SOLVERS = {
    "GurobiOptimizer": GurobiOptimizer,
    "ORTools": ORTools,
    "HiGHS": HiGHSOptimizer,
}


def resolve_optimization_solver(sqc_method):
    """Return the solver class backing ``sqc_method``, or ``None`` for reactive heuristics.

    ``sqc_method`` may be a registry name (``"ORTools"``), a solver class, or a
    heuristic callable. Name lookup is tried first so registry overrides by name keep
    working; an identity fallback covers classes registered under an alias key.
    """
    name = getattr(sqc_method, "__name__", str(sqc_method))
    solver = OPTIMIZATION_SOLVERS.get(name)
    if solver is not None:
        return solver
    for registered in OPTIMIZATION_SOLVERS.values():
        if registered is sqc_method:
            return registered
    return None


def is_optimization_scheduler(sqc_method) -> bool:
    """Whether ``sqc_method`` selects a centralized optimization scheduler."""
    return resolve_optimization_solver(sqc_method) is not None
