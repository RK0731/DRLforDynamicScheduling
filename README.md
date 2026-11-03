# DRLforDynamicScheduling

## Overview

This repository contains the source code for the research project titled **"Multi-agent Imitational, Chronological, and Asynchronous Reinforcement Learning Framework for Production Scheduling."**

The project tackles the **Dynamic Job Shop Scheduling Problem (DJSSP)** under real-world shopfloor uncertainties (Poisson job arrivals, non-preemptive machine breakdowns, and stochastic processing times). It includes:
1. **Discrete-Event Simulator (`src/simulator/`)**: Built on `SimPy`, modeling machines, jobs, queues, strategic idleness, and dynamic disruptions.
2. **Centralized Production Scheduler & Heuristics (`src/scheduler/`)**: Online rescheduling via Mixed-Integer Linear Programming (**Gurobi**) and Constraint Programming (**Google OR-Tools CP-SAT**), alongside classical dispatching rules (`FIFO`, `LIFO`, `Slack`, `CR`, `SPT`, `LWKR`).
3. **Deep Multi-Agent Reinforcement Learning (`src/DRL/`)**: Parameter-sharing Double Deep Q-Network (DDQN) agents for decentralized real-time job sequencing.

> **For AI Coding Agents & Contributors**: Please read [`AGENTS.md`](AGENTS.md) for a comprehensive architectural deep-dive, module-by-module guide, mathematical formulation, and simulation lifecycle invariants.

## Requirements

- **Python**: `>= 3.10`
- **Dependencies**: Listed in `requirements.txt` (`simpy`, `numpy`, `ortools`, `gurobipy`, `torch`, `matplotlib`, `tabulate`, etc.)
- **Solver Note**: A valid **Gurobi** license is required to use the default `GurobiOptimizer` scheduler. If you do not have a Gurobi license, use the `ORTools` CP-SAT solver (`-sqc ORTools`) or any heuristic sequencing rule.

## Project Structure

```text
.
├── AGENTS.md                       # Architectural & domain guide for AI agents and developers
├── doc/                            # Documentation, figures, and research proposals
├── logs/                           # Timestamped simulation logs and generated Gantt charts
├── math_modeling/                  # Mathematical optimization formulation (JSP_model.md)
├── src/                            # Source code package
│   ├── DRL/                        # Deep Multi-Agent Reinforcement Learning (DDQN, state/reward/padding)
│   │   ├── brain_sequencing.py     # Parameter-sharing DDQN agent and neural network architectures
│   │   ├── brain_routing.py        # Placeholder for routing MARL agent
│   │   ├── pytorch_padding.py      # Variable-length sequence padding/packing reference for RNN/LSTM
│   │   └── reward_function.py      # Reward function definitions
│   ├── scheduler/                  # Centralized production schedulers and heuristic dispatching rules
│   │   ├── scheduler.py            # CentralScheduler, GurobiOptimizer (MILP), and ORTools (CP-SAT)
│   │   ├── sequencing_rule.py      # Heuristic rules (FIFO, LIFO, Slack, CR, SPT, LWKR) & placeholders
│   │   └── ORToolsJSP.py           # Standalone OR-Tools CP-SAT JSSP example
│   ├── simulator/                  # SimPy discrete-event simulator
│   │   ├── simulator.py            # Simulator, SimulatorMultiThread, and Shopfloor classes
│   │   ├── event.py                # Narrator (job arrivals, breakdowns, post-sim KPIs) and Recorder
│   │   ├── machine.py              # Machine entity, queue management, and strategic idleness
│   │   ├── job.py                  # Job entity, routing trajectory, and operation records
│   │   └── exc.py                  # Custom simulator exception classes
│   └── utilities.py                # Logger configuration and Gantt chart plotting
├── main.py                         # Main CLI entry point to run simulations
├── README.md                       # Project documentation
└── requirements.txt                # Python package dependencies
```

## Quick Start & Usage

Run `main.py` for a demonstration (defaults to 5 machines, nominal span of 100 time units, 60% expected utilization, and `GurobiOptimizer`):

```bash
python main.py
```

No Gurobi license? Use Google **OR-Tools** (`ORTools`) or a heuristic rule (`FIFO`, `LIFO`, `Slack`, `CR`, `SPT`, `LWKR`):

```bash
python main.py -sqc ORTools
python main.py -sqc CR -seed 42
```

### Command-Line Arguments

| Flag | Long Option | Default | Description |
| :--- | :--- | :--- | :--- |
| `-m_no` | `--m_no` | `5` | Number of machines (workstations) in the shopfloor |
| `-seed` | `--seed` | `0` | Random seed (`0` generates a random seed automatically) |
| `-span` | `--span` | `100` | Nominal simulation horizon for generating job arrivals |
| `-utl` | `--E_utliz` | `0.6` | Expected system utilization rate (controls Poisson arrival rate) |
| `-dt` | `--due_tightness` | `2.0` | Due date tightness factor (upper multiplier on total processing time) |
| `-pt_r` | `--pt_range` | `1 10` | Range of nominal operation processing times `[min_pt, max_pt]` |
| `-pt_v` | `--processing_time_variability` | `False` | Enable stochastic processing times |
| `-pt_cv` | `--pt_cv` | `0.1` | Coefficient of variation (CV) when `-pt_v` is enabled |
| `-mbkd` | `--machine_breakdown` | `True` | Flag to disable machine breakdowns when passed |
| `-mtbf` | `--MTBF` | `50.0` | Mean Time Between Failures (MTBF) |
| `-rnd_mtbf` | `--random_MTBF` | `True` | Sample time between failures from an Exponential distribution |
| `-mttr` | `--MTTR` | `10.0` | Mean Time To Repair (MTTR) |
| `-rnd_mttr` | `--random_MTTR` | `False` | Sample repair time from Uniform `[0.5 * MTTR, 1.5 * MTTR]` |
| `-db` | `--db_path` | `logs/simulation.db` | Path to persistent SQLite database storing all simulation runs |
| `-draw` | `--draw_gantt` | `5` | Seconds to display interactive Gantt chart (`0` disables pop-up; disabled if `span >= 250`) |
| `-save_gantt` | `--save_gantt` | `True` | Flag to disable saving `gantt_chart.png` to the run's log directory |
| `-ns` | `--no_stream` | `False` | Suppress console (`INFO`) log output |
| `-sqc` | `--sqc_method` | `GurobiOptimizer` | Scheduler/rule: `GurobiOptimizer`, `ORTools`, `FIFO`, `LIFO`, `Slack`, `CR`, `SPT`, `LWKR`, `DRL_scheduler` |

### Logging & SQLite Database (`logs/simulation.db`)

Every simulation episode automatically persists its full structured data into a single SQLite database (`logs/simulation.db`) across 6 relational tables:
- `runs`: Episode configuration, parameters, and aggregate KPIs (mean/max tardiness, flowtime, utilization, wall/solver time)
- `jobs`: Per-job creation time, due date, completion time, status, tardiness, flowtime, and routing/PT JSON arrays
- `operations`: Per-operation execution records (`m_idx`, `decision_type`, `arrival_t`, `begin_t`, `end_t`, `wait_t`, `expected_pt`, `actual_pt`, `scheduled_begin_t`)
- `breakdowns`: Machine breakdown intervals, expected repair times, and production delays
- `schedule_cycles`: Centralized rescheduling snapshots, solver status, solve time, and machine/job schedules
- `sim_events`: Chronological discrete-event trace (`JOB_CREATED`, `JOB_IN`, `OP_START`, `OP_OUT`, `JOB_END`, `IDL_ON`, `STR_IDL_ON`, `BKD_START`, `RESCHEDULE`, etc.)

In addition, the 3 most recent timestamped text log folders (`logs/<timestamp>/sim.log` and `gantt_chart.png`) are retained for quick inspection.

### Example Simulation Output (Gantt Chart)

![Example of simulation result](doc/example.png)

