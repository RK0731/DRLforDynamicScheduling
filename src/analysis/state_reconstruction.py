#!/usr/bin/env python3
"""
Offline reconstruction of the complete shopfloor condition at an arbitrary time.

The simulator persists a time-stamped trace of every operation (`operations`), every
breakdown (`breakdowns`), every sequencing decision (`decisions`), every rescheduling
snapshot (`schedule_cycles`), and every discrete event (`sim_events`). This module
replays that trace to rebuild what the entire shopfloor looked like at time `t`,
without re-running SimPy.

This is intended for:

  - building observation/state vectors for offline RL training or imitation learning;
  - replaying a specific decision (``snapshot_at_decision``) with the exact queue that
    the deciding machine saw;
  - counterfactual / debugging analysis of scheduler behaviour.

The returned dictionary is JSON-serializable (``json.dumps`` converts integer dict
keys to strings) and has the following shape::

    {
      "run_id": 1,
      "t": 77.0,
      "machines": [
        {
          "m_idx": 0,
          "status": "idle" | "processing" | "strategic_idle" | "down",
          "current_job": {...} | None,
          "queue": [ {...}, ... ],          # ordered waiting jobs (arrival order)
          "queue_len": 2,
          "breakdown": {...} | None,
          "next_job_in_schedule": 3 | None,
          "release_T": 77.0,                # expected release time visible to scheduler
          "hidden_release_T": 77.0,         # actual release time (may differ)
        },
        ...
      ],
      "jobs": {                              # jobs currently in the system
        "3": {
          "j_idx": 3, "status": "queuing" | "processing" | "not_started",
          "current_machine": 2, "op_seq": 1,
          "creation_t": 60.0, "due_t": 120.0,
          "arrival_t": 70.0, "wait_t": 7.0, "begin_t": None,
          "completed_ops": 1,
          "remaining_machines": [2, 4, 1],
          "remaining_expected_pt": [5.0, 3.0, 8.0],
          "remaining_actual_pt": [5.0, 3.0, 8.0],
          "tardiness_so_far": 0.0, "flowtime_so_far": 17.0,
          "scheduled_begin_t": 76.0,
        },
        ...
      },
      "summary": {
        "num_machines": 5, "num_jobs_in_system": 7,
        "num_processing_machines": 2, "num_machines_down": 0,
        "num_strategic_idle": 0, "num_waiting_jobs": 5,
        "num_completed_jobs": 3, "num_created_jobs": 10,
      },
      "decisions": [ {...}, ... ],           # decisions made exactly at `t`
      "target_decision": {...},              # only for `snapshot_at_decision`
    }

Usage::

    from src.analysis import ShopfloorStateReconstructor

    recon = ShopfloorStateReconstructor(run_id=1)
    state = recon.snapshot(t=77.0)
    state = recon.snapshot_at_decision(decision_id=42)

    for state in recon.iter_snapshots():
        ...  # one state per recorded decision

Command line::

    python -m src.analysis --run_id 1 --time 77.0
    python -m src.analysis --run_id 1 --decision_id 42
"""

from __future__ import annotations

import json
import sqlite3
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..utilities import DEFAULT_DB_PATH


class ShopfloorStateReconstructor:
    """
    Rebuild the full shopfloor state at any time from one simulation run's SQLite trace.

    Parameters
    ----------
    db_path : str or Path, optional
        Path to the persistent SQLite database (defaults to ``logs/simulation.db``).
    run_id : int, optional
        Episode to reconstruct. Defaults to the latest run in the database.
    """

    def __init__(self, db_path: Optional[Path] = None, run_id: Optional[int] = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        self.run_id = int(run_id) if run_id is not None else self._latest_run_id()
        self._load()

    # ------------------------------------------------------------------ loading
    def _latest_run_id(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT MAX(run_id) FROM runs").fetchone()
        if row is None or row[0] is None:
            raise ValueError(f"No simulation runs found in {self.db_path}")
        return int(row[0])

    def _load(self) -> None:
        """Load and index the run's jobs, operations, breakdowns, decisions, and events."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            run = conn.execute("SELECT * FROM runs WHERE run_id = ?", (self.run_id,)).fetchone()
            if run is None:
                raise ValueError(f"run_id={self.run_id} not found in {self.db_path}")
            self.run: Dict[str, Any] = dict(run)
            self.m_no = int(self.run["m_no"])

            # ---- jobs -------------------------------------------------------
            self.jobs: Dict[int, Dict[str, Any]] = {}
            for row in conn.execute("SELECT * FROM jobs WHERE run_id = ?", (self.run_id,)):
                job = dict(row)
                job["trajectory"] = json.loads(job.pop("trajectory_json"))
                job["exp_pt"] = json.loads(job.pop("exp_pt_json"))
                job["actual_pt"] = json.loads(job.pop("actual_pt_json"))
                self.jobs[int(job["j_idx"])] = job

            # ---- operations -------------------------------------------------
            self.ops_by_machine: Dict[int, List[Dict[str, Any]]] = {m: [] for m in range(self.m_no)}
            self.ops_by_job: Dict[int, List[Dict[str, Any]]] = {j: [] for j in self.jobs}
            for row in conn.execute("SELECT * FROM operations WHERE run_id = ?", (self.run_id,)):
                op = dict(row)
                self.ops_by_machine[int(op["m_idx"])].append(op)
                self.ops_by_job[int(op["j_idx"])].append(op)

            # Machine operations must be sorted by start time for bisection
            for ops in self.ops_by_machine.values():
                ops.sort(key=lambda o: (o["begin_t"], o["op_seq"]))
            # Job operations must be sorted by operation sequence
            for ops in self.ops_by_job.values():
                ops.sort(key=lambda o: o["op_seq"])
            self._begin_times = {
                m: [o["begin_t"] for o in ops] for m, ops in self.ops_by_machine.items()
            }

            # ---- breakdowns -------------------------------------------------
            self.breakdowns_by_machine: Dict[int, List[Dict[str, Any]]] = {
                m: [] for m in range(self.m_no)
            }
            for row in conn.execute("SELECT * FROM breakdowns WHERE run_id = ?", (self.run_id,)):
                bkd = dict(row)
                self.breakdowns_by_machine[int(bkd["m_idx"])].append(bkd)
            for bkds in self.breakdowns_by_machine.values():
                bkds.sort(key=lambda b: b["begin_t"])

            # ---- decisions --------------------------------------------------
            self.decisions: List[Dict[str, Any]] = []
            self.decision_by_id: Dict[int, Dict[str, Any]] = {}
            self.decisions_by_time: Dict[float, List[Dict[str, Any]]] = {}
            try:
                rows = conn.execute(
                    "SELECT * FROM decisions WHERE run_id = ? ORDER BY sim_t, m_idx",
                    (self.run_id,),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []  # database predates the `decisions` table
            for row in rows:
                dec = dict(row)
                # `candidates_json` may be NULL for rows written by an older schema;
                # treat missing replay metadata as an empty queue/candidate map.
                dec["queue"] = json.loads(dec.pop("queue_json") or "[]")
                dec["candidates"] = json.loads(dec.pop("candidates_json") or "{}")
                self.decisions.append(dec)
                self.decision_by_id[int(dec["decision_id"])] = dec
                t_key = round(float(dec["sim_t"]), 2)
                self.decisions_by_time.setdefault(t_key, []).append(dec)

            # `decision_id` is auto-increment and reflects the actual (SimPy event) order
            # decisions were made, so use it to define chronological order.
            self.decisions.sort(key=lambda d: int(d["decision_id"]))
            for decs in self.decisions_by_time.values():
                decs.sort(key=lambda d: int(d["decision_id"]))

            # Attach each operation's decision id so that simultaneous (same-timestamp)
            # decisions can be ordered: a job starting exactly at `t` is "processing" iff
            # its sequencing decision preceded the decision being reconstructed.
            self.op_decision_id: Dict[Tuple[int, int, int], int] = {}
            for dec in self.decisions:
                key = (int(dec["m_idx"]), int(dec["j_idx"]), int(dec["op_seq"]))
                self.op_decision_id[key] = int(dec["decision_id"])
            for ops in self.ops_by_machine.values():
                for op in ops:
                    key = (int(op["m_idx"]), int(op["j_idx"]), int(op["op_seq"]))
                    op["decision_id"] = self.op_decision_id.get(key)

            # ---- schedule cycles (for next_job_in_schedule) -----------------
            self.schedule_cycles: List[Dict[str, Any]] = []
            try:
                rows = conn.execute(
                    "SELECT * FROM schedule_cycles WHERE run_id = ? ORDER BY sim_t",
                    (self.run_id,),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []  # database predates the `schedule_cycles` table
            for row in rows:
                cycle = dict(row)
                cycle["m_schedule"] = json.loads(cycle.pop("m_schedule_json"))
                cycle["j_schedule"] = json.loads(cycle.pop("j_schedule_json"))
                self.schedule_cycles.append(cycle)
            self._cycle_times = [c["sim_t"] for c in self.schedule_cycles]

            # ---- discrete events (queue order + strategic idleness) ---------
            # `event_id` is monotonic in insertion order, which is exactly the order in
            # which jobs were appended to a machine queue (resolves same-timestamp ties).
            self.arrival_seq: Dict[Tuple[int, int], int] = {}
            self.si_events: Dict[int, List[Dict[str, Any]]] = {
                m: [] for m in range(self.m_no)
            }
            for row in conn.execute(
                """
                SELECT event_id, sim_t, event_type, m_idx, j_idx
                FROM sim_events
                WHERE run_id = ? AND event_type IN ('JOB_IN', 'STR_IDL_ON', 'STR_IDL_END')
                ORDER BY event_id
                """,
                (self.run_id,),
            ):
                ev = dict(row)
                m_idx = ev.get("m_idx")
                if m_idx is None:
                    continue
                if ev["event_type"] == "JOB_IN":
                    if ev.get("j_idx") is not None:
                        self.arrival_seq[(int(m_idx), int(ev["j_idx"]))] = int(ev["event_id"])
                elif ev["event_type"] in ("STR_IDL_ON", "STR_IDL_END"):
                    self.si_events[int(m_idx)].append(ev)

    # ------------------------------------------------------------- public API
    def decision_ids(self) -> List[int]:
        """Return all recorded decision ids in chronological order."""
        return [int(d["decision_id"]) for d in self.decisions]

    def snapshot(self, t: float, tie: str = "post") -> Dict[str, Any]:
        """
        Reconstruct the complete shopfloor condition at continuous time ``t``.

        This is a pure timeline reconstruction: an operation is "processing" on a machine
        iff it started at or before ``t`` and has not yet finished. Use this for arbitrary
        inspection. To reproduce the exact pre-decision state of one sequencing decision
        (with simultaneous same-timestamp decisions correctly ordered), use
        :meth:`snapshot_at_decision` instead.

        Parameters
        ----------
        t : float
            Simulation timestamp to reconstruct.
        tie : {"post", "pre"}
            How to treat operations that start exactly at ``t``:
              - ``"post"`` (default): a job starting at ``t`` is already processing.
              - ``"pre"``: a job starting at ``t`` is still considered queued.

        Returns
        -------
        dict
            JSON-serializable shopfloor state (see module docstring).
        """
        if tie not in ("pre", "post"):
            raise ValueError("tie must be 'pre' or 'post'")
        return self._snapshot(round(float(t), 2), tie, None)

    def snapshot_at_decision(self, decision_id: int) -> Dict[str, Any]:
        """
        Reconstruct the shopfloor state exactly as seen at one recorded decision.

        The deciding machine's queue is taken verbatim from ``decisions.queue_json`` and
        its chosen job is still queued (not yet processing). Every other machine is
        reconstructed from the timeline, with operations that start at the same timestamp
        ``t`` classified by decision order: a machine whose decision at ``t`` precedes this
        one is already processing, while a machine whose decision follows it is still queuing.
        """
        dec = self.decision_by_id.get(int(decision_id))
        if dec is None:
            raise KeyError(f"decision_id={decision_id} not found in run {self.run_id}")
        t = round(float(dec["sim_t"]), 2)
        state = self._snapshot(t, "post", int(decision_id))

        # Replace the deciding machine's reconstructed queue with the authoritative
        # `queue_json` / `candidates_json` (also resolves same-timestamp arrival ties).
        m_idx = int(dec["m_idx"])
        machine = state["machines"][m_idx]
        machine["queue"] = self._recorded_queue(dec, state["jobs"])
        machine["current_job"] = None
        machine["queue_len"] = len(machine["queue"])

        # The queue override can change the aggregate count, so recompute it from the
        # final per-machine state rather than keeping the timeline-only value.
        state["summary"]["num_waiting_jobs"] = sum(m["queue_len"] for m in state["machines"])

        state["target_decision"] = self._decision_info(dec)
        return state

    def iter_snapshots(
        self, decision_ids: Optional[Iterable[int]] = None
    ) -> Iterable[Dict[str, Any]]:
        """Yield one reconstructed pre-decision state per recorded decision (in order)."""
        ids = list(decision_ids) if decision_ids is not None else self.decision_ids()
        for decision_id in ids:
            yield self.snapshot_at_decision(decision_id)

    # ------------------------------------------------------- snapshot internals
    def _snapshot(self, t: float, tie: str, cutoff: Optional[int]) -> Dict[str, Any]:
        """Core reconstruction shared by ``snapshot`` (``cutoff=None``) and ``snapshot_at_decision``."""
        # 1. Job-level state for every job currently in the system
        job_states: Dict[int, Dict[str, Any]] = {}
        for j_idx, job in self.jobs.items():
            state = self._job_state(job, t, tie, cutoff)
            if state is not None:
                job_states[j_idx] = state

        # 2. Machine-level state (queue, current job, breakdown, strategic idleness)
        machines = [
            self._machine_state(m, t, tie, cutoff, job_states) for m in range(self.m_no)
        ]

        # 3. Aggregate summary
        summary = {
            "num_machines": self.m_no,
            "num_jobs_in_system": len(job_states),
            "num_processing_machines": sum(1 for m in machines if m["status"] == "processing"),
            "num_machines_down": sum(1 for m in machines if m["status"] == "down"),
            "num_strategic_idle": sum(1 for m in machines if m["status"] == "strategic_idle"),
            "num_waiting_jobs": sum(m["queue_len"] for m in machines),
            "num_completed_jobs": sum(
                1
                for job in self.jobs.values()
                if job["completion_t"] is not None and job["completion_t"] <= t
            ),
            "num_created_jobs": sum(1 for job in self.jobs.values() if job["creation_t"] <= t),
        }

        return {
            "run_id": self.run_id,
            "t": t,
            "machines": machines,
            "jobs": job_states,
            "summary": summary,
            "decisions": [
                self._decision_info(dec) for dec in self.decisions_by_time.get(t, [])
            ],
        }

    def _recorded_queue(
        self, dec: Dict[str, Any], job_states: Dict[int, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build the deciding machine's queue from the authoritative ``queue_json``/``candidates_json``."""
        queue: List[Dict[str, Any]] = []
        for j_idx in dec["queue"]:
            entry = {"j_idx": int(j_idx)}
            entry.update(dec["candidates"].get(str(j_idx), {}))
            # `op_seq` is not stored in candidates_json; recover it from the job state
            entry["op_seq"] = job_states.get(int(j_idx), {}).get("op_seq")
            queue.append(entry)
        return queue

    # --------------------------------------------------------- job internals
    @staticmethod
    def _phase(op: Dict[str, Any], t: float, tie: str, cutoff: Optional[int] = None) -> str:
        """
        Classify an operation relative to time ``t``.

        Returns one of: ``"done"``, ``"processing"``, ``"waiting"``, ``"future"``.

        Operations starting exactly at ``t`` are ambiguous: when ``cutoff`` (a decision id)
        is given, they are "processing" iff their own decision preceded ``cutoff`` (this is
        how simultaneous same-timestamp decisions are ordered); otherwise ``tie`` decides
        ("post" -> processing, "pre" -> waiting).
        """
        # An operation flagged `completed=0` started but never finished before the episode
        # ended: its recorded `end_t` is only the expected completion time, so it must
        # remain "processing" for every t after it began.
        if int(op.get("completed", 1)) == 0:
            if op["begin_t"] < t:
                return "processing"
            if op["begin_t"] == t:
                # Respect the same-timestamp tie rule as a completed operation: the
                # decision that starts this operation is still pre-decision for itself.
                if cutoff is not None:
                    op_did = op.get("decision_id")
                    return "processing" if (op_did is not None and op_did < cutoff) else "waiting"
                return "processing" if tie == "post" else "waiting"
            if op["arrival_t"] <= t:
                return "waiting"
            return "future"
        if op["end_t"] <= t:
            return "done"
        if op["begin_t"] < t:
            return "processing"
        if op["begin_t"] == t:
            if cutoff is not None:
                op_did = op.get("decision_id")
                return "processing" if (op_did is not None and op_did < cutoff) else "waiting"
            return "processing" if tie == "post" else "waiting"
        if op["arrival_t"] <= t:
            return "waiting"
        return "future"

    def _job_state(
        self, job: Dict[str, Any], t: float, tie: str, cutoff: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Build the state of one job at time ``t``, or ``None`` if it is not in the system."""
        if job["creation_t"] > t:
            return None

        current_op: Optional[Dict[str, Any]] = None
        for op in self.ops_by_job.get(int(job["j_idx"]), []):
            if self._phase(op, t, tie, cutoff) != "done":
                current_op = op
                break

        # All operations finished -> the job has left the system
        if current_op is None:
            return None

        phase = self._phase(current_op, t, tie, cutoff)
        completed_ops = int(current_op["op_seq"])
        remaining_machines = job["trajectory"][completed_ops:]
        remaining_exp_pt = job["exp_pt"][completed_ops:]
        remaining_actual_pt = job["actual_pt"][completed_ops:]

        if phase == "processing":
            status = "processing"
            current_machine = int(current_op["m_idx"])
            arrival_t = float(current_op["arrival_t"])
            wait_t = float(current_op["begin_t"]) - arrival_t
            begin_t = float(current_op["begin_t"])
        elif phase == "waiting":
            status = "queuing"
            current_machine = int(current_op["m_idx"])
            arrival_t = float(current_op["arrival_t"])
            wait_t = t - arrival_t
            begin_t = None
        else:  # "future" (created but not yet routed to a machine)
            status = "not_started"
            current_machine = None
            arrival_t = wait_t = begin_t = None

        return {
            "j_idx": int(job["j_idx"]),
            "status": status,
            "current_machine": current_machine,
            "op_seq": int(current_op["op_seq"]),
            "creation_t": round(float(job["creation_t"]), 2),
            "due_t": round(float(job["due_t"]), 2),
            "arrival_t": round(arrival_t, 2) if arrival_t is not None else None,
            "wait_t": round(wait_t, 2) if wait_t is not None else None,
            "begin_t": round(begin_t, 2) if begin_t is not None else None,
            "completed_ops": completed_ops,
            "remaining_machines": [int(m) for m in remaining_machines],
            "remaining_expected_pt": [round(float(x), 2) for x in remaining_exp_pt],
            "remaining_actual_pt": [round(float(x), 2) for x in remaining_actual_pt],
            "tardiness_so_far": round(max(0.0, t - float(job["due_t"])), 2),
            "flowtime_so_far": round(t - float(job["creation_t"]), 2),
            "scheduled_begin_t": current_op.get("scheduled_begin_t"),
        }

    # ----------------------------------------------------- machine internals
    def _current_op(
        self, m_idx: int, t: float, tie: str, cutoff: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Return the operation currently being processed on machine ``m_idx`` at time ``t``."""
        ops = self.ops_by_machine.get(m_idx, [])
        begin_times = self._begin_times.get(m_idx, [])
        if not ops:
            return None
        # Operations on a machine are sequential, so only the latest started op can span t.
        # First check the last op that started strictly before t.
        idx = bisect_left(begin_times, t) - 1
        if idx >= 0:
            op = ops[idx]
            # An incomplete operation never releases the machine, even past its expected end.
            if int(op.get("completed", 1)) == 0 or op["end_t"] > t:
                return op
        # Then the (at most one) op starting exactly at t.
        j = bisect_left(begin_times, t)
        if j < len(ops) and ops[j]["begin_t"] == t:
            op = ops[j]
            if int(op.get("completed", 1)) == 0 or op["end_t"] > t:
                if cutoff is not None:
                    op_did = op.get("decision_id")
                    if op_did is not None and op_did < cutoff:
                        return op
                elif tie == "post":
                    return op
        return None

    def _breakdown(self, m_idx: int, t: float) -> Optional[Dict[str, Any]]:
        """Return the actual breakdown interval covering ``t`` for machine ``m_idx``, if any."""
        for bkd in self.breakdowns_by_machine.get(m_idx, []):
            if bkd["begin_t"] <= t < bkd["end_t"]:
                return bkd
        return None

    def _strategic_idle(self, m_idx: int, t: float) -> Optional[Dict[str, Any]]:
        """
        Return the active ``STR_IDL_ON`` event for machine ``m_idx`` at ``t``, if the
        latest strategic-idleness transition is an "on" (not followed by an "end").
        """
        last_on: Optional[Dict[str, Any]] = None
        last_end: Optional[Dict[str, Any]] = None
        for ev in self.si_events.get(m_idx, []):
            if ev["sim_t"] > t:
                break
            if ev["event_type"] == "STR_IDL_ON":
                last_on = ev
            else:
                last_end = ev
        if last_on is not None and (last_end is None or last_end["event_id"] < last_on["event_id"]):
            return last_on
        return None

    def _next_job_in_schedule(self, m_idx: int, t: float) -> Optional[int]:
        """
        Best-effort reconstruction of ``Machine.next_job_in_schedule`` at time ``t``:
        start from the latest rescheduling snapshot at/before ``t`` and remove every job
        that has already started on this machine since that snapshot.
        """
        if not self.schedule_cycles:
            return None
        idx = bisect_right(self._cycle_times, t) - 1
        if idx < 0:
            return None
        cycle = self.schedule_cycles[idx]
        schedule = cycle["m_schedule"].get(str(m_idx), [])
        if not schedule:
            return None
        started = {
            int(op["j_idx"])
            for op in self.ops_by_machine.get(m_idx, [])
            if cycle["sim_t"] <= op["begin_t"] <= t
        }
        for j_idx in schedule:
            if int(j_idx) not in started:
                return int(j_idx)
        return None

    def _queue_entry(self, job: Dict[str, Any], op: Dict[str, Any], t: float) -> Dict[str, Any]:
        """Feature snapshot of one waiting job (mirrors ``Machine._snapshot_candidates``)."""
        completed_ops = int(op["op_seq"])
        remaining_exp_pt = job["exp_pt"][completed_ops:]
        remaining_work = float(sum(remaining_exp_pt))
        ttd = float(job["due_t"]) - t
        return {
            "j_idx": int(job["j_idx"]),
            "op_seq": completed_ops,
            "arrival_t": round(float(op["arrival_t"]), 2),
            "wait_t": round(t - float(op["arrival_t"]), 2),
            "due_t": round(float(job["due_t"]), 2),
            "ttd": round(ttd, 2),
            "slack": round(ttd - remaining_work, 2),
            "current_pt": round(float(remaining_exp_pt[0]), 2) if remaining_exp_pt else None,
            "remaining_work": round(remaining_work, 2),
            "num_remaining_ops": len(remaining_exp_pt),
            "critical_ratio": round(ttd / max(remaining_work, 1e-6), 4),
        }

    def _current_job_entry(self, job: Dict[str, Any], op: Dict[str, Any], t: float) -> Dict[str, Any]:
        """Feature snapshot of the job currently being processed on a machine."""
        expected_end = float(op["begin_t"]) + float(op["expected_pt"])
        remaining_exp_pt = job["exp_pt"][int(op["op_seq"]):]
        remaining_work = float(sum(remaining_exp_pt))
        ttd = float(job["due_t"]) - t
        return {
            "j_idx": int(job["j_idx"]),
            "op_seq": int(op["op_seq"]),
            "begin_t": round(float(op["begin_t"]), 2),
            "expected_end_t": round(expected_end, 2),
            "actual_end_t": round(float(op["end_t"]), 2),
            "expected_pt": round(float(op["expected_pt"]), 2),
            "actual_pt": round(float(op["actual_pt"]), 2),
            "remaining_expected_pt": round(max(0.0, expected_end - t), 2),
            "remaining_actual_pt": round(max(0.0, float(op["end_t"]) - t), 2),
            "due_t": round(float(job["due_t"]), 2),
            "ttd": round(ttd, 2),
            "slack": round(ttd - remaining_work, 2),
            "scheduled_begin_t": op.get("scheduled_begin_t"),
        }

    def _machine_state(
        self,
        m_idx: int,
        t: float,
        tie: str,
        cutoff: Optional[int],
        job_states: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build the complete state of machine ``m_idx`` at time ``t``."""
        current_op = self._current_op(m_idx, t, tie, cutoff)
        waiting_ops = [
            op
            for op in self.ops_by_machine.get(m_idx, [])
            if self._phase(op, t, tie, cutoff) == "waiting"
        ]
        # Preserve exact arrival order using the JOB_IN event sequence (falls back to j_idx)
        waiting_ops.sort(
            key=lambda o: (
                float(o["arrival_t"]),
                self.arrival_seq.get((m_idx, int(o["j_idx"])), int(o["j_idx"])),
            )
        )

        current = (
            self._current_job_entry(self.jobs[int(current_op["j_idx"])], current_op, t)
            if current_op is not None
            else None
        )
        queue = [self._queue_entry(self.jobs[int(op["j_idx"])], op, t) for op in waiting_ops]

        breakdown = self._breakdown(m_idx, t)
        strategic_idle = self._strategic_idle(m_idx, t)

        if current is not None:
            status = "processing"
        elif breakdown is not None:
            status = "down"
        elif strategic_idle is not None:
            status = "strategic_idle"
        else:
            status = "idle"

        # Expected vs. actual release times, mirroring Machine.release_T / hidden_release_T
        if current is not None:
            release_T = current["expected_end_t"]
            hidden_release_T = current["actual_end_t"]
        elif breakdown is not None:
            release_T = round(float(breakdown["expected_end_t"]), 2)
            hidden_release_T = round(float(breakdown["end_t"]), 2)
        else:
            release_T = hidden_release_T = t

        breakdown_info = None
        if breakdown is not None:
            breakdown_info = {
                "begin_t": round(float(breakdown["begin_t"]), 2),
                "expected_end_t": round(float(breakdown["expected_end_t"]), 2),
                "actual_end_t": round(float(breakdown["end_t"]), 2),
                "expected_remaining": round(max(0.0, float(breakdown["expected_end_t"]) - t), 2),
                "actual_remaining": round(max(0.0, float(breakdown["end_t"]) - t), 2),
                "actual_duration": round(float(breakdown["actual_duration"]), 2),
                "prod_delay": breakdown.get("prod_delay"),
            }

        return {
            "m_idx": m_idx,
            "status": status,
            "current_job": current,
            "queue": queue,
            "queue_len": len(queue),
            "breakdown": breakdown_info,
            "next_job_in_schedule": self._next_job_in_schedule(m_idx, t),
            "release_T": round(float(release_T), 2),
            "hidden_release_T": round(float(hidden_release_T), 2),
        }

    # ------------------------------------------------------- decision internals
    @staticmethod
    def _decision_info(dec: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a loaded `decisions` row into a clean, JSON-serializable record."""
        return {
            "decision_id": int(dec["decision_id"]),
            "sim_t": round(float(dec["sim_t"]), 2),
            "m_idx": int(dec["m_idx"]),
            "decision_type": dec["decision_type"],
            "queue_len": int(dec["queue_len"]),
            "queue": [int(j) for j in dec["queue"]],
            "chosen_pos": int(dec["chosen_pos"]),
            "chosen_j_idx": int(dec["j_idx"]),
            "chosen_op_seq": int(dec["op_seq"]),
            "candidates": dec["candidates"],
        }


def reconstruct_state(
    db_path: Optional[Path] = None,
    run_id: Optional[int] = None,
    t: Optional[float] = None,
    decision_id: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience wrapper around :class:`ShopfloorStateReconstructor`.

    Provide either ``t`` (arbitrary time) or ``decision_id`` (pre-decision state).
    """
    recon = ShopfloorStateReconstructor(db_path=db_path, run_id=run_id)
    if decision_id is not None:
        return recon.snapshot_at_decision(decision_id)
    if t is None:
        raise ValueError("Provide either `t` or `decision_id`")
    return recon.snapshot(t, **kwargs)
