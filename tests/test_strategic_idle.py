"""Strategic idleness: the `in_strategic_idle` flag must survive the breakdown flow.

A machine can be strategically idle (waiting for its scheduled head job) when a
breakdown hits. `Narrator.process_machine_breakdown` overwrites `Machine.status`
with "down" and then "idle", which historically suppressed the `STR_IDL_END` event
(the log guards tested `status == "strategic_idle"`). The result was a dangling
`STR_IDL_ON` and a wrong "strategic_idle" phase in offline reconstruction.

These tests pin the fix: strategic idleness is tracked by a dedicated flag that is
independent of `status`, so it survives the breakdown clobbering.
"""

from __future__ import annotations

import unittest

import simpy

from src.simulator.event import Recorder
from src.simulator.machine import Machine


class _FakeJob:
    def __init__(self, j_idx: int) -> None:
        self.j_idx = j_idx

    def after_arrival(self) -> None:
        pass


class _FakeLogger:
    def info(self, *args, **kwargs) -> None:
        pass

    def debug(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass

    def exception(self, *args, **kwargs) -> None:
        pass


class TestStrategicIdleSurvivesBreakdown(unittest.TestCase):
    def _make_machine(self):
        env = simpy.Environment()
        recorder = Recorder(m_no=1)
        m = Machine(
            env=env,
            logger=_FakeLogger(),
            recorder=recorder,
            m_idx=0,
            schedule_mode=True,
        )
        # A competing job is already waiting; the scheduled head (5) has not arrived.
        m.next_job_in_schedule = 5
        m.queue.append(_FakeJob(1))
        return recorder, m

    def _events(self, recorder, event_type):
        return [e for e in recorder.sim_events_list if e[2] == event_type]

    def test_entry_sets_flag_and_str_idl_on(self):
        recorder, m = self._make_machine()
        m.check_strategic_idleness()

        self.assertTrue(m.in_strategic_idle)
        self.assertEqual(m.status, "strategic_idle")
        self.assertEqual(recorder.sqc_cnt_SI, 1)
        self.assertEqual(len(self._events(recorder, "STR_IDL_ON")), 1)

    def test_breakdown_clobbering_status_keeps_strategic_idle_flag(self):
        recorder, m = self._make_machine()
        m.check_strategic_idleness()

        # Mirrors Narrator.process_machine_breakdown: status is forced to "down" on
        # failure and back to "idle" on repair, losing the "strategic_idle" value.
        m.status = "down"
        m.status = "idle"

        # The flag (source of truth) must still be set even though `status` says "idle".
        self.assertTrue(m.in_strategic_idle)

        # Head job finally arrives -> the span must close with STR_IDL_END.
        m.job_arrival(_FakeJob(5))
        self.assertFalse(m.in_strategic_idle)
        self.assertEqual(len(self._events(recorder, "STR_IDL_END")), 1)

    def test_update_status_after_new_schedule_closes_span(self):
        recorder, m = self._make_machine()
        m.check_strategic_idleness()

        # A rebuild makes an already-queued job the new head.
        m.next_job_in_schedule = 1
        m.update_status_after_new_schedule()

        self.assertFalse(m.in_strategic_idle)
        self.assertEqual(len(self._events(recorder, "STR_IDL_END")), 1)

    def test_notify_schedule_update_closes_span_on_empty_schedule(self):
        recorder, m = self._make_machine()
        m.check_strategic_idleness()

        # A rebuild leaves no scheduled work for this machine.
        m.next_job_in_schedule = -1
        m.notify_schedule_update()

        self.assertFalse(m.in_strategic_idle)
        end_events = self._events(recorder, "STR_IDL_END")
        self.assertEqual(len(end_events), 1)
        self.assertIsNone(end_events[0][4])  # j_idx is NULL for the sentinel case


if __name__ == "__main__":
    unittest.main()
