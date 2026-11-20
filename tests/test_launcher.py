"""
P4 launcher tests for run_system.py with mocked child processes.

No real processes are started: subprocess.Popen returns scripted fakes, and
time.sleep is patched out.
"""

import io
import subprocess
import unittest
from contextlib import redirect_stdout
from unittest import mock

import run_system

STILL_RUNNING = "still-running"  # wait() raises TimeoutExpired


class FakeProcess:
    """Scripted child process. Each wait() consumes one step of the script."""

    MAX_WAITS = 200  # guard against an accidental infinite wait loop in tests

    def __init__(self, name, wait_script=(), exit_on_terminate=True):
        self.name = name
        self.script = list(wait_script)
        self.returncode = None
        self.exit_on_terminate = exit_on_terminate
        self.terminate_calls = 0
        self.kill_calls = 0
        self.waits = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.waits += 1
        if self.waits > self.MAX_WAITS:
            raise AssertionError(f"{self.name}: too many wait() calls")
        if self.returncode is not None:
            return self.returncode
        step = self.script.pop(0) if self.script else STILL_RUNNING
        if step == STILL_RUNNING:
            raise subprocess.TimeoutExpired(self.name, timeout)
        if isinstance(step, type) and issubclass(step, BaseException):
            raise step()
        self.returncode = step
        return step

    def terminate(self):
        self.terminate_calls += 1
        if self.exit_on_terminate:
            self.returncode = -15

    def kill(self):
        self.kill_calls += 1
        self.returncode = -9


def run_launcher(dashboard, scanner):
    out = io.StringIO()
    with mock.patch.object(run_system.subprocess, "Popen", side_effect=[dashboard, scanner]) as popen, \
         mock.patch.object(run_system.time, "sleep"), \
         redirect_stdout(out):
        code = run_system.main()
    return code, out.getvalue(), popen


class TestLauncher(unittest.TestCase):
    def test_normal_scanner_exit_stops_dashboard(self):
        dashboard = FakeProcess("dashboard")
        scanner = FakeProcess("scanner", [STILL_RUNNING, 0])
        code, out, popen = run_launcher(dashboard, scanner)

        self.assertEqual(code, 0)
        self.assertEqual(dashboard.terminate_calls, 1)
        self.assertEqual(scanner.terminate_calls, 0)
        self.assertIn("Scanner closed normally", out)
        self.assertNotIn("NOT ACTIVE", out)
        self.assertEqual(popen.call_count, 2, "scanner is started once and never restarted")

    def test_abnormal_exit_keeps_dashboard_until_ctrl_c(self):
        dashboard = FakeProcess("dashboard", [STILL_RUNNING, STILL_RUNNING, KeyboardInterrupt])
        scanner = FakeProcess("scanner", [2])
        code, out, popen = run_launcher(dashboard, scanner)

        self.assertEqual(code, 2)
        self.assertIn("exit code 2", out)
        self.assertIn("ATTENDANCE SCANNING IS NOT ACTIVE", out)
        self.assertIn("will not restart automatically", out)
        self.assertGreaterEqual(dashboard.waits, 3, "dashboard kept running while waiting")
        self.assertEqual(dashboard.terminate_calls, 1, "Ctrl+C stops the dashboard")
        self.assertEqual(popen.call_count, 2, "scanner is not restarted")

    def test_abnormal_exit_then_dashboard_exits_on_its_own(self):
        dashboard = FakeProcess("dashboard", [STILL_RUNNING, 1])
        scanner = FakeProcess("scanner", [2])
        code, out, _ = run_launcher(dashboard, scanner)

        self.assertEqual(code, 2)
        self.assertIn("Dashboard exited on its own (exit code 1)", out)
        self.assertEqual(dashboard.terminate_calls, 0, "already exited; nothing to stop")

    def test_ctrl_c_while_scanning_gives_scanner_time_then_stops_dashboard(self):
        dashboard = FakeProcess("dashboard")
        # Ctrl+C arrives during the scanner wait; the scanner then exits itself.
        scanner = FakeProcess("scanner", [STILL_RUNNING, KeyboardInterrupt, 130])
        code, out, _ = run_launcher(dashboard, scanner)

        self.assertEqual(code, 130)
        self.assertEqual(scanner.terminate_calls, 0, "scanner exited within the grace period")
        self.assertEqual(dashboard.terminate_calls, 1)
        self.assertIn("Ctrl+C received", out)

    def test_ctrl_c_with_unresponsive_scanner_terminates_it(self):
        dashboard = FakeProcess("dashboard")
        scanner = FakeProcess("scanner", [KeyboardInterrupt, STILL_RUNNING])
        code, _, _ = run_launcher(dashboard, scanner)

        self.assertEqual(code, 130)
        self.assertEqual(scanner.terminate_calls, 1)
        self.assertEqual(dashboard.terminate_calls, 1)

    def test_ctrl_c_during_dashboard_startup_stops_only_dashboard(self):
        dashboard = FakeProcess("dashboard")
        out = io.StringIO()
        with mock.patch.object(run_system.subprocess, "Popen", side_effect=[dashboard]) as popen, \
             mock.patch.object(run_system.time, "sleep", side_effect=KeyboardInterrupt), \
             redirect_stdout(out):
            code = run_system.main()
        self.assertEqual(code, 130)
        self.assertEqual(popen.call_count, 1, "scanner never started")
        self.assertEqual(dashboard.terminate_calls, 1)

    def test_dashboard_that_ignores_terminate_is_killed(self):
        dashboard = FakeProcess("dashboard", exit_on_terminate=False)
        scanner = FakeProcess("scanner", [0])
        run_launcher(dashboard, scanner)
        self.assertEqual(dashboard.terminate_calls, 1)
        self.assertEqual(dashboard.kill_calls, 1)


if __name__ == "__main__":
    unittest.main()
