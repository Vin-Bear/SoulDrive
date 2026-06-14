import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import sidecar_runtime


class SidecarRuntimeTests(unittest.TestCase):
    def test_configure_runtime_streams_handles_gui_mode_without_console(self):
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        original_stream = sidecar_runtime._RUNTIME_LOG_STREAM

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                with patch.dict("os.environ", {"SOULDRIVE_APP_ROOT": temp_dir}):
                    sys.stdout = None
                    sys.stderr = None

                    sidecar_runtime.configure_runtime_streams()
                    print("local sidecar log check")

                    self.assertIsNotNone(sys.stdout)
                    self.assertIs(sys.stdout, sys.stderr)
                    self.assertTrue((Path(temp_dir) / "runtime" / "sidecar.log").exists())
            finally:
                stream = sidecar_runtime._RUNTIME_LOG_STREAM
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                sidecar_runtime._RUNTIME_LOG_STREAM = original_stream
                if stream is not None:
                    stream.close()

    def test_uvicorn_log_config_writes_to_runtime_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict("os.environ", {"SOULDRIVE_APP_ROOT": temp_dir}):
                config = sidecar_runtime.uvicorn_log_config()

        default_handler = config["handlers"]["default"]

        self.assertEqual(default_handler["class"], "logging.FileHandler")
        self.assertTrue(default_handler["filename"].endswith(str(Path("runtime") / "sidecar.log")))
        self.assertEqual(config["loggers"]["uvicorn.access"]["handlers"], [])

    def test_parent_pid_from_env_ignores_invalid_values(self):
        for raw_pid in ("", "abc", "-1", str(sidecar_runtime.os.getpid())):
            with patch.dict("os.environ", {"SOULDRIVE_PARENT_PID": raw_pid}, clear=False):
                self.assertIsNone(sidecar_runtime.parent_pid_from_env())

    def test_parent_process_is_alive_returns_false_for_missing_pid(self):
        self.assertFalse(sidecar_runtime.parent_process_is_alive(99999999))

    def test_removable_watch_requires_explicit_env_flag(self):
        with patch.dict("os.environ", {"SOULDRIVE_WATCH_REMOVABLE": "1"}, clear=False):
            self.assertTrue(sidecar_runtime.removable_watch_enabled())

        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(sidecar_runtime.removable_watch_enabled())

    def test_frozen_indexer_command_forces_process_exit_after_worker_returns(self):
        with patch.object(sidecar_runtime.sys, "frozen", True, create=True):
            with patch("core.indexer_worker.main", return_value=7) as worker_main:
                with patch.object(sidecar_runtime.os, "_exit") as exit_process:
                    result = sidecar_runtime.main(["indexer", "D:\\drive", "LITE"])

        worker_main.assert_called_once_with(["D:\\drive", "LITE"])
        exit_process.assert_called_once_with(7)
        self.assertEqual(result, 7)

    def test_frozen_indexer_command_exits_nonzero_when_worker_raises(self):
        with patch.object(sidecar_runtime.sys, "frozen", True, create=True):
            with patch("core.indexer_worker.main", side_effect=RuntimeError("index failed")):
                with patch.object(sidecar_runtime.os, "_exit", side_effect=SystemExit) as exit_process:
                    with self.assertRaises(SystemExit):
                        sidecar_runtime.main(["indexer", "D:\\drive", "LITE"])

        exit_process.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
