from __future__ import annotations

import importlib.util
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase, mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "data_loop.py"


def load_module():
    spec = importlib.util.spec_from_file_location("data_loop_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DataLoopTrainingTest(TestCase):
    def test_axis_model_training_is_triggered_for_accepted_images(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "axis_labelled" / "accepted").mkdir(parents=True)
            (root / "axis_labelled" / "accepted" / "sample.jpg").write_bytes(b"fake")

            with mock.patch("subprocess.run") as run_mock:
                module.maybe_train_axis_model(root)

            self.assertTrue(run_mock.called)
            commands = [call.args[0] if hasattr(call, "args") else call[0] for call in run_mock.call_args_list]
            any_cmd = any(
                isinstance(cmd, (list, tuple)) and any("prepare_axis_yolo_dataset.py" in str(part) for part in cmd)
                for cmd in commands
            )
            self.assertTrue(any_cmd)

    def test_impact_batch_auto_generates_predictions_then_launches_manual_review(self):
        impact_module_path = Path(__file__).resolve().parents[1] / "src" / "process_impact_batch.py"
        spec = importlib.util.spec_from_file_location("impact_batch_under_test", impact_module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "homographied" / "accepted"
            pending = Path(tmp_dir) / "impact_labelled" / "pending"
            accepted = Path(tmp_dir) / "impact_labelled" / "accepted"
            src.mkdir(parents=True)
            image_path = src / "sample_flat.jpg"
            image_path.write_bytes(b"fake")
            weights = Path(tmp_dir) / "best.pt"
            weights.write_bytes(b"fake")

            with mock.patch.object(module, "detect_impacts_yolo") as detect_mock:
                detect_mock.return_value = pending / "sample_flat.jpg"
                with mock.patch("subprocess.run") as run_mock:
                    module.process_impact_batch(src, pending, accepted, weights=weights)

            self.assertTrue(detect_mock.called)
            self.assertTrue(run_mock.called)
            commands = [call.args[0] if hasattr(call, "args") else call[0] for call in run_mock.call_args_list]
            any_labeler = any(
                isinstance(cmd, (list, tuple)) and any("label_impacts.py" in str(part) for part in cmd)
                for cmd in commands
            )
            self.assertTrue(any_labeler)
            self.assertFalse(
                any(
                    isinstance(cmd, (list, tuple)) and "--skip-done" in cmd
                    for cmd in commands
                )
            )


if __name__ == "__main__":
    import unittest

    unittest.main()
