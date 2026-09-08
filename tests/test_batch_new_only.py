from __future__ import annotations

from pathlib import Path
from unittest import TestCase, mock

from src.process_axis_batch import process_axis_batch


class BatchNewOnlyTest(TestCase):
    def test_process_axis_batch_keeps_only_unprocessed_images(self):
        root = Path("tmp_new_only_axis")
        if root.exists():
            for child in sorted(root.rglob("*"), reverse=True):
                if child.is_file() or child.is_symlink():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
        src = root / "src"
        src.mkdir(parents=True)
        (src / "new.jpg").write_bytes(b"new")
        (src / "old.jpg").write_bytes(b"old")

        pending = root / "pending"
        accepted = root / "accepted"
        rejected = root / "rejected"
        accepted.mkdir(parents=True)
        rejected.mkdir(parents=True)
        (accepted / "old.jpg").write_bytes(b"already_ok")

        with mock.patch("src.process_axis_batch.subprocess.run") as run_mock:
            count = process_axis_batch(src=src, pending_dir=pending, accepted_dir=accepted, weights=None)

        self.assertEqual(count, 1)
        self.assertEqual(sorted(p.name for p in pending.iterdir()), ["new.jpg"])
        self.assertTrue(run_mock.called)


if __name__ == "__main__":
    import unittest
    unittest.main()
