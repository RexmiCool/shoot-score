from __future__ import annotations

from pathlib import Path
from unittest import TestCase

from src.process_homography_batch import filter_new_images


class HomographyNewOnlyTest(TestCase):
    def test_homography_skips_already_flattened_images(self):
        root = Path("tmp_homography_new_only")
        if root.exists():
            for child in sorted(root.rglob("*"), reverse=True):
                if child.is_file() or child.is_symlink():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()

        src = root / "axis_accepted"
        src.mkdir(parents=True)
        (src / "old.jpg").write_bytes(b"old")
        (src / "new.jpg").write_bytes(b"new")

        pending = root / "homographied" / "pending"
        pending.mkdir(parents=True)
        (pending / "old_flat.jpg").write_bytes(b"flattened")

        accepted = root / "homographied" / "accepted"
        accepted.mkdir(parents=True)
        rejected = root / "homographied" / "rejected"
        rejected.mkdir(parents=True)

        filtered = filter_new_images(src, pending, accepted, rejected)
        self.assertEqual([p.name for p in filtered], ["new.jpg"])


if __name__ == "__main__":
    import unittest
    unittest.main()
