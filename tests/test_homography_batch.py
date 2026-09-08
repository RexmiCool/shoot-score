from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.process_homography_batch import process_homography_batch


class HomographyBatchReviewTest(unittest.TestCase):
    def test_homography_batch_triggers_review_for_pending_images(self):
        src = Path("tmp_axis_accepted")
        src.mkdir(parents=True, exist_ok=True)
        image_path = src / "sample.jpg"
        image_path.write_bytes(b"fake")

        label_path = src / "sample_axis_labels.json"
        label_path.write_text(
            json.dumps(
                {
                    "markers": {
                        "left": {"x": 10, "y": 20},
                        "top": {"x": 30, "y": 40},
                        "right": {"x": 50, "y": 60},
                        "bottom": {"x": 70, "y": 80},
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        pending_dir = Path("tmp_homographied_pending")
        accepted_dir = Path("tmp_homographied_accepted")
        rejected_dir = Path("tmp_homographied_rejected")

        with mock.patch("src.process_homography_batch.load_axis_labels", return_value={
            "left": (10, 20),
            "top": (30, 40),
            "right": (50, 60),
            "bottom": (70, 80),
        }), mock.patch("src.process_homography_batch.cv2.imread", return_value=np.zeros((100, 100, 3), dtype=np.uint8)), mock.patch("src.process_homography_batch.flatten_with_markers", return_value=np.zeros((50, 50, 3), dtype=np.uint8)), mock.patch("src.process_homography_batch.save_flatten", return_value=pending_dir / "sample_flat.jpg"), mock.patch("src.process_homography_batch.subprocess.run") as run_mock:
            count = process_homography_batch(
                src=src,
                pending_dir=pending_dir,
                accepted_dir=accepted_dir,
                rejected_dir=rejected_dir,
            )

        self.assertEqual(count, 1)
        self.assertTrue(run_mock.called)
        command = run_mock.call_args[0][0]
        self.assertIn("review_pending.py", " ".join(str(part) for part in command))
        self.assertIn("--accepted", " ".join(str(part) for part in command))


if __name__ == "__main__":
    unittest.main()
