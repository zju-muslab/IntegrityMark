import unittest

from omegaconf import OmegaConf
import torch

from utils.dual_sliding_window import DualSlidingWindowVerifier, VerificationStatus
from utils.wm_process import MsgGenerator


class MessageGeneratorTest(unittest.TestCase):
    def test_keyed_batch_has_independent_manifests_and_segments(self):
        config = OmegaConf.create({
            "fix": {"length": 0},
            "temp": {"length": 0, "interval": 512, "interval_range": False},
            "val": {
                "length": 4,
                "schedule_version": "keyed_prf_v1",
                "d_min": 512,
                "d_max": 1984,
                "length_step": 32,
                "interval": 512,
                "interval_range": [1, 2],
            },
        })
        generator = MsgGenerator(config, master_key=b"x" * 32)
        result = generator.msg_generate(
            torch.zeros(2, 1, 8000),
            return_seg=True,
            stream_ids=["a", "b"],
            creator_ids=["creator", "creator"],
        )
        self.assertEqual(result[0].shape, (2, 4, 8000))
        self.assertEqual(len(result[4][1]), 2)
        self.assertEqual([m.stream_id for m in generator.last_manifests], ["a", "b"])
        self.assertNotEqual(result[4][1][0], result[4][1][1])

    def test_published_legacy_config_is_migrated(self):
        config = OmegaConf.create({
            "fix": {"length": 0},
            "temp": {"length": 0, "interval": 512, "interval_range": False},
            "val": {
                "length": 4,
                "d_min": 512,
                "d_max": 2064,
                "increment": 32,
                "interval": 512,
                "interval_range": [1, 2],
            },
        })
        generator = MsgGenerator(config, master_key=b"x" * 32)
        self.assertEqual(generator.schedule_version, "keyed_prf_v1")
        self.assertEqual(generator.d_max, 1984)

    def test_generated_segments_verify_with_their_manifest(self):
        config = OmegaConf.create({
            "fix": {"length": 0},
            "temp": {"length": 0, "interval": 512, "interval_range": False},
            "val": {
                "length": 4,
                "schedule_version": "keyed_prf_v1",
                "d_min": 512,
                "d_max": 1984,
                "length_step": 32,
                "interval": 512,
                "interval_range": [1, 2],
            },
        })
        master_key = b"x" * 32
        generator = MsgGenerator(config, master_key=master_key)
        result = generator.msg_generate(
            torch.zeros(1, 1, 50_000),
            return_seg=True,
            stream_ids=["stream"],
            creator_ids=["creator"],
        )
        verification = DualSlidingWindowVerifier().verify(
            result[4][1], generator.last_manifests[0], master_key
        )
        self.assertEqual(verification.status, VerificationStatus.VERIFIED)


if __name__ == "__main__":
    unittest.main()
