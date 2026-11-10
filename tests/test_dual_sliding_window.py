import unittest

from utils.dual_sliding_window import DualSlidingWindowVerifier, VerificationStatus
from utils.integrity_manifest import IntegrityManifest
from utils.keyed_schedule import ScheduleConfig, build_schedule


class DualSlidingWindowTest(unittest.TestCase):
    def setUp(self):
        self.master_key = b"test-only-master-key-32-bytes!!"
        self.config = ScheduleConfig(512, 1984, 32)
        self.schedule = build_schedule(
            self.master_key, "stream-a", "creator-a", self.config
        )
        self.initial_state = self.schedule.initial_state(4)
        self.manifest = IntegrityManifest(
            stream_id="stream-a",
            creator_id="creator-a",
            segment_count=40,
            initial_state=self.initial_state,
        )
        self.verifier = DualSlidingWindowVerifier()
        self.segments = self._segments(40)

    def _segments(self, count):
        result = []
        position = 0
        for index in range(count):
            length = self.schedule.length_at(index)
            state = (self.initial_state + index) % 16
            bits = [int(bit) for bit in f"{state:04b}"]
            result.append([position, position + length, bits])
            position += length
        return result

    def test_authentic_sequence_passes(self):
        result = self.verifier.verify(self.segments, self.manifest, self.master_key)
        self.assertEqual(result.status, VerificationStatus.VERIFIED)
        self.assertEqual(result.invalid_points, [])

    def test_deleted_segment_is_reported(self):
        attacked = self.segments[:15] + self.segments[16:]
        result = self.verifier.verify(attacked, self.manifest, self.master_key)
        self.assertEqual(result.status, VerificationStatus.VIOLATED)
        self.assertTrue(result.invalid_points)

    def test_wrong_length_with_correct_state_is_reported(self):
        attacked = [list(segment) for segment in self.segments]
        attacked[15] = [attacked[15][0], attacked[15][1] + 96, attacked[15][2]]
        result = self.verifier.verify(attacked, self.manifest, self.master_key)
        self.assertEqual(result.status, VerificationStatus.VIOLATED)
        self.assertTrue(result.invalid_points)

    def test_length_error_inside_tolerance_passes(self):
        attacked = [list(segment) for segment in self.segments]
        attacked[15] = [attacked[15][0], attacked[15][1] + 64, attacked[15][2]]
        result = self.verifier.verify(attacked, self.manifest, self.master_key)
        self.assertEqual(result.status, VerificationStatus.VERIFIED)

    def test_wrong_key_cannot_synchronize(self):
        result = self.verifier.verify(self.segments, self.manifest, b"wrong-key")
        self.assertEqual(result.status, VerificationStatus.VIOLATED)
        self.assertEqual(result.invalid_points, [0])


if __name__ == "__main__":
    unittest.main()
