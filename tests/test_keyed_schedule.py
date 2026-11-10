import unittest

from utils.keyed_schedule import ScheduleConfig, build_schedule, derive_stream_key


class KeyedScheduleTest(unittest.TestCase):
    def setUp(self):
        self.master_key = bytes(range(32))
        self.config = ScheduleConfig(512, 1984, 32)

    def test_schedule_is_reproducible_and_quantized(self):
        first = build_schedule(self.master_key, "stream-a", "creator", self.config)
        second = build_schedule(self.master_key, "stream-a", "creator", self.config)
        lengths = first.lengths(0, 128)
        self.assertEqual(lengths, second.lengths(0, 128))
        self.assertTrue(all(512 <= value <= 1984 for value in lengths))
        self.assertTrue(all((value - 512) % 32 == 0 for value in lengths))

    def test_stream_identity_separates_schedules(self):
        first = build_schedule(self.master_key, "stream-a", "creator", self.config)
        second = build_schedule(self.master_key, "stream-b", "creator", self.config)
        self.assertNotEqual(first.lengths(0, 32), second.lengths(0, 32))
        self.assertNotEqual(
            derive_stream_key(self.master_key, "stream-a", "creator"),
            derive_stream_key(self.master_key, "stream-b", "creator"),
        )


if __name__ == "__main__":
    unittest.main()
