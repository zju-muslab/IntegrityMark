
from __future__ import annotations

from dataclasses import asdict, dataclass
import json

from .keyed_schedule import ScheduleConfig


@dataclass(frozen=True)
class IntegrityManifest:
    stream_id: str
    creator_id: str
    segment_count: int
    state_bits: int = 4
    initial_state: int = 0
    sample_rate: int = 16_000
    schedule_version: str = "keyed_prf_v1"
    d_min: int = 512
    d_max: int = 1984
    length_step: int = 32
    signature: str | None = None

    @property
    def schedule_config(self) -> ScheduleConfig:
        return ScheduleConfig(self.d_min, self.d_max, self.length_step)

    def signing_payload(self) -> bytes:
        data = asdict(self)
        data.pop("signature", None)
        return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def validate(self) -> None:
        if not self.stream_id or not self.creator_id:
            raise ValueError("stream_id and creator_id are required")
        if self.segment_count < 0 or self.state_bits <= 0:
            raise ValueError("invalid segment_count or state_bits")
        if not 0 <= self.initial_state < 2**self.state_bits:
            raise ValueError("initial_state is outside the state domain")
        self.schedule_config
