
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
_HASH = hashlib.sha256
_HASH_LEN = _HASH().digest_size


def _as_bytes(value: str | bytes) -> bytes:
    return value if isinstance(value, bytes) else value.encode("utf-8")


def hkdf_sha256(
    input_key_material: bytes,
    *,
    salt: bytes = b"IntegrityMark-HKDF-v1",
    info: bytes = b"",
    length: int = 32,
) -> bytes:
    if length <= 0 or length > 255 * _HASH_LEN:
        raise ValueError("invalid HKDF output length")
    prk = hmac.new(salt, input_key_material, _HASH).digest()
    output = bytearray()
    previous = b""
    counter = 1
    while len(output) < length:
        previous = hmac.new(prk, previous + info + bytes([counter]), _HASH).digest()
        output.extend(previous)
        counter += 1
    return bytes(output[:length])


def derive_stream_key(
    master_key: bytes,
    stream_id: str | bytes,
    creator_id: str | bytes,
) -> bytes:
    if not master_key:
        raise ValueError("master_key must not be empty")
    sid = _as_bytes(stream_id)
    creator = _as_bytes(creator_id)
    info = (
        b"IntegrityMark/stream-key/v1\x00"
        + len(sid).to_bytes(4, "big")
        + sid
        + len(creator).to_bytes(4, "big")
        + creator
    )
    return hkdf_sha256(master_key, info=info)


@dataclass(frozen=True)
class ScheduleConfig:
    d_min: int = 512
    d_max: int = 1984
    length_step: int = 32

    def __post_init__(self) -> None:
        if self.d_min <= 0 or self.d_max < self.d_min:
            raise ValueError("expected 0 < d_min <= d_max")
        if self.length_step <= 0:
            raise ValueError("length_step must be positive")
        if (self.d_max - self.d_min) % self.length_step:
            raise ValueError("d_max - d_min must be divisible by length_step")

    @property
    def bucket_count(self) -> int:
        return (self.d_max - self.d_min) // self.length_step + 1


class KeyedLengthSchedule:

    def __init__(self, stream_key: bytes, config: ScheduleConfig):
        if not stream_key:
            raise ValueError("stream_key must not be empty")
        self.stream_key = stream_key
        self.config = config

    def _prf(self, domain: bytes, index: int) -> int:
        if index < 0:
            raise ValueError("segment index must be non-negative")
        message = domain + index.to_bytes(8, "big", signed=False)
        return int.from_bytes(hmac.new(self.stream_key, message, _HASH).digest(), "big")

    def length_at(self, index: int) -> int:
        bucket = self._prf(b"length\x00", index) % self.config.bucket_count
        return self.config.d_min + self.config.length_step * bucket

    def lengths(self, start_index: int, count: int) -> list[int]:
        if count < 0:
            raise ValueError("count must be non-negative")
        return [self.length_at(i) for i in range(start_index, start_index + count)]

    def initial_state(self, state_bits: int) -> int:
        if state_bits <= 0:
            raise ValueError("state_bits must be positive")
        return self._prf(b"initial-state\x00", 0) % (2**state_bits)


def build_schedule(
    master_key: bytes,
    stream_id: str | bytes,
    creator_id: str | bytes,
    config: ScheduleConfig,
) -> KeyedLengthSchedule:
    return KeyedLengthSchedule(
        derive_stream_key(master_key, stream_id, creator_id),
        config,
    )
