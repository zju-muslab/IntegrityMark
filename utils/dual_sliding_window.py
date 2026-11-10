
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import torch

from .integrity_manifest import IntegrityManifest
from .keyed_schedule import build_schedule


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    VIOLATED = "violated"
    UNVERIFIABLE = "unverifiable"


@dataclass
class VerificationResult:
    status: VerificationStatus
    invalid_points: list[int] = field(default_factory=list)
    matched_schedule_indices: list[int] = field(default_factory=list)
    diagnostics: list[dict] = field(default_factory=list)


def state_to_int(state) -> int:
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().tolist()
    if isinstance(state, int):
        return state
    return int("".join(str(int(bit)) for bit in state), 2)


class DualSlidingWindowVerifier:
    def __init__(
        self,
        *,
        history_size: int = 8,
        lookahead_window: int = 5,
        length_tolerance: int = 64,
        min_history_match_ratio: float = 0.5,
    ):
        if history_size <= 0 or lookahead_window <= 0:
            raise ValueError("window sizes must be positive")
        self.history_size = history_size
        self.lookahead_window = lookahead_window
        self.length_tolerance = length_tolerance
        self.min_history_match_ratio = min_history_match_ratio

    @staticmethod
    def _normalize(segments: Sequence) -> list[tuple[int, int, int]]:
        normalized = []
        for start, end, state in segments:
            start, end = int(start), int(end)
            if end > start:
                normalized.append((start, end, state_to_int(state)))
        return normalized

    @staticmethod
    def _expected_state(manifest: IntegrityManifest, index: int) -> int:
        return (manifest.initial_state + index) % (2**manifest.state_bits)

    def _match_index(
        self, segment, index, manifest, schedule, *, allow_terminal_partial=False
    ) -> bool:
        start, end, state = segment
        if state != self._expected_state(manifest, index):
            return False
        observed_length = end - start
        expected_length = schedule.length_at(index)
        if allow_terminal_partial and observed_length <= expected_length:
            return True
        return abs(observed_length - expected_length) <= self.length_tolerance

    def _synchronize(self, observed, manifest, schedule) -> tuple[int, int] | None:
        if not observed or manifest.segment_count <= 0:
            return None
        history = observed[: self.history_size]
        best_start, best_score = None, -1
        last_start = max(0, manifest.segment_count - len(history))
        for candidate in range(last_start + 1):
            score = sum(
                self._match_index(segment, candidate + offset, manifest, schedule)
                for offset, segment in enumerate(history)
            )
            if score > best_score:
                best_start, best_score = candidate, score
        required = max(1, int(len(history) * self.min_history_match_ratio + 0.999999))
        if best_start is None or best_score < required:
            return None
        return best_start, best_score

    def verify(
        self,
        detected_segments: Sequence,
        manifest: IntegrityManifest,
        master_key: bytes,
    ) -> VerificationResult:
        manifest.validate()
        if manifest.schedule_version != "keyed_prf_v1":
            return VerificationResult(VerificationStatus.UNVERIFIABLE, diagnostics=[{
                "reason": "unsupported_schedule_version",
                "version": manifest.schedule_version,
            }])
        segments = self._normalize(detected_segments)
        if len(segments) < self.history_size:
            return VerificationResult(VerificationStatus.UNVERIFIABLE, diagnostics=[{
                "reason": "insufficient_segments",
                "observed": len(segments),
                "required": self.history_size,
            }])

        schedule = build_schedule(
            master_key, manifest.stream_id, manifest.creator_id, manifest.schedule_config
        )
        sync = self._synchronize(segments, manifest, schedule)
        if sync is None:
            return VerificationResult(
                VerificationStatus.VIOLATED,
                invalid_points=[segments[0][0]],
                diagnostics=[{"reason": "history_sync_failed"}],
            )

        schedule_start, _ = sync
        matched = list(range(schedule_start, schedule_start + min(self.history_size, len(segments))))
        invalid_points: list[int] = []
        diagnostics: list[dict] = []
        observed_index = self.history_size
        last_schedule_index = matched[-1]

        while observed_index < len(segments):
            found = None
            schedule_end = min(
                last_schedule_index + self.lookahead_window,
                manifest.segment_count - 1,
            )
            observed_end = min(observed_index + self.lookahead_window, len(segments))
            for obs in range(observed_index, observed_end):
                for expected in range(last_schedule_index + 1, schedule_end + 1):
                    is_terminal = obs == len(segments) - 1
                    if self._match_index(
                        segments[obs],
                        expected,
                        manifest,
                        schedule,
                        allow_terminal_partial=is_terminal,
                    ):
                        found = (obs, expected)
                        break
                if found is not None:
                    break

            if found is None:
                start = segments[observed_index][0]
                invalid_points.append(start)
                diagnostics.append({
                    "reason": "dual_check_failed",
                    "position": start,
                    "after_schedule_index": last_schedule_index,
                })
                remaining = segments[observed_index:]
                resync = self._synchronize(remaining, manifest, schedule)
                if resync is None:
                    break
                new_start, _ = resync
                consume = min(self.history_size, len(remaining))
                matched.extend(range(new_start, new_start + consume))
                last_schedule_index = new_start + consume - 1
                observed_index += consume
                continue

            obs, expected = found
            if obs > observed_index or expected > last_schedule_index + 1:
                invalid_points.append(segments[observed_index][0])
                diagnostics.append({
                    "reason": "sequence_discontinuity",
                    "position": segments[observed_index][0],
                    "matched_schedule_index": expected,
                })
            matched.append(expected)
            last_schedule_index = expected
            observed_index = obs + 1

        invalid_points = sorted(set(invalid_points))
        status = VerificationStatus.VIOLATED if invalid_points else VerificationStatus.VERIFIED
        return VerificationResult(status, invalid_points, matched, diagnostics)
