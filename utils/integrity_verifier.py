
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from .dual_sliding_window import (
    DualSlidingWindowVerifier,
    VerificationResult,
    VerificationStatus,
)
from .integrity_manifest import IntegrityManifest


@dataclass
class IntegrityVerificationResult:
    status: VerificationStatus
    cross_source_intervals: list[tuple[int, int]] = field(default_factory=list)
    in_source_result: VerificationResult | None = None


class IntegrityVerifier:

    def __init__(
        self,
        dual_verifier: DualSlidingWindowVerifier | None = None,
        manifest_signature_verifier: Callable[[IntegrityManifest], bool] | None = None,
    ):
        self.dual_verifier = dual_verifier or DualSlidingWindowVerifier()
        self.manifest_signature_verifier = manifest_signature_verifier

    def verify(
        self,
        presence_segments: Sequence,
        message_segments: Sequence,
        manifest: IntegrityManifest,
        master_key: bytes,
    ) -> IntegrityVerificationResult:
        if (
            self.manifest_signature_verifier is not None
            and not self.manifest_signature_verifier(manifest)
        ):
            return IntegrityVerificationResult(VerificationStatus.UNVERIFIABLE)
        missing = [
            (int(start), int(end))
            for start, end, present in presence_segments
            if not bool(present) and int(end) > int(start)
        ]
        present = any(bool(segment[2]) for segment in presence_segments)
        if not present:
            return IntegrityVerificationResult(
                VerificationStatus.UNVERIFIABLE,
                cross_source_intervals=missing,
            )

        sequential = self.dual_verifier.verify(message_segments, manifest, master_key)
        status = sequential.status
        if missing or status == VerificationStatus.VIOLATED:
            status = VerificationStatus.VIOLATED
        return IntegrityVerificationResult(status, missing, sequential)
