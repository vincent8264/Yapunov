"""Stateful decision policy for safety-relevant sound events."""

from collections.abc import Callable, Mapping
import math
import time

from edge_ai.decision import Decision
from edge_ai.inference.base import InferenceResult


class SoundDecisionPolicy:
    def __init__(
        self,
        thresholds: Mapping[str, float],
        *,
        confirmations: int | Mapping[str, int] = 2,
        hold_seconds: float = 3.0,
        notification_cooldown_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not thresholds:
            raise ValueError("sound decision thresholds must not be empty")
        if any(not 0.0 <= value <= 1.0 for value in thresholds.values()):
            raise ValueError("sound decision thresholds must be between 0 and 1")
        if isinstance(confirmations, bool):
            raise ValueError("confirmations must be an integer or event mapping")
        if isinstance(confirmations, int):
            if confirmations < 1:
                raise ValueError("confirmations must be at least 1")
            confirmations_by_event = dict.fromkeys(thresholds, confirmations)
        else:
            confirmations_by_event = dict(confirmations)
            if set(confirmations_by_event) != set(thresholds):
                raise ValueError("confirmation events must match threshold events")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in confirmations_by_event.values()
            ):
                raise ValueError("confirmations must be positive integers")
        if hold_seconds < 0.0:
            raise ValueError("hold_seconds must be non-negative")
        if notification_cooldown_seconds < 0.0:
            raise ValueError("notification cooldown must be non-negative")
        self.thresholds = dict(thresholds)
        self.confirmations_by_event = confirmations_by_event
        self.hold_seconds = hold_seconds
        self.notification_cooldown_seconds = notification_cooldown_seconds
        self._clock = clock
        self._candidate: str | None = None
        self._candidate_count = 0
        self._active_event: str | None = None
        self._active_confidence: float | None = None
        self._active_until = -math.inf
        self._last_notification: dict[str, float] = {}

    def __call__(self, result: InferenceResult) -> Decision:
        now = self._clock()
        threshold = self.thresholds.get(result.label)
        if threshold is not None and result.confidence >= threshold:
            if result.label == self._candidate:
                self._candidate_count += 1
            else:
                self._candidate = result.label
                self._candidate_count = 1

            if self._candidate_count >= self.confirmations_by_event[result.label]:
                self._active_event = result.label
                self._active_confidence = result.confidence
                self._active_until = now + self.hold_seconds
                last = self._last_notification.get(result.label, -math.inf)
                should_notify = now - last >= self.notification_cooldown_seconds
                if should_notify:
                    self._last_notification[result.label] = now
                return Decision(
                    action="alert",
                    event=result.label,
                    confidence=result.confidence,
                    notify=should_notify,
                )
            return self._held_decision(now)

        self._candidate = None
        self._candidate_count = 0
        return self._held_decision(now)

    def _held_decision(self, now: float) -> Decision:
        if self._active_event is not None and now < self._active_until:
            return Decision(
                action="alert",
                event=self._active_event,
                confidence=self._active_confidence,
            )
        self._active_event = None
        self._active_confidence = None
        return Decision(action="idle")
