from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Intent:
    action: str
    reason: str


class ExecutionSmoke:
    """One explicit virtual round trip for checking execution, not a market signal."""
    version = "paper-smoke-v1"
    def __init__(self, state):
        self.state = state
    def decide(self, history, mid, has_position):
        if has_position:
            return Intent("REQUEST_EXIT", "EXECUTION_TEST")
        if self.state().entries == 0:
            return Intent("REQUEST_ENTRY", "EXECUTION_TEST")
        return Intent("WAIT", "EXECUTION_TEST_COMPLETE")


class DemoMomentum:
    """Mechanical test signal, deliberately not a validated trading strategy."""
    version = "demo-momentum-v1"

    def decide(self, history: list[str], mid: Decimal, has_position: bool) -> Intent:
        if len(history) < 5:
            return Intent("WAIT", "WARMUP")
        # Only earlier observed quotes; the current price isn't in this window.
        mean = sum(Decimal(p) for p in history[-5:]) / 5
        if not has_position and mid > mean * Decimal("1.003"):
            return Intent("REQUEST_ENTRY", "MOMENTUM_ABOVE_MEAN")
        if has_position and mid < mean * Decimal("0.997"):
            return Intent("REQUEST_EXIT", "MOMENTUM_BELOW_MEAN")
        return Intent("WAIT", "NO_SIGNAL")
