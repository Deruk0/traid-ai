"""Reuse a compatible dashboard, or choose a free local port."""
import json
from pathlib import Path
import socket
import subprocess
import sys
from urllib.request import urlopen
import webbrowser


def choose_port():
    free = []
    for port in (8787, 8788, 8789, 8790, 8791):
        try:
            with urlopen(f"http://127.0.0.1:{port}/api/state", timeout=1) as response:
                state = json.load(response)
            with urlopen(f"http://127.0.0.1:{port}/api/alpaca/state", timeout=1) as response:
                alpaca = json.load(response)
            if ("control_token" in state and "sessions" in state and alpaca.get("mode") == "paper"
                    and all(c in state.get("capabilities", []) for c in
                            ("qwen-local-decisions-v1", "candle-context-v1", "watchdog-v1", "comparison-v1", "accounting-v1", "decision-evidence-v1", "technical-health-v1", "bounded-broker-reads-v1", "session-diagnostics-v1", "wait-review-v1", "trailing-objective-v1", "qwen-thinking-v1", "qwen-model-selection-v2", "broker-decision-handoff-v1", "panel-usability-v1", "qwen-result-delivery-v1", "execution-safety-guards-v1", "optional-max-hold-v1", "terminal-balance-reconciliation-v1", "trade-blocking-log-v1"))):
                return port, True
        except (OSError, ValueError):
            pass
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
                free.append(port)
            except OSError:
                pass
    if not free:
        raise ValueError("Порты 8787–8791 заняты. Закройте старую панель или выберите --port.")
    return free[0], False


def main():
    root = Path(__file__).resolve().parent.parent
    try:
        port, running = choose_port()
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    if running:
        webbrowser.open(f"http://127.0.0.1:{port}/")
        return 0
    return subprocess.call([sys.executable, "-m", "traid_ai", "dashboard",
                            "--port", str(port), "--open"], cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
