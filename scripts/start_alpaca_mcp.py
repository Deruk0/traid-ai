"""Run the installed official MCP with fixed Paper mode and crypto-data tools."""
import os
from pathlib import Path
import sys
import subprocess


def main():
    root = Path(__file__).resolve().parent.parent
    executable = root / "data" / "alpaca-mcp-env" / "Scripts" / "alpaca-mcp-server.exe"
    if not executable.is_file():
        print("Install Alpaca MCP as described in docs/ALPACA.md.", file=sys.stderr)
        return 1
    if not all(os.environ.get(name) for name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY")):
        print("Alpaca MCP needs Paper Only credentials in ALPACA_API_KEY and ALPACA_SECRET_KEY. "
              "Do not paste credentials into chat.", file=sys.stderr)
        return 1
    environment = dict(os.environ)
    environment.update(ALPACA_PAPER_TRADE="true", ALPACA_TOOLSETS="crypto-data",
                       DATA_API_URL="https://data.alpaca.markets", PYTHONUTF8="1")
    # Windows execve cannot reliably replace a Python process with a pip launcher.
    # Explicit handles preserve the MCP stdio transport through this wrapper.
    return subprocess.call([str(executable), "--transport", "stdio"], env=environment,
                           stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
