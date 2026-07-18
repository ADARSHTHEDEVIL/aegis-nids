"""
run_live_capture.py

Standalone CLI entrypoint for LIVE network capture (as opposed to the
replay-mode verification baked into src/simulation/packet_sniffer.py's
__main__ block, which intentionally avoids touching a real NIC so it can
run anywhere).

Usage:
    python run_live_capture.py                          # auto-select interface, run until Ctrl+C
    python run_live_capture.py --interface "Wi-Fi"       # capture on a specific interface
    python run_live_capture.py --duration 60             # stop after 60 seconds
    python run_live_capture.py --list-interfaces         # list available interfaces and exit

REQUIRES:
  - Npcap installed (Windows) — see https://npcap.com
  - Terminal running as Administrator (Windows) or with sudo (Linux/Mac)
  - A trained model + preprocessor already in src/models/registry/
    (run `python -m src.models.train` first if you haven't)
"""

import argparse
import sys

from src.simulation.packet_sniffer import NIDSLiveEngine, list_available_interfaces
from src.utils.exceptions import AegisNIDSError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aegis-NIDS live packet capture")
    parser.add_argument(
        "--interface", type=str, default=None,
        help="Network interface name to sniff (e.g. 'Wi-Fi', 'Ethernet'). "
             "Default: let Scapy auto-select.",
    )
    parser.add_argument(
        "--duration", type=int, default=None,
        help="Stop capture after this many seconds. Default: run until Ctrl+C.",
    )
    parser.add_argument(
        "--packet-count", type=int, default=0,
        help="Stop after this many packets (0 = unbounded). Default: 0.",
    )
    parser.add_argument(
        "--list-interfaces", action="store_true",
        help="List available network interfaces and exit.",
    )
    args = parser.parse_args()

    if args.list_interfaces:
        interfaces = list_available_interfaces()
        if not interfaces:
            print("No interfaces found, or Scapy/Npcap is not set up correctly.")
            return 1
        print("Available network interfaces:\n")
        for iface in interfaces:
            ip_str = ", ".join(ip for ip in iface["ips"] if ip) or "(no IP assigned)"
            print(f"  Name        : {iface['name']}")
            print(f"  Description : {iface['description'] or '(none)'}")
            print(f"  IP address  : {ip_str}")
            print(f"  --interface value to use: \"{iface['name']}\"")
            print()
        print("Tip: pick the entry whose IP address matches your active network "
              "(usually starts with 192.168.x.x or 10.x.x.x), and use its "
              "Name value with --interface.")
        return 0

    try:
        engine = NIDSLiveEngine()
    except AegisNIDSError as e:
        logger.error(f"Failed to initialize NIDS engine: {e}")
        return 1

    try:
        summary = engine.run_live(
            interface=args.interface,
            duration_seconds=args.duration,
            packet_count=args.packet_count,
        )
    except AegisNIDSError as e:
        logger.error(f"Live capture failed: {e}")
        return 1

    print(f"\nCapture complete. Connections analyzed: {summary['connection_count']}, "
          f"Attack alerts: {summary['alert_count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
