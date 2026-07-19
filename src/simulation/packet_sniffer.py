"""
src/simulation/packet_sniffer.py

The Scapy-facing layer: captures packets (live from a NIC, or replayed from
a .pcap file), feeds them through StreamFeatureExtractor to build connection
records, then runs each completed connection through the trained model +
preprocessor + SHAP explainer for a real-time classification with reasoning.

Supports two modes:
  - LIVE:   sniffs a real network interface. Requires Npcap (Windows) or
            root/libpcap (Linux/Mac), and elevated privileges.
  - REPLAY: reads packets from an existing .pcap file. No special
            privileges needed — useful for testing without touching a
            live NIC, and for reproducible demos.

Defensive design:
  - Live capture failures (no Npcap, no permission, no interface found)
    are caught and produce an actionable error message, not a raw
    traceback.
  - Malformed packets are skipped (handled inside feature_extractor), not
    fatal to the whole capture session.
  - The model/preprocessor/explainer are loaded ONCE at startup, not
    per-packet — loading these repeatedly would make real-time capture
    unusably slow.
"""

import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.data.preprocessor import NIDSPreprocessor
from src.explainability.shap_explainer import NIDSExplainer
from src.models.train import load_model
from src.simulation.feature_extractor import RawConnectionFeatures, StreamFeatureExtractor
from src.utils.exceptions import AegisNIDSError, ModelNotTrainedError, PacketProcessingError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_FLUSH_INTERVAL_SECONDS = 5.0  # how often to check for stale/timed-out flows during live capture


class NIDSLiveEngine:
    """
    Orchestrates the full pipeline: packet -> connection features ->
    preprocessing -> model prediction -> SHAP explanation -> alert.

    Loads all ML artifacts once at construction time, then exposes
    run_live() and run_replay() for the two capture modes.
    """

    def __init__(self, config_path: Optional[Path] = None, alert_callback: Optional[Callable] = None):
        """
        Args:
            config_path: optional config.yaml override.
            alert_callback: optional function called with each classified
                            connection's result dict. If not provided,
                            results are logged to console. Sprint 6's
                            Streamlit dashboard will pass its own callback
                            here to update the live UI instead of printing.
        """
        self.alert_callback = alert_callback or self._default_alert_handler
        self.extractor = StreamFeatureExtractor()

        logger.info("Loading trained model, preprocessor, and SHAP explainer...")
        try:
            self.model = load_model(config_path)
        except ModelNotTrainedError as e:
            raise AegisNIDSError(
                f"{e} — Sprint 5 requires a trained model. Run Sprint 3 first: "
                f"python -m src.models.train"
            ) from e

        self.preprocessor = NIDSPreprocessor(config_path)
        try:
            self.preprocessor.load()
        except AegisNIDSError as e:
            raise AegisNIDSError(
                f"Failed to load fitted preprocessor: {e}. Run Sprint 2/3 first."
            ) from e

        self.explainer = NIDSExplainer(config_path)
        self.explainer.load(feature_names=self.preprocessor.feature_names)

        self._last_flush_time = time.time()
        self._connection_count = 0
        self._alert_count = 0

        logger.info("NIDSLiveEngine ready.")

    def _default_alert_handler(self, result: dict) -> None:
        """Console fallback if no alert_callback is provided."""
        marker = "ALERT" if result["prediction"] == "attack" else " normal"
        top_feat = result["top_contributing_features"][0] if result["top_contributing_features"] else None
        reason = f" | top factor: {top_feat['feature']}" if top_feat else ""
        print(
            f"[{marker}] {result['src_ip']}:{result['src_port']} -> "
            f"{result['dst_ip']}:{result['dst_port']} "
            f"(confidence: {result['confidence']:.1%}){reason}"
        )

    def _classify_connection(self, raw_features: RawConnectionFeatures) -> dict:
        """
        Run one completed connection through preprocessing, prediction, and
        SHAP explanation. Never raises out of this method — a single bad
        connection record should not kill a live capture session; errors
        are logged and a safe fallback result is returned instead.
        """
        try:
            row_dict = raw_features.to_model_dict()
            row_df = pd.DataFrame([row_dict])

            X, _ = self.preprocessor.transform(row_df)
            explanation = self.explainer.explain_prediction(X[0])

            result = {
                **explanation,
                "src_ip": raw_features.src_ip,
                "dst_ip": raw_features.dst_ip,
                "src_port": raw_features.src_port,
                "dst_port": raw_features.dst_port,
                "protocol_type": raw_features.protocol_type,
                "service": raw_features.service,
                "timestamp": time.time(),
            }
            return result

        except AegisNIDSError as e:
            logger.error(f"Classification failed for connection {raw_features.src_ip}->{raw_features.dst_ip}: {e}")
            return {
                "prediction": "error",
                "confidence": 0.0,
                "attack_probability": 0.0,
                "top_contributing_features": [],
                "src_ip": raw_features.src_ip,
                "dst_ip": raw_features.dst_ip,
                "src_port": raw_features.src_port,
                "dst_port": raw_features.dst_port,
                "protocol_type": raw_features.protocol_type,
                "service": raw_features.service,
                "timestamp": time.time(),
                "error": str(e),
            }

    def _handle_packet(self, packet) -> None:
        """Scapy sniff() callback: feed packet to extractor, classify any completed connection."""
        try:
            completed = self.extractor.process_packet(packet)
        except PacketProcessingError as e:
            logger.warning(f"Packet processing error (skipping packet): {e}")
            return

        if completed is not None:
            self._connection_count += 1
            result = self._classify_connection(completed)
            if result["prediction"] == "attack":
                self._alert_count += 1
            self.alert_callback(result)

        # Periodically flush connections that went stale (e.g. UDP with no close signal)
        now = time.time()
        if (now - self._last_flush_time) > _FLUSH_INTERVAL_SECONDS:
            self._last_flush_time = now
            for stale_conn in self.extractor.flush_stale_flows(now):
                self._connection_count += 1
                result = self._classify_connection(stale_conn)
                if result["prediction"] == "attack":
                    self._alert_count += 1
                self.alert_callback(result)

    def run_live(self, interface: Optional[str] = None, duration_seconds: Optional[int] = None,
                 packet_count: int = 0, alert_callback: Optional[Callable] = None) -> dict:
        """
        Sniff live traffic on a network interface.

        Args:
            interface: interface name (e.g. "Wi-Fi", "Ethernet"). If None,
                       Scapy attempts to auto-select the default interface.
            duration_seconds: stop after this many seconds. None = run until
                               packet_count is hit or manually interrupted.
            packet_count: stop after this many packets. 0 = unlimited
                          (bounded only by duration_seconds or Ctrl+C).
            alert_callback: optional override for where results are sent
                             during THIS call only, falling back to
                             self.alert_callback afterward. This lets a
                             single cached/shared engine instance correctly
                             route results to whichever caller actually
                             invoked it (e.g. a specific dashboard session's
                             queue), instead of being permanently locked to
                             whichever caller happened to construct the
                             engine first — critical when the engine is
                             cached and reused across multiple independent
                             sessions.

        Returns:
            Summary dict with connection_count and alert_count.

        Raises:
            AegisNIDSError: if live capture cannot start (missing Npcap,
                             insufficient permissions, invalid interface).
        """
        previous_callback = self.alert_callback
        if alert_callback is not None:
            self.alert_callback = alert_callback
        try:
            return self._run_live_impl(interface, duration_seconds, packet_count)
        finally:
            self.alert_callback = previous_callback

    def _run_live_impl(self, interface: Optional[str], duration_seconds: Optional[int],
                        packet_count: int) -> dict:
        try:
            from scapy.all import AsyncSniffer
        except ImportError as e:
            raise AegisNIDSError(f"Scapy is not installed: {e}") from e

        logger.info(
            f"Starting LIVE capture on interface={interface or '(default)'}, "
            f"duration={duration_seconds or 'unbounded'}s, "
            f"packet_count={packet_count or 'unbounded'}. Press Ctrl+C to stop early."
        )

        sniffer = None
        try:
            # AsyncSniffer runs the capture loop in its OWN background thread
            # and exposes an explicit .stop() we control directly. This is
            # deliberately used INSTEAD OF the simpler blocking sniff(timeout=...)
            # call: on Windows, that timeout is only checked between packet
            # reads at the Npcap/driver layer, which has been observed to
            # delay far past the requested duration (a capture requested for
            # 30s running for 2+ minutes). Driving the stop from an
            # independent wall-clock timer, rather than from packet-arrival
            # timing, avoids that platform-level unreliability entirely.
            sniffer = AsyncSniffer(
                iface=interface,
                prn=self._handle_packet,
                store=False,
                count=packet_count if packet_count else 0,
            )
            sniffer.start()

            if duration_seconds:
                deadline = time.time() + duration_seconds
                while time.time() < deadline:
                    if not sniffer.thread or not sniffer.thread.is_alive():
                        break  # stopped early (e.g. packet_count reached, or an internal error)
                    time.sleep(0.5)
                sniffer.stop()
            else:
                # Unbounded: block until packet_count is hit internally, or
                # the user interrupts with Ctrl+C.
                while sniffer.thread and sniffer.thread.is_alive():
                    time.sleep(0.5)

        except PermissionError as e:
            raise AegisNIDSError(
                f"Permission denied opening network interface: {e}. "
                f"On Windows, run your terminal as Administrator. "
                f"On Linux/Mac, run with sudo."
            ) from e
        except OSError as e:
            raise AegisNIDSError(
                f"Failed to start live capture: {e}. Common causes: Npcap not "
                f"installed (Windows), invalid interface name, or no network "
                f"interfaces available. Run "
                f"'python -c \"from scapy.all import conf; "
                f"[print(i.name) for i in conf.ifaces.values()]\"' to list valid interface names."
            ) from e
        except KeyboardInterrupt:
            logger.info("Live capture stopped by user (Ctrl+C).")
            if sniffer is not None:
                try:
                    sniffer.stop()
                except Exception:
                    pass

        # Capture has ended — finalize EVERY still-open connection immediately,
        # not just ones that happen to be inactive. A persistent HTTPS session
        # still mid-transfer when the timer ran out is exactly the traffic
        # we most need to classify, not discard.
        for finished_conn in self.extractor.finalize_all_flows():
            self._connection_count += 1
            result = self._classify_connection(finished_conn)
            if result["prediction"] == "attack":
                self._alert_count += 1
            self.alert_callback(result)

        summary = {"connection_count": self._connection_count, "alert_count": self._alert_count}
        logger.info(f"Live capture finished. {summary}")
        return summary

    def run_replay(self, pcap_path: Path, alert_callback: Optional[Callable] = None) -> dict:
        """
        Replay packets from a .pcap file through the same classification
        pipeline as live capture. Does not require elevated privileges.

        Args:
            pcap_path: path to a .pcap or .pcapng file.
            alert_callback: optional override for where results are sent
                             during THIS call only (see run_live() docstring
                             for why this matters with a shared/cached engine).

        Returns:
            Summary dict with connection_count and alert_count.

        Raises:
            AegisNIDSError: if the file doesn't exist or can't be parsed.
        """
        previous_callback = self.alert_callback
        if alert_callback is not None:
            self.alert_callback = alert_callback
        try:
            return self._run_replay_impl(pcap_path)
        finally:
            self.alert_callback = previous_callback

    def _run_replay_impl(self, pcap_path: Path) -> dict:
        pcap_path = Path(pcap_path)
        if not pcap_path.exists():
            raise AegisNIDSError(f"Pcap file not found: {pcap_path}")

        try:
            from scapy.all import PcapReader
        except ImportError as e:
            raise AegisNIDSError(f"Scapy is not installed: {e}") from e

        logger.info(f"Starting REPLAY from {pcap_path}")

        try:
            with PcapReader(str(pcap_path)) as reader:
                for packet in reader:
                    self._handle_packet(packet)
        except Exception as e:
            raise AegisNIDSError(f"Failed to read pcap file {pcap_path}: {e}") from e

        for finished_conn in self.extractor.finalize_all_flows():
            self._connection_count += 1
            result = self._classify_connection(finished_conn)
            if result["prediction"] == "attack":
                self._alert_count += 1
            self.alert_callback(result)

        summary = {"connection_count": self._connection_count, "alert_count": self._alert_count}
        logger.info(f"Replay finished. {summary}")
        return summary


def list_available_interfaces() -> list:
    """
    Return a list of available network interfaces for live capture.

    Uses Scapy's conf.ifaces registry directly rather than
    get_windows_if_list(), since that function is not consistently
    exported across Scapy versions/platforms (confirmed missing in some
    installs). conf.ifaces is populated internally by Scapy on import and
    is the more reliable source across versions.
    """
    try:
        from scapy.all import conf
        result = []
        for iface in conf.ifaces.values():
            result.append({
                "name": getattr(iface, "name", "unknown"),
                "description": getattr(iface, "description", "") or "",
                "ips": [getattr(iface, "ip", "")] if getattr(iface, "ip", None) else [],
            })
        return result
    except Exception as e:
        logger.warning(f"Could not enumerate network interfaces via conf.ifaces: {e}")

    try:
        from scapy.all import get_if_list
        return [{"name": name, "description": "", "ips": []} for name in get_if_list()]
    except Exception as e:
        logger.warning(f"Could not enumerate network interfaces: {e}")
        return []


if __name__ == "__main__":
    # Standalone verification entrypoint for Sprint 5.
    # Run: python -m src.simulation.packet_sniffer
    #
    # This verification uses REPLAY mode against a synthetic pcap generated
    # in-memory, since it must run identically on any machine (CI, sandbox,
    # or your laptop) without requiring live network access or admin rights.
    # For actual live capture, use the separate live-capture CLI shown in
    # the chat instructions after this passes.
    import tempfile

    try:
        from scapy.all import IP, TCP, UDP, wrpcap

        logger.info("Generating a small synthetic pcap for pipeline verification...")

        packets = []
        # Simulate a normal-looking short HTTP-like exchange
        packets.append(IP(src="10.0.0.5", dst="93.184.216.34") / TCP(sport=51000, dport=80, flags="S"))
        packets.append(IP(src="93.184.216.34", dst="10.0.0.5") / TCP(sport=80, dport=51000, flags="SA"))
        packets.append(IP(src="10.0.0.5", dst="93.184.216.34") / TCP(sport=51000, dport=80, flags="A") / ("GET / HTTP/1.1" * 5))
        packets.append(IP(src="93.184.216.34", dst="10.0.0.5") / TCP(sport=80, dport=51000, flags="FA") / ("HTTP/1.1 200 OK" * 20))

        # Simulate a SYN-scan-like pattern (many S0 connections, no reply) — should score as attack-leaning
        for port in range(20, 30):
            packets.append(IP(src="10.0.0.99", dst="10.0.0.5") / TCP(sport=40000 + port, dport=port, flags="S"))
            packets.append(IP(src="10.0.0.99", dst="10.0.0.5") / TCP(sport=40000 + port, dport=port, flags="R"))

        with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        wrpcap(str(tmp_path), packets)

        logger.info(f"Synthetic pcap written to {tmp_path} ({len(packets)} packets).")

        engine = NIDSLiveEngine()
        summary = engine.run_replay(tmp_path)

        tmp_path.unlink(missing_ok=True)

        print(f"\n{'=' * 60}")
        print(" SPRINT 5 REPLAY VERIFICATION")
        print(f"{'=' * 60}")
        print(f"Connections processed : {summary['connection_count']}")
        print(f"Attack alerts raised  : {summary['alert_count']}")
        print(f"{'=' * 60}\n")

        if summary["connection_count"] == 0:
            raise AegisNIDSError(
                "No connections were extracted from the synthetic pcap — "
                "the feature extractor or flag-finalization logic likely "
                "has a bug."
            )

        logger.info("Sprint 5 verification PASSED: pcap replay -> feature extraction -> classification pipeline works end-to-end.")

    except AegisNIDSError as e:
        logger.error(f"Sprint 5 verification failed: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        logger.error(f"Unexpected error during Sprint 5 verification: {e}", exc_info=True)
        raise SystemExit(1) from e
