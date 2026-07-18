"""
src/simulation/feature_extractor.py

Converts a stream of raw packets into NSL-KDD-schema connection records
that the trained model can score.

IMPORTANT DESIGN NOTE — read before trusting these numbers blindly:
NSL-KDD's 41 features fall into three groups, and they are NOT all equally
reconstructable from a generic packet sniffer:

  1. BASIC features (duration, protocol_type, service, flag, src_bytes,
     dst_bytes, land, wrong_fragment, urgent) — fully reconstructable from
     packet headers. Implemented faithfully here.

  2. TRAFFIC features (count, srv_count, *_error_rate, same_srv_rate,
     diff_srv_rate, dst_host_* variants) — reconstructable via sliding
     windows over recently observed connections (2-second time window for
     count/error rates, last-100-connections window for dst_host_*
     features), matching the original KDD feature-generation methodology.
     Implemented here.

  3. CONTENT features (num_failed_logins, logged_in, root_shell,
     num_compromised, num_file_creations, is_guest_login, etc.) — these
     were originally derived by deep-inspecting authenticated session
     payloads (FTP/Telnet/rlogin transcripts) in the original KDD Cup 99
     data generation process. A generic sniffer without protocol-aware
     payload parsers CANNOT faithfully reconstruct these. They are set to
     0 (safe default) rather than guessed, and this limitation is logged
     once at startup so it's never silently forgotten. If genuine content
     inspection is needed later, this is the place to add protocol-specific
     parsers (e.g. FTP response code inspection).

This means: predictions on live/simulated traffic should be trusted for
their traffic-pattern reasoning (which is most of what the SHAP analysis
in Sprint 4 showed actually drives this model — src_bytes, count,
dst_host_srv_count all rank far above any content feature), but treated
as a demonstration rather than a production-grade content-aware NIDS.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field, fields
from typing import Optional

from src.utils.exceptions import PacketProcessingError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CONTENT_FEATURE_WARNING_LOGGED = False

# --- Port -> NSL-KDD service name mapping (approximation of the ~70 KDD service categories) ---
_PORT_SERVICE_MAP = {
    20: "ftp_data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
    37: "time", 53: "domain_u", 69: "tftp_u", 79: "finger", 80: "http",
    88: "kerberos", 109: "pop_2", 110: "pop_3", 111: "sunrpc",
    113: "auth", 119: "nntp", 123: "ntp_u", 135: "loc_srv",
    137: "netbios_ns", 138: "netbios_dgm", 139: "netbios_ssn",
    143: "imap4", 161: "snmp", 162: "snmp", 179: "bgp", 194: "IRC",
    389: "ldap", 443: "http_443", 445: "microsoft-ds", 512: "exec",
    513: "login", 514: "shell", 515: "printer", 520: "efs",
    530: "courier", 531: "chat", 532: "netnews", 540: "uucp",
    543: "klogin", 544: "kshell", 587: "smtp", 636: "ldap",
    993: "imap4", 995: "pop_3", 1433: "sql_net", 1521: "sql_net",
    2049: "nfs", 3306: "sql_net", 3389: "http_443", 5432: "sql_net",
    6000: "X11", 6667: "IRC", 8080: "http",
}
_DEFAULT_SERVICE = "private"   # NSL-KDD's catch-all for unrecognized/ephemeral ports

# --- Sliding window sizes, matching the original KDD feature-generation methodology ---
_TIME_WINDOW_SECONDS = 2.0     # for count/srv_count/*_error_rate (traffic features)
_HOST_WINDOW_SIZE = 100        # for dst_host_* features (host-based traffic features)
_FLOW_TIMEOUT_SECONDS = 60.0   # inactive flow is considered finished after this long


@dataclass
class RawConnectionFeatures:
    """
    A single completed connection's feature values, using the exact NSL-KDD
    column names so it can be fed straight into a pandas DataFrame matching
    the schema src.data.loader.NSL_KDD_COLUMNS expects (minus label/difficulty).
    """
    duration: float = 0.0
    protocol_type: str = "tcp"
    service: str = _DEFAULT_SERVICE
    flag: str = "OTH"
    src_bytes: int = 0
    dst_bytes: int = 0
    land: int = 0
    wrong_fragment: int = 0
    urgent: int = 0
    hot: int = 0
    num_failed_logins: int = 0
    logged_in: int = 0
    num_compromised: int = 0
    root_shell: int = 0
    su_attempted: int = 0
    num_root: int = 0
    num_file_creations: int = 0
    num_shells: int = 0
    num_access_files: int = 0
    num_outbound_cmds: int = 0
    is_host_login: int = 0
    is_guest_login: int = 0
    count: int = 0
    srv_count: int = 0
    serror_rate: float = 0.0
    srv_serror_rate: float = 0.0
    rerror_rate: float = 0.0
    srv_rerror_rate: float = 0.0
    same_srv_rate: float = 0.0
    diff_srv_rate: float = 0.0
    srv_diff_host_rate: float = 0.0
    dst_host_count: int = 0
    dst_host_srv_count: int = 0
    dst_host_same_srv_rate: float = 0.0
    dst_host_diff_srv_rate: float = 0.0
    dst_host_same_src_port_rate: float = 0.0
    dst_host_srv_diff_host_rate: float = 0.0
    dst_host_serror_rate: float = 0.0
    dst_host_srv_serror_rate: float = 0.0
    dst_host_rerror_rate: float = 0.0
    dst_host_srv_rerror_rate: float = 0.0

    # --- metadata, not part of the model schema, useful for the dashboard ---
    src_ip: str = ""
    dst_ip: str = ""
    src_port: int = 0
    dst_port: int = 0

    def to_model_dict(self) -> dict:
        """Return only the 41 NSL-KDD feature fields (excludes IP/port metadata)."""
        metadata_fields = {"src_ip", "dst_ip", "src_port", "dst_port"}
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in metadata_fields
        }


@dataclass
class _ActiveFlow:
    """Internal mutable state for an in-progress connection."""
    initiator_ip: str
    initiator_port: int
    responder_ip: str
    responder_port: int
    protocol: str
    start_time: float
    last_seen: float
    src_bytes: int = 0     # bytes sent BY the initiator (request direction)
    dst_bytes: int = 0     # bytes sent BY the responder (response direction)
    syn_seen: bool = False
    synack_seen: bool = False
    fin_seen: bool = False
    rst_seen: bool = False
    wrong_fragment: int = 0
    urgent: int = 0

    @property
    def src_ip(self) -> str:
        return self.initiator_ip

    @property
    def dst_ip(self) -> str:
        return self.responder_ip

    @property
    def src_port(self) -> int:
        return self.initiator_port

    @property
    def dst_port(self) -> int:
        return self.responder_port


def _map_service(port: int) -> str:
    return _PORT_SERVICE_MAP.get(port, _DEFAULT_SERVICE)


def _map_flag(flow: _ActiveFlow) -> str:
    """
    Approximate NSL-KDD's connection-state 'flag' feature from observed
    TCP handshake/teardown behavior. NSL-KDD flag values: SF, S0, REJ,
    RSTR, RSTO, SH, S1, S2, S3, RSTOS0, OTH.
    """
    if flow.protocol != "tcp":
        return "SF"  # UDP/ICMP connections are conventionally marked SF (no handshake concept)

    if flow.rst_seen and not flow.synack_seen:
        return "REJ" if flow.syn_seen else "RSTOS0"
    if flow.rst_seen:
        return "RSTO"
    if flow.syn_seen and not flow.synack_seen:
        return "S0"       # connection attempt, no reply — classic scan signature
    if flow.syn_seen and flow.synack_seen and flow.fin_seen:
        return "SF"        # normal, established, cleanly closed
    if flow.syn_seen and flow.synack_seen and not flow.fin_seen:
        return "S1"         # established but not yet closed (still active/timed out)
    return "OTH"


class StreamFeatureExtractor:
    """
    Stateful extractor: feed it packets one at a time via process_packet(),
    it groups them into connections, and yields a RawConnectionFeatures
    object each time a connection is finalized (TCP FIN/RST, or timeout for
    UDP/ICMP/stalled TCP).

    Thread-safe: uses a lock around shared state, since Scapy's live sniff()
    callback and any periodic flush/timeout-check logic may run from
    different execution contexts.
    """

    def __init__(self):
        self._active_flows: dict = {}       # flow_key -> _ActiveFlow
        self._recent_connections: deque = deque()   # (timestamp, dst_ip, dst_port, protocol, flag) for time-window stats
        self._host_history: dict = {}       # dst_ip -> deque of recent connection summaries (max _HOST_WINDOW_SIZE)
        self._lock = threading.Lock()

        global _CONTENT_FEATURE_WARNING_LOGGED
        if not _CONTENT_FEATURE_WARNING_LOGGED:
            logger.warning(
                "StreamFeatureExtractor initialized: content-inspection features "
                "(num_failed_logins, logged_in, root_shell, num_compromised, etc.) "
                "are NOT reconstructable from generic packet headers and are fixed "
                "at 0 for all connections. See module docstring for details."
            )
            _CONTENT_FEATURE_WARNING_LOGGED = True

    def _flow_key(self, src_ip: str, dst_ip: str, src_port: int, dst_port: int, protocol: str) -> tuple:
        """
        Canonical, ORDER-INDEPENDENT key: request packets (A->B) and
        response packets (B->A) of the same logical connection must map
        to the same key, or response bytes/flags get silently lost in an
        untracked shadow flow. Sorting the two endpoints achieves this.
        """
        endpoint_a = (src_ip, src_port)
        endpoint_b = (dst_ip, dst_port)
        if endpoint_a <= endpoint_b:
            return (endpoint_a, endpoint_b, protocol)
        return (endpoint_b, endpoint_a, protocol)

    def process_packet(self, packet) -> Optional[RawConnectionFeatures]:
        """
        Process a single Scapy packet. Returns a completed
        RawConnectionFeatures if this packet finalized a connection
        (TCP FIN/RST observed), otherwise returns None (connection still
        in progress). Call flush_stale_flows() periodically to finalize
        timed-out connections (UDP has no explicit close signal).
        """
        try:
            from scapy.layers.inet import IP, TCP, UDP, ICMP
        except ImportError as e:
            raise PacketProcessingError(
                f"Scapy is not installed or failed to import: {e}"
            ) from e

        if IP not in packet:
            return None  # non-IP traffic (ARP, etc.) is out of scope for NSL-KDD-style features

        try:
            ip_layer = packet[IP]
            src_ip, dst_ip = ip_layer.src, ip_layer.dst
            packet_len = int(len(packet))
            now = float(packet.time) if hasattr(packet, "time") else time.time()

            if TCP in packet:
                protocol = "tcp"
                src_port, dst_port = int(packet[TCP].sport), int(packet[TCP].dport)
                tcp_flags = packet[TCP].flags
            elif UDP in packet:
                protocol = "udp"
                src_port, dst_port = int(packet[UDP].sport), int(packet[UDP].dport)
                tcp_flags = None
            elif ICMP in packet:
                protocol = "icmp"
                src_port, dst_port = 0, 0
                tcp_flags = None
            else:
                return None  # unsupported transport layer

        except (AttributeError, IndexError) as e:
            # Malformed or unexpected packet structure — log and skip rather
            # than crash the whole capture session over one bad packet.
            logger.warning(f"Skipping malformed packet: {e}")
            return None

        with self._lock:
            key = self._flow_key(src_ip, dst_ip, src_port, dst_port, protocol)
            flow = self._active_flows.get(key)

            if flow is None:
                # First packet observed for this connection defines the
                # "initiator" (request direction) for the lifetime of the flow.
                flow = _ActiveFlow(
                    initiator_ip=src_ip, initiator_port=src_port,
                    responder_ip=dst_ip, responder_port=dst_port,
                    protocol=protocol, start_time=now, last_seen=now,
                )
                self._active_flows[key] = flow

            flow.last_seen = now
            is_forward = (src_ip == flow.initiator_ip and src_port == flow.initiator_port)
            if is_forward:
                flow.src_bytes += packet_len
            else:
                flow.dst_bytes += packet_len

            if hasattr(packet, "getlayer") and packet.getlayer("Padding") is None and ip_layer.frag:
                flow.wrong_fragment += 1

            connection_finished = False
            if protocol == "tcp" and tcp_flags is not None:
                flag_str = str(tcp_flags)
                if "S" in flag_str and "A" not in flag_str and is_forward:
                    flow.syn_seen = True
                if "S" in flag_str and "A" in flag_str and not is_forward:
                    flow.synack_seen = True
                if "F" in flag_str:
                    flow.fin_seen = True
                    connection_finished = True
                if "R" in flag_str:
                    flow.rst_seen = True
                    connection_finished = True
                if "U" in flag_str:
                    flow.urgent += 1

            if connection_finished:
                return self._finalize_flow(key)

        return None

    def _finalize_flow(self, key: tuple) -> RawConnectionFeatures:
        """
        Must be called while holding self._lock. Removes the flow from
        active tracking, computes traffic/host-based statistics from the
        sliding windows, and returns the completed record.
        """
        flow = self._active_flows.pop(key)
        now = flow.last_seen
        duration = max(0.0, flow.last_seen - flow.start_time)

        # land: same source and destination IP+port (a classic anomaly signature)
        land = int(flow.src_ip == flow.dst_ip and flow.src_port == flow.dst_port)

        flag = _map_flag(flow)
        service = _map_service(flow.dst_port)
        is_error_flag = flag in ("S0", "REJ", "RSTO", "RSTR", "RSTOS0")

        # --- Time-window traffic features (last 2 seconds) ---
        cutoff = now - _TIME_WINDOW_SECONDS
        while self._recent_connections and self._recent_connections[0][0] < cutoff:
            self._recent_connections.popleft()

        same_host_conns = [c for c in self._recent_connections if c[1] == flow.dst_ip]
        same_srv_conns = [c for c in self._recent_connections if c[3] == service]

        count = len(same_host_conns) + 1
        srv_count = len(same_srv_conns) + 1
        serror_rate = (
            sum(1 for c in same_host_conns if c[4] in ("S0",)) / count if count else 0.0
        )
        rerror_rate = (
            sum(1 for c in same_host_conns if c[4] in ("REJ", "RSTO", "RSTR", "RSTOS0")) / count
            if count else 0.0
        )
        srv_serror_rate = (
            sum(1 for c in same_srv_conns if c[4] == "S0") / srv_count if srv_count else 0.0
        )
        srv_rerror_rate = (
            sum(1 for c in same_srv_conns if c[4] in ("REJ", "RSTO", "RSTR", "RSTOS0")) / srv_count
            if srv_count else 0.0
        )
        same_srv_rate = (len(same_srv_conns) + 1) / count if count else 0.0
        diff_srv_rate = 1.0 - same_srv_rate
        distinct_hosts_for_srv = len({c[1] for c in same_srv_conns}) + 1
        srv_diff_host_rate = (
            (distinct_hosts_for_srv - 1) / srv_count if srv_count > 1 else 0.0
        )

        self._recent_connections.append((now, flow.dst_ip, flow.dst_port, service, flag))

        # --- Host-based features (last 100 connections to this destination host) ---
        host_hist = self._host_history.setdefault(flow.dst_ip, deque(maxlen=_HOST_WINDOW_SIZE))
        dst_host_count = len(host_hist) + 1
        dst_host_srv_count = sum(1 for h in host_hist if h["service"] == service) + 1
        dst_host_same_srv_rate = dst_host_srv_count / dst_host_count if dst_host_count else 0.0
        dst_host_diff_srv_rate = 1.0 - dst_host_same_srv_rate
        dst_host_same_src_port_rate = (
            (sum(1 for h in host_hist if h["src_port"] == flow.src_port) + 1) / dst_host_count
            if dst_host_count else 0.0
        )
        distinct_src_hosts = len({h["src_ip"] for h in host_hist if h["service"] == service}) + 1
        dst_host_srv_diff_host_rate = (
            (distinct_src_hosts - 1) / dst_host_srv_count if dst_host_srv_count > 1 else 0.0
        )
        dst_host_serror_rate = (
            sum(1 for h in host_hist if h["flag"] == "S0") / dst_host_count if dst_host_count else 0.0
        )
        dst_host_srv_serror_rate = dst_host_serror_rate  # approximation without full per-service breakdown
        dst_host_rerror_rate = (
            sum(1 for h in host_hist if h["flag"] in ("REJ", "RSTO", "RSTR", "RSTOS0")) / dst_host_count
            if dst_host_count else 0.0
        )
        dst_host_srv_rerror_rate = dst_host_rerror_rate  # same approximation as above

        host_hist.append({
            "src_ip": flow.src_ip, "src_port": flow.src_port,
            "service": service, "flag": flag,
        })

        return RawConnectionFeatures(
            duration=duration,
            protocol_type=flow.protocol,
            service=service,
            flag=flag,
            src_bytes=flow.src_bytes,
            dst_bytes=flow.dst_bytes,
            land=land,
            wrong_fragment=flow.wrong_fragment,
            urgent=flow.urgent,
            count=count,
            srv_count=srv_count,
            serror_rate=serror_rate,
            srv_serror_rate=srv_serror_rate,
            rerror_rate=rerror_rate,
            srv_rerror_rate=srv_rerror_rate,
            same_srv_rate=same_srv_rate,
            diff_srv_rate=diff_srv_rate,
            srv_diff_host_rate=srv_diff_host_rate,
            dst_host_count=dst_host_count,
            dst_host_srv_count=dst_host_srv_count,
            dst_host_same_srv_rate=dst_host_same_srv_rate,
            dst_host_diff_srv_rate=dst_host_diff_srv_rate,
            dst_host_same_src_port_rate=dst_host_same_src_port_rate,
            dst_host_srv_diff_host_rate=dst_host_srv_diff_host_rate,
            dst_host_serror_rate=dst_host_serror_rate,
            dst_host_srv_serror_rate=dst_host_srv_serror_rate,
            dst_host_rerror_rate=dst_host_rerror_rate,
            dst_host_srv_rerror_rate=dst_host_srv_rerror_rate,
            src_ip=flow.src_ip, dst_ip=flow.dst_ip,
            src_port=flow.src_port, dst_port=flow.dst_port,
        )

    def finalize_all_flows(self) -> list:
        """
        Force-finalize EVERY currently active flow immediately, regardless
        of the inactivity timeout. Call this specifically when capture or
        replay is ending — no more packets are coming, so a still-open,
        actively-used connection (e.g. a persistent HTTPS session that was
        mid-transfer when a 30-second capture window ended) must still be
        classified rather than silently dropped. flush_stale_flows() alone
        would miss these, since they don't meet the "inactive" bar.
        """
        finalized = []
        with self._lock:
            keys = list(self._active_flows.keys())
            for key in keys:
                finalized.append(self._finalize_flow(key))
        if finalized:
            logger.info(f"Force-finalized {len(finalized)} still-active flow(s) at capture end.")
        return finalized

    def flush_stale_flows(self, now: Optional[float] = None) -> list:
        """
        Finalize any active flows that have gone quiet for longer than
        _FLOW_TIMEOUT_SECONDS (handles UDP, which has no close signal, and
        TCP connections that stalled without a clean FIN/RST). Call this
        periodically (e.g. every few seconds) during live capture.

        Returns a list of RawConnectionFeatures for all newly finalized flows.
        """
        now = now if now is not None else time.time()
        finalized = []
        with self._lock:
            stale_keys = [
                key for key, flow in self._active_flows.items()
                if (now - flow.last_seen) > _FLOW_TIMEOUT_SECONDS
            ]
            for key in stale_keys:
                finalized.append(self._finalize_flow(key))
        if finalized:
            logger.info(f"Flushed {len(finalized)} stale/inactive flow(s).")
        return finalized

    @property
    def active_flow_count(self) -> int:
        with self._lock:
            return len(self._active_flows)
