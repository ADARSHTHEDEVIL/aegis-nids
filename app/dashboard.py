"""
app/dashboard.py

Aegis-NIDS Streamlit dashboard — the final integration point tying
together every previous sprint:
  - Sprint 2/3: preprocessor + trained XGBoost model
  - Sprint 4: SHAP explainability
  - Sprint 5: live/replay packet capture engine

Run with:
    streamlit run app/dashboard.py

DESIGN NOTE — why a background thread + queue:
Streamlit's execution model reruns the ENTIRE script top-to-bottom on
every interaction. A long-running blocking call like Scapy's sniff()
would freeze the whole app. Instead, live capture runs in a daemon
background thread that pushes classified connection results onto a
thread-safe queue.Queue; the main Streamlit thread polls that queue on
each rerun and updates st.session_state, which IS safe to mutate from
the main thread. The background thread never touches st.session_state
directly — cross-thread Streamlit session-state access is not safe and
would risk race conditions or crashes.

VISUAL DESIGN NOTE:
Styled as a signal-monitoring console (dark slate-navy, teal/red signal
accents, monospace data readouts) rather than a default light SaaS
dashboard — grounded in how real network-operations/SOC consoles present
live traffic, not a generic "hacker green terminal" cliche. Custom CSS
is injected once via st.markdown; the connection feed is hand-built HTML
(not st.dataframe) for full control over the console aesthetic.
"""

import html
import queue
import sys
import threading
import time
from pathlib import Path

# streamlit run app/dashboard.py executes this file directly (not as a
# package), so Python only adds app/ to sys.path — NOT the project root
# where src/ lives. Every other entrypoint in this project is run via
# `python -m src.xxx.yyy`, which handles this automatically; Streamlit
# does not support that invocation style, so the fix has to live here.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.simulation.packet_sniffer import NIDSLiveEngine, list_available_interfaces
from src.utils.exceptions import AegisNIDSError, ModelNotTrainedError
from src.utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="Aegis-NIDS", page_icon="🛰️", layout="wide")

# --------------------------------------------------------------------- #
# Design tokens — signal-console palette. One accent for "signal/normal"
# (teal), one reserved strictly for genuine attack alerts (red) so the
# alert color never gets diluted by decorative use elsewhere.
# --------------------------------------------------------------------- #
_BG = "#0B1220"
_PANEL = "#131B2E"
_PANEL_RAISED = "#182238"
_BORDER = "#232D42"
_TEXT = "#E7ECF5"
_TEXT_MUTED = "#8A93A6"
_SIGNAL = "#00D9C0"    # teal — normal traffic, idle/live state
_ALERT = "#FF5C5C"     # red — reserved for attack classifications only
_AMBER = "#F5A623"     # used sparingly for warnings/errors, not attacks


def _inject_console_css() -> None:
    st.markdown(f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] {{
            font-family: 'IBM Plex Sans', sans-serif;
        }}
        .stApp {{
            background: {_BG};
            color: {_TEXT};
        }}
        [data-testid="stSidebar"] {{
            background: {_PANEL};
            border-right: 1px solid {_BORDER};
        }}
        [data-testid="stSidebar"] * {{
            color: {_TEXT};
        }}
        h1, h2, h3 {{
            font-family: 'Space Grotesk', sans-serif !important;
            letter-spacing: -0.01em;
        }}
        hr {{
            border-color: {_BORDER} !important;
        }}
        .stButton > button {{
            background: {_SIGNAL};
            color: {_BG};
            border: none;
            border-radius: 6px;
            font-weight: 600;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            font-size: 0.8rem;
        }}
        .stButton > button:hover {{
            background: #00F0D4;
            color: {_BG};
        }}
        [data-testid="stFileUploader"] section {{
            background: {_PANEL_RAISED};
            border: 1px dashed {_BORDER};
        }}

        /* --- Header / signature signal indicator --- */
        .ops-header {{
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 4px;
        }}
        .ops-pulse-wrap {{
            position: relative;
            width: 22px;
            height: 22px;
            flex-shrink: 0;
        }}
        .ops-pulse-dot {{
            position: absolute;
            top: 6px; left: 6px;
            width: 10px; height: 10px;
            border-radius: 50%;
            background: var(--pulse-color);
        }}
        .ops-pulse-ring {{
            position: absolute;
            top: 0; left: 0;
            width: 22px; height: 22px;
            border-radius: 50%;
            border: 2px solid var(--pulse-color);
            opacity: 0;
        }}
        .ops-pulse-active .ops-pulse-ring {{
            animation: ops-pulse 1.6s ease-out infinite;
        }}
        @keyframes ops-pulse {{
            0%   {{ transform: scale(0.4); opacity: 0.9; }}
            100% {{ transform: scale(1.8); opacity: 0; }}
        }}
        @media (prefers-reduced-motion: reduce) {{
            .ops-pulse-active .ops-pulse-ring {{ animation: none; opacity: 0.4; }}
        }}
        .ops-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.6rem;
            font-weight: 700;
            color: {_TEXT};
            line-height: 1.1;
        }}
        .ops-subtitle {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.78rem;
            color: {_TEXT_MUTED};
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}

        /* --- Metric cards --- */
        .ops-metric {{
            background: {_PANEL};
            border: 1px solid {_BORDER};
            border-radius: 8px;
            padding: 14px 16px;
        }}
        .ops-metric-label {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.7rem;
            color: {_TEXT_MUTED};
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 6px;
        }}
        .ops-metric-value {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.8rem;
            font-weight: 700;
            color: {_TEXT};
        }}

        /* --- Connection feed --- */
        .ops-feed {{
            max-height: 360px;
            overflow-y: auto;
            border: 1px solid {_BORDER};
            border-radius: 8px;
            background: {_PANEL};
        }}
        .ops-row {{
            display: grid;
            grid-template-columns: 88px 1fr 24px 1fr 70px 130px;
            align-items: center;
            gap: 10px;
            padding: 9px 14px;
            border-bottom: 1px solid {_BORDER};
            font-size: 0.85rem;
        }}
        .ops-row:last-child {{ border-bottom: none; }}
        .ops-row-attack {{ background: rgba(255, 92, 92, 0.08); }}
        .ops-badge {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            padding: 3px 8px;
            border-radius: 4px;
            text-align: center;
        }}
        .ops-badge-normal {{ background: rgba(0, 217, 192, 0.15); color: {_SIGNAL}; }}
        .ops-badge-attack {{ background: rgba(255, 92, 92, 0.18); color: {_ALERT}; }}
        .ops-badge-error {{ background: rgba(245, 166, 35, 0.18); color: {_AMBER}; }}
        .ops-mono {{
            font-family: 'IBM Plex Mono', monospace;
            color: {_TEXT};
            font-size: 0.82rem;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .ops-arrow {{ color: {_TEXT_MUTED}; text-align: center; }}
        .ops-conf {{
            font-family: 'IBM Plex Mono', monospace;
            color: {_TEXT_MUTED};
            font-size: 0.8rem;
        }}
        .ops-proto {{
            font-family: 'IBM Plex Mono', monospace;
            color: {_TEXT_MUTED};
            font-size: 0.75rem;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .ops-feed-header {{
            display: grid;
            grid-template-columns: 88px 1fr 24px 1fr 70px 130px;
            gap: 10px;
            padding: 8px 14px;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.68rem;
            color: {_TEXT_MUTED};
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-bottom: 1px solid {_BORDER};
        }}
        .ops-empty {{
            padding: 40px 20px;
            text-align: center;
            color: {_TEXT_MUTED};
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.85rem;
        }}
    </style>
    """, unsafe_allow_html=True)


# --------------------------------------------------------------------- #
# Cached resource loading — model/preprocessor/SHAP explainer load ONCE
# per session, not on every Streamlit rerun (which happens on nearly
# every UI interaction). Loading these repeatedly would make the
# dashboard painfully slow and would reconstruct SHAP's TreeExplainer
# constantly for no reason.
# --------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading trained model, preprocessor, and SHAP explainer...")
def load_engine(_result_callback):
    """
    Load the NIDSLiveEngine once and cache it for the session.
    The leading underscore on _result_callback tells Streamlit's cache
    not to try hashing the callback function (it's not meaningfully
    hashable, and doesn't need to be — the engine itself is what we're
    caching).
    """
    return NIDSLiveEngine(alert_callback=_result_callback)


class CaptureController:
    """
    Owns the background capture thread and the thread-safe queue results
    flow through. Lives in st.session_state so it survives Streamlit's
    script reruns without losing capture state.
    """

    def __init__(self):
        self.result_queue: queue.Queue = queue.Queue()
        self.thread = None
        self.is_running: bool = False
        self.error: str = None
        self.summary: dict = None

    def _queue_callback(self, result: dict) -> None:
        """Called from the background thread — must ONLY touch the queue, nothing else."""
        self.result_queue.put(result)

    def start_live(self, engine: NIDSLiveEngine, interface: str, duration: int) -> None:
        if self.is_running:
            return

        # Set this BEFORE starting the thread, not inside it. Streamlit calls
        # st.rerun() immediately after this returns; if the flag were only
        # set from within _run(), the thread might not have been scheduled
        # yet by the time the page reruns and checks is_running, causing the
        # UI to show the Start button again instead of the in-progress state
        # — capture could be silently running (or silently failing) with no
        # visible feedback at all.
        self.is_running = True
        self.error = None

        def _run():
            try:
                self.summary = engine.run_live(interface=interface, duration_seconds=duration)
            except AegisNIDSError as e:
                self.error = str(e)
                logger.error(f"Live capture thread failed: {e}")
            finally:
                self.is_running = False

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()

    def start_replay(self, engine: NIDSLiveEngine, pcap_path: Path) -> None:
        if self.is_running:
            return

        self.is_running = True
        self.error = None

        def _run():
            try:
                self.summary = engine.run_replay(pcap_path)
            except AegisNIDSError as e:
                self.error = str(e)
                logger.error(f"Replay thread failed: {e}")
            finally:
                self.is_running = False

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()

    def drain_queue(self) -> list:
        """Pull all currently available results off the queue without blocking."""
        results = []
        while True:
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return results


def _init_session_state():
    if "controller" not in st.session_state:
        st.session_state.controller = CaptureController()
    if "results" not in st.session_state:
        st.session_state.results = []  # list of classified connection dicts


def _plotly_dark_theme(fig: go.Figure) -> go.Figure:
    """Apply the console's dark palette consistently across all charts."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_PANEL,
        plot_bgcolor=_PANEL,
        font=dict(family="IBM Plex Sans, sans-serif", color=_TEXT),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _render_header(is_running: bool, last_prediction: str = None) -> None:
    status_color = _ALERT if last_prediction == "attack" else _SIGNAL
    pulse_class = "ops-pulse-active" if is_running else ""
    status_text = "LIVE CAPTURE" if is_running else "IDLE"
    st.markdown(f"""
    <div class="ops-header">
        <div class="ops-pulse-wrap {pulse_class}" style="--pulse-color:{status_color}">
            <div class="ops-pulse-ring"></div>
            <div class="ops-pulse-dot"></div>
        </div>
        <div>
            <div class="ops-title">AEGIS&#8209;NIDS</div>
            <div class="ops-subtitle">AI Network Intrusion Detection &middot; {status_text}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _metric_card(label: str, value: str, accent: str = None) -> str:
    style = f'style="color:{accent};"' if accent else ""
    return f"""
    <div class="ops-metric">
        <div class="ops-metric-label">{html.escape(label)}</div>
        <div class="ops-metric-value" {style}>{html.escape(str(value))}</div>
    </div>
    """


def _render_feed(results: list) -> None:
    if not results:
        st.markdown(
            '<div class="ops-feed"><div class="ops-empty">'
            'No connections analyzed yet. Start a capture from the console on the left.'
            '</div></div>',
            unsafe_allow_html=True,
        )
        return

    header = (
        '<div class="ops-feed-header">'
        '<span>Status</span><span>Source</span><span></span>'
        '<span>Destination</span><span>Conf.</span><span>Protocol / Service</span>'
        '</div>'
    )

    rows = []
    for r in reversed(results):  # newest first
        prediction = r.get("prediction", "error")
        badge_class = {"normal": "ops-badge-normal", "attack": "ops-badge-attack"}.get(
            prediction, "ops-badge-error"
        )
        row_class = "ops-row-attack" if prediction == "attack" else ""
        src = html.escape(f"{r.get('src_ip', '?')}:{r.get('src_port', '?')}")
        dst = html.escape(f"{r.get('dst_ip', '?')}:{r.get('dst_port', '?')}")
        proto = html.escape(f"{r.get('protocol_type', '?')} / {r.get('service', '?')}")
        conf = f"{r.get('confidence', 0):.0%}"
        rows.append(f"""
        <div class="ops-row {row_class}">
            <span class="ops-badge {badge_class}">{prediction.upper()}</span>
            <span class="ops-mono">{src}</span>
            <span class="ops-arrow">&rarr;</span>
            <span class="ops-mono">{dst}</span>
            <span class="ops-conf">{conf}</span>
            <span class="ops-proto">{proto}</span>
        </div>
        """)

    st.markdown(f'<div class="ops-feed">{header}{"".join(rows)}</div>', unsafe_allow_html=True)


def _render_shap_panel(result: dict) -> None:
    st.markdown("### Why was this connection classified this way?")
    if result.get("prediction") == "error":
        st.error(f"Classification failed for this connection: {result.get('error', 'unknown error')}")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        prediction = result["prediction"]
        color = _ALERT if prediction == "attack" else _SIGNAL
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["confidence"] * 100,
            number={"suffix": "%", "font": {"color": _TEXT}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": _TEXT_MUTED},
                "bar": {"color": color},
                "bgcolor": _PANEL_RAISED,
                "borderwidth": 1,
                "bordercolor": _BORDER,
            },
            title={"text": f"Confidence ({prediction.upper()})", "font": {"color": _TEXT, "size": 14}},
        ))
        fig.update_layout(height=230)
        st.plotly_chart(_plotly_dark_theme(fig), use_container_width=True)

        st.markdown(f"""
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.85rem; color:{_TEXT_MUTED}; line-height:1.9;">
            <div><span style="color:{_TEXT};">Attack probability:</span> {result['attack_probability']:.1%}</div>
            <div><span style="color:{_TEXT};">Connection:</span> {html.escape(result['src_ip'])}:{result['src_port']}
                &rarr; {html.escape(result['dst_ip'])}:{result['dst_port']}</div>
            <div><span style="color:{_TEXT};">Protocol / Service:</span> {html.escape(result['protocol_type'])} / {html.escape(result['service'])}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        features = result.get("top_contributing_features", [])
        if not features:
            st.info("No feature attribution available for this connection.")
            return
        feat_df = pd.DataFrame(features)
        feat_df["direction"] = feat_df["shap_contribution"].apply(
            lambda v: "Toward ATTACK" if v > 0 else "Toward NORMAL"
        )
        fig = px.bar(
            feat_df.sort_values("shap_contribution"),
            x="shap_contribution", y="feature", color="direction", orientation="h",
            color_discrete_map={"Toward ATTACK": _ALERT, "Toward NORMAL": _SIGNAL},
            labels={"shap_contribution": "SHAP contribution", "feature": ""},
        )
        fig.update_layout(height=300, showlegend=True, legend_title_text="")
        st.plotly_chart(_plotly_dark_theme(fig), use_container_width=True)


def main():
    _inject_console_css()
    _init_session_state()
    controller: CaptureController = st.session_state.controller

    # --- Load engine (cached) ---
    try:
        engine = load_engine(controller._queue_callback)
    except ModelNotTrainedError:
        _render_header(False)
        st.error(
            "No trained model found. Run Sprint 3 training first from a terminal:\n\n"
            "`python -m src.models.train`\n\nthen reload this dashboard."
        )
        st.stop()
    except AegisNIDSError as e:
        _render_header(False)
        st.error(f"Failed to initialize the NIDS engine: {e}")
        st.stop()

    results = st.session_state.results
    last_prediction = results[-1]["prediction"] if results else None
    _render_header(controller.is_running, last_prediction)
    st.caption("Real-time traffic classification with explainable AI — NSL-KDD trained XGBoost + SHAP")

    # --- Sidebar: capture controls ---
    with st.sidebar:
        st.markdown("#### Capture Console")
        mode = st.radio("Mode", ["Live Capture", "Replay .pcap"], disabled=controller.is_running,
                         label_visibility="collapsed")

        if mode == "Live Capture":
            try:
                interfaces = list_available_interfaces()
            except Exception as e:
                interfaces = []
                st.warning(f"Could not list interfaces: {e}")

            iface_names = [i["name"] for i in interfaces if i.get("ips") and any(i["ips"])] or \
                          [i["name"] for i in interfaces]
            selected_iface = st.selectbox(
                "Network interface", iface_names, disabled=controller.is_running,
                help="Prefer the interface with a real IP (e.g. 192.168.x.x or 10.x.x.x).",
            )
            duration = st.slider("Capture duration (seconds)", 10, 180, 30, disabled=controller.is_running)

            if not controller.is_running:
                if st.button("Start Live Capture", type="primary", use_container_width=True):
                    if not selected_iface:
                        st.error("No network interface selected.")
                    else:
                        controller.start_live(engine, selected_iface, duration)
                        st.rerun()
            else:
                st.markdown(
                    f'<div style="color:{_SIGNAL}; font-family:IBM Plex Mono, monospace; '
                    f'font-size:0.8rem;">&#9679; capture active (up to {duration}s)...</div>',
                    unsafe_allow_html=True,
                )
                time.sleep(1)
                st.rerun()

        else:
            uploaded = st.file_uploader("Upload a .pcap file", type=["pcap", "pcapng"], disabled=controller.is_running)
            if not controller.is_running and uploaded is not None:
                if st.button("Start Replay", type="primary", use_container_width=True):
                    tmp_path = Path("data/external") / uploaded.name
                    tmp_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        tmp_path.write_bytes(uploaded.getvalue())
                    except OSError as e:
                        st.error(f"Failed to save uploaded pcap: {e}")
                    else:
                        controller.start_replay(engine, tmp_path)
                        st.rerun()
            elif controller.is_running:
                st.markdown(
                    f'<div style="color:{_SIGNAL}; font-family:IBM Plex Mono, monospace; '
                    f'font-size:0.8rem;">&#9679; replay in progress...</div>',
                    unsafe_allow_html=True,
                )
                time.sleep(1)
                st.rerun()

        st.divider()
        if st.button("Clear results", use_container_width=True):
            st.session_state.results = []
            st.rerun()

        if controller.error:
            st.error(f"Last capture error: {controller.error}")

    # --- Drain queue into session state (main thread only) ---
    new_results = controller.drain_queue()
    if new_results:
        st.session_state.results.extend(new_results)
        results = st.session_state.results

    # --- Top-line metrics ---
    total = len(results)
    attacks = sum(1 for r in results if r.get("prediction") == "attack")
    normal = sum(1 for r in results if r.get("prediction") == "normal")
    errors = sum(1 for r in results if r.get("prediction") == "error")
    attack_rate = f"{(attacks / total * 100):.1f}%" if total else "—"

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(_metric_card("Connections Analyzed", total), unsafe_allow_html=True)
    m2.markdown(_metric_card("Attack Alerts", attacks, accent=_ALERT if attacks else None), unsafe_allow_html=True)
    m3.markdown(_metric_card("Normal Traffic", normal, accent=_SIGNAL if normal else None), unsafe_allow_html=True)
    m4.markdown(_metric_card("Attack Rate", attack_rate), unsafe_allow_html=True)
    if errors:
        st.warning(f"{errors} connection(s) failed classification — see logs for details.")

    st.write("")

    if total == 0:
        _render_feed(results)
        return

    # --- Charts ---
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        pred_counts = pd.Series([r.get("prediction", "error") for r in results]).value_counts()
        fig = px.pie(
            values=pred_counts.values, names=pred_counts.index,
            color=pred_counts.index,
            color_discrete_map={"normal": _SIGNAL, "attack": _ALERT, "error": _AMBER},
            title="Traffic classification breakdown", hole=0.55,
        )
        st.plotly_chart(_plotly_dark_theme(fig), use_container_width=True)

    with chart_col2:
        conf_df = pd.DataFrame([
            {"confidence": r["confidence"], "prediction": r["prediction"]}
            for r in results if r.get("prediction") in ("normal", "attack")
        ])
        if not conf_df.empty:
            fig2 = px.histogram(
                conf_df, x="confidence", color="prediction", nbins=20,
                color_discrete_map={"normal": _SIGNAL, "attack": _ALERT},
                title="Model confidence distribution",
            )
            st.plotly_chart(_plotly_dark_theme(fig2), use_container_width=True)

    st.write("")
    st.markdown("#### Live Signal Feed")
    _render_feed(results)

    st.write("")

    # --- SHAP detail panel for a selected connection ---
    options = list(range(len(results)))
    selected = st.selectbox(
        "Select a connection to explain", options,
        format_func=lambda i: (
            f"#{i}: {results[i].get('src_ip','?')}:{results[i].get('src_port','?')} -> "
            f"{results[i].get('dst_ip','?')}:{results[i].get('dst_port','?')} "
            f"[{results[i].get('prediction','?').upper()}]"
        ),
        index=len(results) - 1,
    )
    if selected is not None:
        _render_shap_panel(results[selected])


if __name__ == "__main__":
    main()
