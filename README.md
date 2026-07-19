# Aegis-NIDS — AI-Based Network Intrusion Detection System

A production-grade, end-to-end NIDS built on the NSL-KDD dataset: data pipeline → trained
XGBoost classifier → SHAP explainability → live Scapy packet capture → interactive
Streamlit dashboard.

## What this project does

Aegis-NIDS classifies network traffic as **normal** or **attack** in real time, using a
model trained on the NSL-KDD intrusion detection benchmark. Every prediction comes with a
SHAP-based explanation (which features drove the decision, and in which direction), so
alerts are auditable rather than a black-box verdict.

## Architecture

```
aegis-nids/
├── config/config.yaml          # central config: paths, hyperparameters, feature lists
├── src/
│   ├── data/
│   │   ├── loader.py            # NSL-KDD schema, safe download/load, validation
│   │   └── preprocessor.py      # fit/transform pipeline, persisted via joblib
│   ├── models/
│   │   └── train.py             # RF vs XGBoost comparison, metric-driven selection
│   ├── explainability/
│   │   └── shap_explainer.py    # local + global SHAP explanations
│   ├── simulation/
│   │   ├── feature_extractor.py # raw packets -> NSL-KDD-schema connection records
│   │   └── packet_sniffer.py    # Scapy live capture + pcap replay engine
│   └── utils/                   # logging, custom exceptions, shared config loading
├── app/
│   └── dashboard.py             # Streamlit live UI
├── run_live_capture.py          # CLI entrypoint for live capture (no UI)
└── requirements.txt
```

## Setup

**Requires Python 3.11** (not 3.12+, not 3.14). Several pinned dependencies
(`pillow`, `shap`, `numpy`) do not have prebuilt wheels for newer Python
versions and will fail to install otherwise. If you have multiple Python
versions installed, use the version-specific launcher when creating the
virtual environment:

```bash
# Windows (if multiple Python versions are installed):
py -3.11 -m venv venv
venv\Scripts\Activate.ps1

# Linux/Mac:
python3.11 -m venv venv
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## Running it

```bash
# 1. Load & validate the dataset (auto-downloads NSL-KDD on first run)
python -m src.data.loader

# 2. Fit and persist the preprocessing pipeline
python -m src.data.preprocessor

# 3. Train Random Forest + XGBoost, compare, save the winner
python -m src.models.train

# 4. Generate SHAP explanations (local + global)
python -m src.explainability.shap_explainer

# 5a. Live capture from the CLI (requires Npcap on Windows + admin/root)
python run_live_capture.py --interface "Wi-Fi" --duration 30

# 5b. Or launch the full interactive dashboard
streamlit run app/dashboard.py
```

## Key engineering decisions

- **Dataset:** NSL-KDD over the full CICIDS2017 — deliberately chosen for faster
  iteration and lower local resource requirements, at the cost of a smaller/older
  feature set.
- **Model selection:** Random Forest and XGBoost were trained and compared on
  ROC-AUC, Recall, Precision, F1, and False Positive Rate — not accuracy alone, since the
  test set's class balance is inverted from training (more attacks than normal traffic).
  XGBoost was selected: it catches materially more attacks (67.0% vs 62.2% recall) for a
  negligible FPR cost (+0.11 percentage points), which matters more for a NIDS where a
  missed attack is typically costlier than one extra reviewed alert. See
  `src/models/registry/model_metadata.json` (generated after training) for the exact
  metrics and the selection rule that was applied.
- **Explainability:** SHAP's `TreeExplainer` computes exact (not sampled) Shapley values
  for the tree-based model, giving both per-connection and global feature attribution.
- **Live capture:** uses `AsyncSniffer` with an independent wall-clock stop timer, rather
  than Scapy's blocking `sniff(timeout=...)`, which was found to be unreliable on Windows
  (captures could run far past the requested duration).

## Known limitations

- **Content-inspection features** (`num_failed_logins`, `logged_in`, `root_shell`, etc.)
  cannot be reconstructed from generic packet headers alone — they originally required
  deep payload inspection of authenticated sessions. These are fixed at 0 for live/replay
  traffic. SHAP analysis shows this has limited practical impact: traffic-pattern features
  (`src_bytes`, `count`, `dst_host_srv_count`, etc.) dominate the model's decisions.
- **Connection tracking is simplified** relative to a full TCP state machine — a
  connection is finalized on the first FIN/RST seen from either side, which can
  occasionally split one logical connection into two records during graceful bidirectional
  close.
- **NSL-KDD's training/test split is intentionally mismatched**: the test set includes
  attack types never seen in training (37 vs 22 types), which is why cross-validation
  ROC-AUC on training data (~1.00) is much higher than held-out test performance — this
  reflects genuine generalization difficulty, not a bug.
- **Streamlit Community Cloud may not reliably read `.python-version`/`runtime.txt`** —
  during deployment, this repo's cloud host kept defaulting to a newer Python version
  despite both files being present and correctly committed. The reliable fix was setting
  the Python version explicitly via the app's Settings panel on share.streamlit.io
  (Settings → Python version → 3.11), not just relying on the repo files.

## Project background

Built iteratively across 6 sprints (environment setup → preprocessing → model training →
explainability → live simulation → dashboard), with each stage verified end-to-end before
moving to the next.
