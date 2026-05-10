import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Ecosystem Stability", layout="wide")
st.title("🌿 Ecological Diversity‑Stability Portfolio Theory")
st.caption("Community matrix (VAR) | Structural stability via eigenvalues | May's theorem")

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(files):
    json_files = [f for f in files if f.endswith('.json') and 'stability' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

files = list_repo_files()
latest = find_latest_json(files)
if not latest:
    st.error("No stability results found. Run trainer first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error loading JSON: {data['error']}")
    st.stop()

st.sidebar.header("ℹ️ Info")
st.sidebar.write(f"**Run date:** {data['run_date']}")
st.sidebar.write(f"**Next trading day:** {next_trading_day()}")
st.sidebar.write("**Method:** VAR(1) community matrix → max real eigenvalue. Positive = unstable.")

universes = data["universes"]
if not universes:
    st.warning("No universe data.")
    st.stop()

st.header("⚠️ Current Ecosystem Stability (Per Universe)")

for universe_name, uni_data in universes.items():
    status = uni_data.get("stability_status", "?")
    lam = uni_data.get("lambda_max", 0.0)
    win = uni_data.get("selected_window", "?")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(f"{universe_name} Stability", status.upper())
    with col2:
        st.metric("λ_max", f"{lam:.4f}")
    with col3:
        st.metric("Window (days)", win)
    with col4:
        st.metric("Effective Diversity", f"{uni_data.get('effective_diversity',0):.1f}")
    # Destabilising ETFs
    destab = uni_data.get("destabilizing_etfs", [])
    if destab:
        st.write("**Most destabilising ETFs:**")
        for d in destab[:3]:
            st.markdown(f"- {d['ticker']} (impact {d['impact']:.3f})")
    st.divider()

# Detailed view
universe_names = list(universes.keys())
selected = st.selectbox("Select Universe for detailed view", universe_names)

if selected:
    uni_data = universes[selected]
    all_tickers = uni_data.get("all_tickers", {})
    if all_tickers:
        rows = [{"ETF": t, "Destabilising Impact": v["destabilizing_impact"]} for t, v in all_tickers.items()]
        df = pd.DataFrame(rows).sort_values("Destabilising Impact", ascending=False)
        st.subheader("📊 Destabilising Impact per ETF (left eigenvector)")
        st.dataframe(df, use_container_width=True, hide_index=True)
        fig = px.bar(df, x="ETF", y="Destabilising Impact", title="Marginal Destabilising Effect")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No per‑ticker data available.")

st.caption("Positive λ_max implies the system is structurally unstable (May's theorem). Destabilising ETFs have high influence on the largest eigenvalue.")
