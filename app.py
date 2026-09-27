import re
import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

from ai.gemini_investigation import generate_investigation_analysis
from ai.report_pdf import build_investigation_pdf
from replay.state import init_state, reset_state
from replay.engine import process_next_observation
from replay.state import ELECTRICITY_TARIFF, INTERVENTION_COST
from ml.artifacts import get_artifact_path
from reports.report_generator import generate_markdown_report
from alerts.sms import is_sms_configured, get_missing_config, test_sms_configuration
from investigation.evidence import gather_evidence
from investigation.explanations import (
    generate_anomaly_explanation,
    generate_corrective_actions,
    generate_explanations,
    generate_investigation_points,
)
from recommendations.engine import generate_recommendations, calculate_savings_summary
from ml.forecast import EnergyForecastModel, ARTIFACT_NAME as FORECAST_ARTIFACT

# Ensure dark theme is default (via config or just using Plotly dark mode)
st.set_page_config(page_title="VOLTURA Auditor", layout="wide", initial_sidebar_state="expanded")

# Load Env
load_dotenv()

# CSS for modern aesthetic
st.markdown("""
    <style>
    .big-font { font-size: 24px !important; font-weight: bold; }
    .status-RUNNING { color: #00ff00; font-weight: bold; }
    .status-READY { color: #aaaaaa; font-weight: bold; }
    .status-STOPPED { color: #ffaa00; font-weight: bold; }
    .status-COMPLETED { color: #00aaff; font-weight: bold; }
    .severity-NORMAL { color: #00cc66; font-weight: bold; }
    .severity-WATCH { color: #ffcc00; font-weight: bold; }
    .severity-INVESTIGATE { color: #ff6600; font-weight: bold; }
    .severity-HIGH { color: #ff0000; font-weight: bold; }
    .severity-CRITICAL { color: #cc0000; font-weight: bold; text-decoration: underline; }
    </style>
""", unsafe_allow_html=True)

# 1. Initialize State
init_state()
if "ai_analysis" not in st.session_state:
    st.session_state["ai_analysis"] = {}

# 2. Load cached data
@st.cache_data
def load_unseen_data():
    if os.path.exists('data_cache/unseen_data.pkl'):
        return pd.read_pickle('data_cache/unseen_data.pkl')
    return None


@st.cache_data
def load_metrics():
    with open(get_artifact_path('metrics.json'), 'r') as metrics_file:
        return json.load(metrics_file)


@st.cache_data(show_spinner=False)
def load_development_data():
    return pd.read_csv('development_dataset.csv')


@st.cache_data(show_spinner=False)
def build_cached_report(events, metrics, tariff, intervention_cost):
    return generate_markdown_report(events, metrics, tariff, intervention_cost)


@st.cache_resource
def _get_cached_forecast_model():
    """Load the forecast model once, return None if not trained yet."""
    path = get_artifact_path(FORECAST_ARTIFACT)
    if os.path.exists(path):
        return EnergyForecastModel.load(path)
    return None

state = st.session_state.replay_state
if state['unseen_data'] is None:
    state['unseen_data'] = load_unseen_data()

# 3. Sidebar Configuration
st.sidebar.title("VOLTURA Controls")

st.sidebar.markdown(f"**Status:** <span class='status-{state['replay_status']}'>{state['replay_status']}</span>", unsafe_allow_html=True)

col1, col2, col3 = st.sidebar.columns(3)

if state['replay_status'] in ['READY', 'STOPPED']:
    if col1.button("START"):
        if state['unseen_data'] is not None and state['replay_status'] != 'COMPLETED':
            state['replay_running'] = True
            state['replay_status'] = 'RUNNING'
            st.rerun()
elif state['replay_status'] == 'RUNNING':
    if col2.button("STOP"):
        state['replay_running'] = False
        state['replay_status'] = 'STOPPED'
        st.rerun()

if col3.button("RESET"):
    reset_state()
    st.rerun()

st.sidebar.divider()
state['fast_demo_mode'] = st.sidebar.checkbox("Fast Demo Mode", value=state['fast_demo_mode'])

pages = ["Live Monitor", "Main Overview", "Equipment Explorer", "Investigation Workspace",
         "Energy Forecast", "Potential Savings", "Recommendations",
         "Data Quality", "Model Intelligence", "Report"]
selected_page = st.sidebar.radio("Navigation", pages, key="current_page")

st.sidebar.divider()
st.sidebar.markdown("### SMS Configuration")
if is_sms_configured():
    st.sidebar.success("SMS Ready")
    if st.sidebar.button("Test SMS"):
        if test_sms_configuration():
            st.sidebar.success("Test sent!")
        else:
            st.sidebar.error("Test failed.")
else:
    st.sidebar.error("SMS NOT CONFIGURED")
    st.sidebar.caption(f"Missing: {get_missing_config()}")

# --- HELPER FUNCS ---
def plot_actual_vs_expected(results_list, title="Actual vs Expected Energy", equipment_id=None):
    if not results_list:
        return go.Figure()

    fig = go.Figure()

    colors = {
        'CHILLER-01': ('#00ff00', '#00aa00'),
        'CHILLER-02': ('#00ccff', '#0077aa'),
        'CHILLER-03': ('#ff00ff', '#aa00aa'),
    }
    grouped = {}
    for observation in results_list:
        current_equipment = observation['equipment_id']
        if equipment_id and equipment_id != 'All Chillers' and current_equipment != equipment_id:
            continue
        grouped.setdefault(current_equipment, []).append(observation)

    for current_equipment, observations in grouped.items():
        actual_color, expected_color = colors.get(current_equipment, ('#ffffff', '#aaaaaa'))
        timestamps = [observation['timestamp'] for observation in observations]
        actual_energy = [observation['actual_energy'] for observation in observations]
        expected_energy = [observation['expected_energy'] for observation in observations]
        anomaly_timestamps = [
            observation['timestamp']
            for observation in observations
            if observation['is_anomaly']
        ]
        anomaly_energy = [
            observation['actual_energy']
            for observation in observations
            if observation['is_anomaly']
        ]

        fig.add_trace(go.Scatter(
            x=timestamps,
            y=actual_energy,
            mode='lines',
            name=f'{current_equipment} Actual',
            line=dict(color=actual_color),
            connectgaps=False,
        ))
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=expected_energy,
            mode='lines',
            name=f'{current_equipment} Expected',
            line=dict(color=expected_color, dash='dash'),
            connectgaps=False,
        ))
        if anomaly_timestamps:
            fig.add_trace(go.Scatter(
                x=anomaly_timestamps,
                y=anomaly_energy,
                mode='markers',
                name=f'{current_equipment} Anomalies',
                marker=dict(color='red', size=8),
            ))

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=400,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="Timestamp",
        yaxis_title="Energy (kWh)",
        uirevision="live-energy-timeline",
    )
    return fig




def render_live_monitor(replay_state):
    st.title("Live Energy Monitor")
    
    # Add a small debug line at the top as requested
    st.caption(f"Replay tick: Index: {replay_state['current_index']} | Processed: {replay_state['current_index']}")
    
    if replay_state['unseen_data'] is not None:
        total = len(replay_state['unseen_data'])
        st.progress(replay_state['current_index'] / max(1, total), text=f"Progress: {replay_state['current_index']} / {total}")
    else:
        st.error("No data found. Please run training pipeline first.")
        st.stop()
        
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Processed", replay_state['current_index'])
    c2.metric("Normal", replay_state['normal_count'])
    c3.metric("Anomalies", replay_state['anomaly_count'])
    c4.metric("Excess Energy", f"{replay_state['total_excess_energy']:.1f} kWh")

    st.dataframe(
        pd.DataFrame([{
            "Total Energy": f"{replay_state['total_energy']:.1f} kWh",
            "Cumulative Energy Variance (kWh)": f"{replay_state['cumulative_energy_variance']:.1f}",
            "ROI (%)": f"{replay_state['roi_pct']:.2f}%",
        }]),
        hide_index=True,
        use_container_width=False,
    )
    st.caption("Electricity Tariff: ₹7.75/kWh | Intervention Cost: ₹10,000")

    st.subheader("Current Observation")
    obs = replay_state['current_observation']
    if obs:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Equipment", obs['equipment_id'])
        m2.metric("Actual (kWh)", f"{obs['actual_energy']:.1f}")
        m3.metric("Expected (kWh)", f"{obs['expected_energy']:.1f}")
        m4.metric("Deviation %", f"{obs['deviation_pct']:.1f}%", delta=f"{obs['deviation_pct']:.1f}%", delta_color="inverse")
        
        severity_html = f"<span class='severity-{obs['severity']}'>{obs['severity']}</span>"
        m5.markdown(f"**Severity:**<br>{severity_html}", unsafe_allow_html=True)
        
        st.caption(f"**Timestamp:** {obs['timestamp']} | **Context Score:** {obs['context_score']:.3f} | **Reason:** {obs['reason']}")
    else:
        st.info("Start replay to see live data.")
        
    st.plotly_chart(plot_actual_vs_expected(replay_state['processed_history']), use_container_width=True)


def render_investigation_graph(replay_state, event):
    history = replay_state['processed_history']
    equipment_history = [
        observation for observation in history
        if observation['equipment_id'] == event['equipment_id']
    ]
    if not equipment_history:
        st.info("No processed observations for this equipment yet.")
        return

    history_df = pd.DataFrame(equipment_history).sort_values('timestamp')
    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
    event_time = pd.to_datetime(event['timestamp'])
    nearest_index = (history_df['timestamp'] - event_time).abs().idxmin()
    position = history_df.index.get_loc(nearest_index)
    start = max(0, position - 30)
    end = min(len(history_df), position + 31)
    window = history_df.iloc[start:end]

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=window['timestamp'], y=window['actual_energy'], mode='lines', name='Actual Energy'))
    figure.add_trace(go.Scatter(x=window['timestamp'], y=window['expected_energy'], mode='lines', name='Expected Energy', line=dict(dash='dash')))
    selected = window[window['timestamp'] == event_time]
    if not selected.empty:
        figure.add_trace(go.Scatter(x=selected['timestamp'], y=selected['actual_energy'], mode='markers', name='Selected Anomaly', marker=dict(color='red', size=11)))
    figure.update_layout(title=f"{event['equipment_id']} Around Selected Event", template="plotly_dark", height=400, xaxis_title="Timestamp", yaxis_title="Energy (kWh)")
    st.plotly_chart(figure, use_container_width=True)


def render_selected_page(replay_state, page):
    if page == "Live Monitor":
        render_live_monitor(replay_state)

    elif page == "Main Overview":
        st.title("Main Overview")
        events = replay_state['anomaly_events']
        st.metric("Total Events Detected", len(events))
        if events:
            events_df = pd.DataFrame(events)
            st.dataframe(events_df[['event_id', 'equipment_id', 'start_time', 'duration', 'severity', 'maximum_deviation', 'total_excess_energy']].tail(10))
        else:
            st.info("No events detected yet.")
        st.plotly_chart(plot_actual_vs_expected(replay_state['processed_history'], title="Latest 300 Observations"), use_container_width=True)

    elif page == "Equipment Explorer":
        st.title("Equipment Explorer")
        history = replay_state['processed_history']
        if not history:
            st.info("No live observations yet.")
        else:
            equipment_options = ['All Chillers'] + sorted({row['equipment_id'] for row in history})
            selected_equipment = st.selectbox("Select Equipment", equipment_options)
            equipment_history = [row for row in history if selected_equipment == 'All Chillers' or row['equipment_id'] == selected_equipment]
            st.metric(f"Latest Observations ({selected_equipment})", len(equipment_history))
            st.plotly_chart(plot_actual_vs_expected(history, equipment_id=selected_equipment), use_container_width=True)

    elif page == "Investigation Workspace":
        st.title("Investigation Workspace")
        events = replay_state['anomaly_events']
        if not events:
            replay_state['current_anomaly_index'] = 0
            replay_state['selected_event_id'] = None
            st.info("No anomaly selected")
            return

        event_ids = [event['event_id'] for event in events]
        selected_index = max(0, min(replay_state['current_anomaly_index'], len(events) - 1))
        replay_state['current_anomaly_index'] = selected_index

        previous_col, counter_col, next_col = st.columns([1, 2, 1])
        if previous_col.button("← Previous", disabled=selected_index == 0, key="previous_anomaly"):
            next_index = max(0, selected_index - 1)
            replay_state['current_anomaly_index'] = next_index
            st.rerun()
        counter_col.markdown(f"<div style='text-align:center; padding-top:0.45rem; font-weight:600;'>Anomaly {selected_index + 1} of {len(events)}</div>", unsafe_allow_html=True)
        if next_col.button("Next →", disabled=selected_index == len(events) - 1, key="next_anomaly"):
            next_index = min(len(events) - 1, selected_index + 1)
            replay_state['current_anomaly_index'] = next_index
            st.rerun()

        event = events[selected_index]
        replay_state['selected_event_id'] = event['event_id']
        timestamp = pd.to_datetime(event['timestamp']).strftime('%Y-%m-%d %H:%M')
        st.markdown(f"### 🚨 {event['severity']}", unsafe_allow_html=True)
        st.markdown(f"**{event['equipment_id']}**  \n`{timestamp}`")

        e1, e2, e3 = st.columns(3)
        e1.metric("Actual Energy", f"{event['actual_energy']:.1f} kWh")
        e2.metric("Expected Energy", f"{event['expected_energy']:.1f} kWh")
        e3.metric("Deviation", f"+{event['deviation_pct']:.1f}%")

        context_row = event.get('row_data')
        if context_row:
            st.subheader("WHAT HAPPENED?")
            st.write(generate_anomaly_explanation(event, context_row))

            st.subheader("WHY IS IT ANOMALOUS?")
            st.write(event['reason'])

            st.subheader("FIELD EVIDENCE")
            evidence = gather_evidence(event, context_row)
            evidence_columns = st.columns(5)
            evidence_values = [
                ("Outside Temp", evidence['environmental_context']['outside_temp'], "°F"),
                ("Humidity", evidence['environmental_context']['humidity'], "%"),
                ("Wind Speed", evidence['environmental_context']['wind_speed'], "mph"),
                ("Building Load", evidence['operating_conditions']['load_rt'], "RT"),
                ("Water Rate", evidence['operating_conditions']['chilled_water_rate'], "L/sec"),
            ]
            for column, (label, value, unit) in zip(evidence_columns, evidence_values):
                display_value = "N/A" if value is None else f"{value} {unit}"
                column.metric(label, display_value)

            st.subheader("INVESTIGATION GRAPH")
            render_investigation_graph(replay_state, event)

            st.subheader("WHAT SHOULD BE INVESTIGATED?")
            for point in generate_investigation_points(event, context_row):
                st.markdown(f"- {point}")

            st.subheader("HOW TO ADDRESS IT")
            for action in generate_corrective_actions(event, context_row):
                st.markdown(f"- {action}")

            st.subheader("Existing Recommendations")
            for rec in generate_explanations(event, context_row):
                st.markdown(f"- {rec}")

            # AI section is rendered OUTSIDE the fragment (see render_ai_investigation_section below)
            # This prevents the 0.1-second auto-rerun from wiping session-state on button click.
        else:
            st.info("No anomaly selected")

# --- PAGES ---

@st.fragment(run_every=0.1)
def replay_scheduler_and_live_ui():
    process_next_observation()
    if selected_page in {"Live Monitor", "Main Overview", "Equipment Explorer"}:
        render_selected_page(state, selected_page)


replay_scheduler_and_live_ui()


# ── AI Investigation Analysis ───────────────────────────────────────
# Rendered OUTSIDE the @st.fragment so that session_state updates from
# button clicks survive across reruns and are not wiped by the 0.1-second
# replay auto-tick.
def render_ai_investigation_section():
    events = state['anomaly_events']
    if not events:
        return
    selected_index = state.get('current_anomaly_index', 0)
    selected_index = max(0, min(selected_index, len(events) - 1))
    event = events[selected_index]
    context_row = event.get('row_data')
    if not context_row:
        return

    event_id = event['event_id']

    st.divider()
    st.subheader("🤖 AI Investigation Analysis")

    cached_analysis = st.session_state["ai_analysis"].get(event_id)

    ai_col1, ai_col2 = st.columns([2, 1])
    with ai_col1:
        generate_clicked = st.button(
            "🔍 Generate AI Analysis",
            key=f"ai_btn_{event_id}",
            type="primary",
        )
    with ai_col2:
        if cached_analysis:
            st.success("✅ Cached — no repeat call")

    if generate_clicked:
        with st.spinner("Analyzing selected anomaly..."):
            analysis, error = generate_investigation_analysis(
                event,
                ELECTRICITY_TARIFF,
                INTERVENTION_COST,
            )
        if error:
            # Show the actual error so it can be diagnosed
            st.error(f"AI analysis failed: {error}")
        elif not analysis:
            st.error("AI analysis failed: empty response returned.")
        else:
            st.session_state["ai_analysis"][event_id] = analysis
            cached_analysis = analysis

    # Display cached or freshly generated analysis
    if cached_analysis:
        SECTION_ICONS = {
            "EXECUTIVE SUMMARY":                  "📋",
            "ENERGY-WASTE ANALYSIS":              "⚡",
            "CONTEXTUAL / ENVIRONMENTAL ANALYSIS":"🌡️",
            "WHY IS IT ANOMALOUS?":               "🔍",
            "POSSIBLE CONTRIBUTING CONDITIONS":   "⚠️",
            "WHAT SHOULD BE INVESTIGATED?":       "🔧",
            "RECOMMENDED ACTION":                 "✅",
            "ENERGY SAVINGS OPPORTUNITY":         "💰",
            "FINANCIAL IMPACT":                   "📊",
            "PRIORITY / ACTIONABILITY":           "🚨",
            "BUSINESS IMPACT":                    "🏢",
        }
        sections = re.split(r'(?=^### )', cached_analysis, flags=re.MULTILINE)
        for section_text in sections:
            section_text = section_text.strip()
            if not section_text:
                continue
            if section_text.startswith("### "):
                lines = section_text.split("\n", 1)
                heading = lines[0][4:].strip()
                body = lines[1].strip() if len(lines) > 1 else ""
                icon = SECTION_ICONS.get(heading.upper(), "📄")
                with st.expander(f"{icon} {heading}", expanded=(heading.upper() == "EXECUTIVE SUMMARY")):
                    st.markdown(body)
            else:
                st.markdown(section_text)

        # PDF download
        st.divider()
        pdf_bytes = build_investigation_pdf(
            event,
            cached_analysis,
            ELECTRICITY_TARIFF,
            INTERVENTION_COST,
        )
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=f"VOLTURA_{event_id}_Investigation_Report.pdf",
            mime="application/pdf",
            key=f"pdf_btn_{event_id}",
        )


if selected_page == "Investigation Workspace":
    render_selected_page(state, selected_page)
    render_ai_investigation_section()

if selected_page == "Data Quality":
    st.title("Data Quality")
    try:
        data = load_metrics()
        st.metric("Training Rows", data['train_size'])
        st.metric("Unseen Rows", data['unseen_size'])
            
        st.subheader("Dataset Summary (from development_dataset.csv)")
        df = load_development_data()
        st.write(f"**Total rows:** {len(df)}")
        st.write(f"**Missing values:**")
        st.write(df.isnull().sum()[df.isnull().sum() > 0])
        st.write(f"**Missing Target Rows:** {df['Chiller Energy Consumption (kWh)'].isnull().sum()}")
        st.write(f"**Equipment count:** {df['equipment_id'].nunique()}")
        st.write(f"**Timestamp range:** {df['timestamp'].min()} to {df['timestamp'].max()}")
        
    except Exception as e:
        st.error(f"Error loading data quality metrics: {e}")

elif selected_page == "Model Intelligence":
    st.title("Model Intelligence")
    try:
        data = load_metrics()
        metrics = data['metrics']
            
        st.write("The final 20% was kept completely unseen during training.")
        
        c1, c2 = st.columns(2)
        c1.subheader("Model Metrics (Unseen Set)")
        c1.metric("MAE", f"{metrics['model']['mae']:.2f}")
        c1.metric("RMSE", f"{metrics['model']['rmse']:.2f}")
        c1.metric("R²", f"{metrics['model']['r2']:.4f}")
        
        c2.subheader("Baseline Metrics (Median)")
        c2.metric("MAE", f"{metrics['baseline']['mae']:.2f}")
        c2.metric("RMSE", f"{metrics['baseline']['rmse']:.2f}")
        c2.metric("R²", f"{metrics['baseline']['r2']:.4f}")
        
        st.subheader("Anomaly Detection Method")
        st.write("- **Contextual Context:** Isolation Forest (trained only on training data)")
        st.write("- **Energy Deviation:** Actual vs Expected energy deviation rules")
        st.write("- **Thresholds:**")
        st.write("  - NORMAL: <= 10%")
        st.write("  - WATCH: 10% - 20%")
        st.write("  - INVESTIGATE: 20% - 30%")
        st.write("  - HIGH: 30% - 50%")
        st.write("  - CRITICAL: >= 50%")
        
    except Exception as e:
        st.error(f"Error loading model intelligence: {e}")

elif selected_page == "Report":
    st.title("Automated Audit Report")
    tariff = st.number_input("Electricity Tariff (₹/kWh)", value=ELECTRICITY_TARIFF)
    
    try:
        metrics = load_metrics()['metrics']
    except:
        metrics = {}
        
    report = build_cached_report(tuple(state['anomaly_events']), metrics, tariff, INTERVENTION_COST)
    st.markdown(report)

# ── NEW FEATURE 1: Energy Forecast ───────────────────────────────────────────
elif selected_page == "Energy Forecast":
    st.title("📈 Energy Forecast")
    st.caption(
        "Forecasted energy consumption based on historical lag patterns. "
        "This is a statistical forecast — not a guaranteed future value."
    )

    forecast_model = _get_cached_forecast_model()
    if forecast_model is None:
        st.warning(
            "Forecast model not found. Run `python forecast_pipeline.py` to train it, "
            "then restart the app."
        )
    else:
        # Equipment selector
        history = state["processed_history"]
        equipments = sorted({r["equipment_id"] for r in history}) if history else \
                     list(forecast_model.forecasters.keys())

        selected_eq = st.selectbox("Select Equipment", equipments, key="fc_eq_select")

        # Use recent processed history as seed rows (live context)
        seed_rows = None
        current_load = None
        current_temp = None

        if history:
            eq_history = [r for r in history if r["equipment_id"] == selected_eq]
            if eq_history:
                seed_rows = pd.DataFrame(eq_history).rename(
                    columns={
                        "actual_energy": "Chiller Energy Consumption (kWh)",
                    }
                )
                # Expand row_data fields if available
                if "row_data" in seed_rows.columns:
                    row_data_expanded = seed_rows["row_data"].apply(
                        lambda x: x if isinstance(x, dict) else {}
                    ).apply(pd.Series)
                    
                    overlap = [c for c in row_data_expanded.columns if c in seed_rows.columns and c != "row_data"]
                    row_data_expanded = row_data_expanded.drop(columns=overlap)
                    
                    seed_rows = pd.concat(
                        [seed_rows.drop(columns=["row_data"]), row_data_expanded], axis=1
                    )
                seed_rows["timestamp"] = pd.to_datetime(seed_rows["timestamp"])

                last_obs = eq_history[-1]
                rd = last_obs.get("row_data", {})
                current_load = rd.get("Building Load (RT)")
                current_temp = rd.get("Outside Temperature (F)")

        forecasts = forecast_model.forecast(
            selected_eq,
            seed_rows=seed_rows,
            current_load_rt=current_load,
            current_outside_temp_f=current_temp,
        )

        if not forecasts:
            st.info(
                "Start the replay and process some observations to generate a live forecast. "
                "The forecaster needs recent history as a seed."
            )
        else:
            # Summary metrics
            current_kwh = None
            if history:
                eq_h = [r for r in history if r["equipment_id"] == selected_eq]
                if eq_h:
                    current_kwh = eq_h[-1]["actual_energy"]

            fc_cols = st.columns(len(forecasts) + (1 if current_kwh else 0))
            col_idx = 0
            if current_kwh is not None:
                fc_cols[col_idx].metric("Current (kWh)", f"{current_kwh:.1f}")
                col_idx += 1

            for fc in forecasts:
                label = f"+{fc['minutes_ahead']}min → {fc['label']}"
                delta = None
                if current_kwh is not None:
                    diff = fc["forecast_kwh"] - current_kwh
                    delta = f"{diff:+.1f} kWh"
                fc_cols[col_idx].metric(label, f"{fc['forecast_kwh']:.1f} kWh", delta=delta)
                col_idx += 1

            # Forecast chart
            if current_kwh is not None:
                chart_data = [{"label": "Now", "kwh": current_kwh, "minutes": 0}]
            else:
                chart_data = []
            for fc in forecasts:
                chart_data.append({
                    "label": fc["label"],
                    "kwh": fc["forecast_kwh"],
                    "minutes": fc["minutes_ahead"],
                })

            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=[d["label"] for d in chart_data],
                y=[d["kwh"] for d in chart_data],
                mode="lines+markers",
                name="Forecasted Energy (kWh)",
                line=dict(color="#00ccff", width=2),
                marker=dict(size=8),
            ))
            fig_fc.update_layout(
                title=f"{selected_eq} — Forecasted Energy",
                template="plotly_dark",
                height=350,
                xaxis_title="Time",
                yaxis_title="Energy (kWh)",
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            st.caption(
                f"Model: Lag-based Random Forest | Resolution: 30-min intervals | "
                f"Horizons: +30min / +1h / +3h | Trained on {len(forecast_model.forecasters)} chillers"
            )

        # Forecast model quality
        metrics_path = get_artifact_path("forecast_metrics.json")
        if os.path.exists(metrics_path):
            with st.expander("Forecast Model Accuracy (MAE on held-out validation set)"):
                with open(metrics_path) as f:
                    fm = json.load(f)
                for eq, m in fm.items():
                    st.write(f"**{eq}**")
                    for k, v in m.items():
                        step_name = k.replace("step_", "+").replace("_mae", " steps MAE")
                        st.write(f"  {step_name}: ±{v} kWh")


# ── NEW FEATURE 2: Potential Savings ─────────────────────────────────────────
elif selected_page == "Potential Savings":
    st.title("💰 Potential Energy Savings")
    st.caption(
        "Estimated potential savings from observed energy waste across all detected anomalies. "
        "All figures are estimates based on observed excess energy and configured tariff. "
        "Not guaranteed financial returns."
    )

    events = state["anomaly_events"]
    if not events:
        st.info("No anomaly events detected yet. Start the replay to generate data.")
    else:
        tariff_input = st.number_input(
            "Electricity Tariff (₹/kWh)",
            value=float(ELECTRICITY_TARIFF),
            min_value=0.1,
            step=0.25,
            key="savings_tariff",
        )

        # Aggregate across all events
        total_excess = 0.0
        total_potential_cost = 0.0
        rows = []
        for ev in events:
            summary = calculate_savings_summary(ev, tariff_input)
            if summary["available"]:
                total_excess += summary["excess_kwh"]
                total_potential_cost += summary["potential_cost_impact_inr"]
                rows.append({
                    "Event ID":           ev["event_id"],
                    "Equipment":          ev["equipment_id"],
                    "Severity":           ev["severity"],
                    "Actual (kWh)":       summary["actual_kwh"],
                    "Expected (kWh)":     summary["expected_kwh"],
                    "Potential Excess (kWh)": summary["excess_kwh"],
                    "Est. Cost Impact (₹)":  summary["potential_cost_impact_inr"],
                })

        # Summary metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Anomaly Events", len(events))
        m2.metric("Total Potential Excess Energy", f"{total_excess:.1f} kWh")
        m3.metric("Estimated Potential Cost Impact", f"₹{total_potential_cost:,.2f}")

        st.caption(
            "⚠️ 'Potential Excess Energy' is the sum of (Actual − Expected) kWh for each anomaly. "
            "These are potential energy-waste events — the actual achievable saving depends "
            "on root cause, corrective action, and implementation."
        )

        if rows:
            st.subheader("Event-by-Event Breakdown")
            df_savings = pd.DataFrame(rows)
            st.dataframe(df_savings, use_container_width=True, hide_index=True)

            # Chart: potential cost impact by event
            fig_savings = go.Figure(go.Bar(
                x=df_savings["Event ID"],
                y=df_savings["Est. Cost Impact (₹)"],
                marker_color="#ff6600",
                name="Est. Cost Impact (₹)",
            ))
            fig_savings.update_layout(
                title="Estimated Potential Cost Impact by Anomaly Event",
                template="plotly_dark",
                height=350,
                xaxis_title="Event ID",
                yaxis_title="Est. Cost Impact (₹)",
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_savings, use_container_width=True)


# ── NEW FEATURE 3: Recommendations / ROI ─────────────────────────────────────
elif selected_page == "Recommendations":
    st.title("🏆 Recommendations & ROI")
    st.caption(
        "Evidence-based recommendations ranked by priority, estimated savings, and payback period. "
        "Implementation costs and savings estimates are documented assumptions — see "
        "`config/recommendation_costs.json`. Not guaranteed financial returns."
    )

    events = state["anomaly_events"]
    if not events:
        st.info("No anomaly events detected yet. Start the replay to generate data.")
    else:
        # Select event to analyse
        event_ids = [e["event_id"] for e in events]
        selected_event_id = st.selectbox(
            "Select Anomaly Event",
            event_ids,
            index=len(event_ids) - 1,
            key="rec_event_select",
        )
        selected_event = next(e for e in events if e["event_id"] == selected_event_id)
        context_row = selected_event.get("row_data") or {}

        tariff_rec = st.number_input(
            "Electricity Tariff (₹/kWh)",
            value=float(ELECTRICITY_TARIFF),
            min_value=0.1,
            step=0.25,
            key="rec_tariff",
        )

        # Event summary
        st.subheader(f"Event: {selected_event_id}")
        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("Equipment", selected_event["equipment_id"])
        ec2.metric("Severity", selected_event["severity"])
        ec3.metric("Deviation", f"{selected_event.get('deviation_pct', 0):.1f}%")

        # Savings for this event
        savings = calculate_savings_summary(selected_event, tariff_rec)
        if savings["available"]:
            sc1, sc2 = st.columns(2)
            sc1.metric("Potential Excess Energy", f"{savings['excess_kwh']:.2f} kWh",
                       help="Actual − Expected energy for this event")
            sc2.metric("Estimated Potential Cost Impact",
                       f"₹{savings['potential_cost_impact_inr']:,.2f}",
                       help="Excess kWh × tariff. Estimated, not guaranteed.")

        # Recommendations
        recs = generate_recommendations(selected_event, context_row, tariff_rec)

        if not recs:
            st.info(
                "No specific recommendations triggered for this event's severity and evidence. "
                "The anomaly may be WATCH level or lack sufficient evidence for specific conditions."
            )
        else:
            PRIORITY_COLORS = {
                "CRITICAL": "#cc0000",
                "HIGH":     "#ff4444",
                "MEDIUM":   "#ff8800",
                "LOW":      "#aaaaaa",
            }

            for rec in recs:
                color = PRIORITY_COLORS.get(rec["priority"], "#888888")
                with st.expander(
                    f"{'🔴' if rec['priority'] in ('CRITICAL','HIGH') else '🟡'} "
                    f"[{rec['priority']}] {rec['title']}",
                    expanded=True,
                ):
                    st.markdown(
                        f"<span style='color:{color};font-weight:bold;'>"
                        f"{rec['priority']} PRIORITY</span>",
                        unsafe_allow_html=True,
                    )
                    st.write(rec["description"])

                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric(
                        "Est. Monthly Saving",
                        f"{rec['estimated_monthly_kwh']:.0f} kWh",
                    )
                    r2.metric(
                        "Est. Monetary Saving",
                        f"₹{rec['estimated_monthly_savings_inr']:,.0f}/mo",
                    )
                    r3.metric(
                        "Est. Implementation Cost",
                        f"₹{rec['estimated_implementation_cost']:,}",
                    )
                    r4.metric(
                        "Estimated Payback",
                        f"{rec['estimated_payback_months']} months"
                        if rec["estimated_payback_months"] else "N/A",
                    )

                    st.caption(f"**Reason:** {rec['reason']}")

        st.divider()
        st.caption(
            "All implementation costs and savings figures are estimates based on documented "
            "assumptions in `config/recommendation_costs.json`. "
            "Payback = Estimated Implementation Cost ÷ Estimated Monthly Savings."
        )
