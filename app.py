from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Thales_Group_Manufacturing.csv"
MODEL_PATH = ROOT / "model.joblib"
METRICS_PATH = ROOT / "reports" / "model_metrics.json"
st.set_page_config(page_title="Manufacturing Efficiency AI", page_icon="🏭", layout="wide")

@st.cache_data
def load_data():
    d = pd.read_csv(DATA)
    d["DateTime"] = pd.to_datetime(d["Date"].astype(str) + " " + d["Timestamp"].astype(str), dayfirst=True, errors="coerce")
    return d.dropna(subset=["DateTime"])

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

df = load_data()
st.title("🏭 AI-Based Manufacturing Efficiency Classification")
st.caption("Interactive analysis of sensor, production, quality and network measurements")
st.info("Prototype using a static CSV. Dashboard filters and predictions update interactively, but this is not a live factory/6G connection.")

with st.sidebar:
    st.header("Dashboard filters")
    machine_list = sorted(df["Machine_ID"].dropna().unique().tolist())
    chosen_machines = st.multiselect("Machines", machine_list, default=machine_list)
    mode_list = sorted(df["Operation_Mode"].dropna().unique().tolist())
    chosen_modes = st.multiselect("Operation mode", mode_list, default=mode_list)
    min_date, max_date = df["DateTime"].min().date(), df["DateTime"].max().date()
    date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    max_latency = st.slider("Maximum network latency (ms)", 1.0, 50.0, 50.0, 1.0)
    max_packet_loss = st.slider("Maximum packet loss (%)", 0.0, 5.0, 5.0, 0.1)
    min_speed = st.slider("Minimum production speed (units/hr)", 50.0, 500.0, 50.0, 10.0)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

view = df[
    df["Machine_ID"].isin(chosen_machines)
    & df["Operation_Mode"].isin(chosen_modes)
    & (df["DateTime"].dt.date >= start_date)
    & (df["DateTime"].dt.date <= end_date)
    & (df["Network_Latency_ms"] <= max_latency)
    & (df["Packet_Loss_%"] <= max_packet_loss)
    & (df["Production_Speed_units_per_hr"] >= min_speed)
].copy()

if view.empty:
    st.warning("No rows match the current filters. Widen the filters.")
    st.stop()

counts = view["Efficiency_Status"].value_counts()
total = len(view)
m1,m2,m3,m4 = st.columns(4)
m1.metric("Records in view", f"{total:,}")
m2.metric("Machines", f"{view['Machine_ID'].nunique():,}")
m3.metric("Low efficiency", f"{100*view['Efficiency_Status'].eq('Low').mean():.1f}%")
m4.metric("Average production speed", f"{view['Production_Speed_units_per_hr'].mean():.1f}")

tab1,tab2,tab3,tab4 = st.tabs(["Overview", "Machine insights", "Network & sensors", "Predict efficiency"])
with tab1:
    st.subheader("Efficiency class distribution")
    st.bar_chart(counts.reindex(["High","Medium","Low"],fill_value=0))
    st.subheader("Mean operational metrics by efficiency status")
    metrics = ["Production_Speed_units_per_hr","Error_Rate_%","Quality_Control_Defect_Rate_%","Temperature_C","Vibration_Hz","Network_Latency_ms","Packet_Loss_%","Power_Consumption_kW","Predictive_Maintenance_Score"]
    st.dataframe(view.groupby("Efficiency_Status")[metrics].mean().round(3),use_container_width=True)
    st.subheader("Records")
    st.dataframe(view[["DateTime","Machine_ID","Operation_Mode"]+metrics+["Efficiency_Status"]].head(500),use_container_width=True)
with tab2:
    st.subheader("Machine-level status trends")
    machine = st.selectbox("Select machine for trend", sorted(view["Machine_ID"].unique()))
    mdf = view[view["Machine_ID"]==machine].sort_values("DateTime").copy()
    st.write(f"Machine {machine}: {len(mdf):,} records in current filters")
    st.line_chart(mdf.set_index("DateTime")[["Production_Speed_units_per_hr","Error_Rate_%"]].resample("D").mean())
    st.bar_chart(mdf["Efficiency_Status"].value_counts().reindex(["High","Medium","Low"],fill_value=0))
    st.subheader("Historical classification patterns")
    st.dataframe(pd.crosstab(mdf["DateTime"].dt.date,mdf["Efficiency_Status"]).tail(30),use_container_width=True)
with tab3:
    st.subheader("Network indicators by efficiency status")
    net_cols=["Network_Latency_ms","Packet_Loss_%"]
    st.line_chart(view.set_index("DateTime")[net_cols].resample("D").mean())
    st.subheader("Compare network vs sensor / production metrics")
    metric_choice=st.selectbox("Metric", ["Network_Latency_ms","Packet_Loss_%","Temperature_C","Vibration_Hz","Power_Consumption_kW","Quality_Control_Defect_Rate_%","Production_Speed_units_per_hr","Predictive_Maintenance_Score","Error_Rate_%"])
    st.bar_chart(view.groupby("Efficiency_Status")[metric_choice].mean().reindex(["High","Medium","Low"]))
    st.caption("Group differences are descriptive associations; they do not establish that a metric caused a change in efficiency.")
with tab4:
    st.subheader("Classify a new observation")
    bundle=load_model()
    if bundle is None:
        st.warning("Model file not found. Run `python train_model.py` first.")
    else:
        with st.form("predict_form"):
            c1,c2,c3=st.columns(3)
            with c1:
                machine_id=st.selectbox("Machine ID",machine_list,key="p_machine")
                mode=st.selectbox("Operation mode",mode_list,key="p_mode")
                temp=st.slider("Temperature (°C)",30.0,90.0,60.0)
                vibration=st.slider("Vibration (Hz)",0.1,5.0,2.5)
            with c2:
                power=st.slider("Power (kW)",1.5,10.0,5.7)
                latency=st.slider("Network latency (ms)",1.0,50.0,25.0)
                packet=st.slider("Packet loss (%)",0.0,5.0,2.5)
                defect=st.slider("Defect rate (%)",0.0,10.0,5.0)
            with c3:
                speed=st.slider("Production speed (units/hr)",50.0,500.0,275.0)
                maintenance=st.slider("Predictive maintenance score",0.0,1.0,0.5)
                error=st.slider("Error rate (%)",0.0,15.0,7.5)
            submit=st.form_submit_button("Predict efficiency status")
        if submit:
            sample=pd.DataFrame([{
                "Machine_ID":str(machine_id),"Operation_Mode":mode,"Temperature_C":temp,
                "Vibration_Hz":vibration,"Power_Consumption_kW":power,"Network_Latency_ms":latency,
                "Packet_Loss_%":packet,"Quality_Control_Defect_Rate_%":defect,
                "Production_Speed_units_per_hr":speed,"Predictive_Maintenance_Score":maintenance,"Error_Rate_%":error
            }])
            model=bundle["model"]
            pred=model.predict(sample)[0]
            st.success(f"Predicted efficiency status: {pred}")
            if hasattr(model,"predict_proba"):
                probs=model.predict_proba(sample)[0]
                st.metric("Highest class probability (not necessarily calibrated confidence)",f"{100*np.max(probs):.1f}%")
                st.bar_chart(pd.Series(probs,index=model.classes_,name="Model probability"))
            st.caption("Prediction is advisory. Validate with operators and actual production records before operational use.")

st.sidebar.caption("Data period: "+str(min_date)+" to "+str(max_date))
