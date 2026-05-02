import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import io

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Bias Detector", layout="wide", page_icon="🔍")
st.title("🔍 Automated Bias Detector")
st.caption("Scan datasets for statistical and systemic biases before they corrupt your models.")

with st.sidebar:
    st.header("Configuration")
    st.slider("Underrepresentation threshold", 0.05, 0.50, 0.20)
    st.slider("Statistical significance (p)", 0.01, 0.10, 0.05)
    st.markdown("---")
    demo = st.selectbox("Load a demo", ["None", "Titanic", "COMPAS", "Adult Income"])

for key in ["df", "profile", "bias_results", "session_id"]:
    if key not in st.session_state:
        st.session_state[key] = None

ID_HINTS = ["id", "index", "row", "uuid", "key", "unnamed"]

def is_id_col(col):
    return any(hint == col.lower().strip() or col.lower().startswith(hint + "_")
               for hint in ID_HINTS)

def get_valid_target_cols(df):
    valid = [col for col in df.columns if not is_id_col(col) and df[col].nunique() <= 10]
    return valid if valid else [c for c in df.columns if not is_id_col(c)]

severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢", "unknown": "⚪"}

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 Upload & Scan",
    "📊 Bias Details",
    "⚡ Impact Simulator",
    "📈 Bias Over Time",
    "⚙️ Custom References",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Upload & Scan
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Upload your dataset")
    snapshot_label = st.text_input(
        "Snapshot label (optional)",
        placeholder="e.g. Jan 2024, v2, post-cleaning",
        help="Used for tracking bias over time"
    )
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if demo != "None" and uploaded_file is None:
        demo_paths = {
            "Titanic": "datasets/titanic.csv",
            "COMPAS": "datasets/compas-scores-two-years.csv",
            "Adult Income": "datasets/adult.csv",
        }
        try:
            st.session_state.df = pd.read_csv(demo_paths[demo])
            st.info(f"Loaded demo: {demo}")
        except FileNotFoundError:
            st.warning("Demo file not found. Please upload manually.")

    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)

    if st.session_state.df is not None:
        df = st.session_state.df
        st.write(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
        st.dataframe(df.head(10), use_container_width=True)

        if st.button("🚀 Run Bias Scan", type="primary"):
            with st.spinner("Analyzing dataset..."):
                try:
                    buf = io.BytesIO()
                    df.to_csv(buf, index=False)
                    buf.seek(0)
                    params = {}
                    if snapshot_label:
                        params["snapshot_label"] = snapshot_label
                    resp = requests.post(
                        f"{API_URL}/analyze",
                        files={"file": ("dataset.csv", buf, "text/csv")},
                        params=params,
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    st.session_state.profile = data["profile"]
                    st.session_state.bias_results = data["bias_results"]
                    st.session_state.session_id = data["session_id"]
                    st.success("✅ Scan complete!")
                except Exception as e:
                    st.error(f"API error: {e}")

    if st.session_state.bias_results:
        st.markdown("---")
        st.subheader("Bias Score Card")
        results = st.session_state.bias_results
        cols = st.columns(len(results))
        for i, result in enumerate(results):
            with cols[i]:
                icon = severity_icon.get(result["severity"], "⚪")
                st.metric(
                    label=result["bias_type"].replace("_", " ").title(),
                    value=f"{icon} {result['severity'].upper()}",
                    delta="Detected" if result["detected"] else "Clean",
                    delta_color="inverse" if result["detected"] else "normal",
                )

        st.markdown("---")
        col_dl, _ = st.columns([1, 3])
        with col_dl:
            if st.button("📄 Download PDF Report"):
                with st.spinner("Generating PDF..."):
                    try:
                        resp = requests.get(
                            f"{API_URL}/report/pdf",
                            params={"session_id": st.session_state.session_id},
                            timeout=60,
                        )
                        resp.raise_for_status()
                        st.download_button(
                            label="⬇️ Save Report",
                            data=resp.content,
                            file_name=f"bias_report_{st.session_state.session_id}.pdf",
                            mime="application/pdf",
                        )
                    except Exception as e:
                        st.error(f"PDF error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Bias Details
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not st.session_state.bias_results:
        st.info("Run a scan first.")
    else:
        for result in st.session_state.bias_results:
            bias_label = result["bias_type"].replace("_", " ").title()
            severity = result["severity"]
            evidence = result.get("evidence", {})
            explain = result.get("explainability", {})

            with st.expander(f"**{bias_label}** — {severity.upper()}", expanded=result["detected"]):
                if result["affected_columns"]:
                    st.write(f"**Affected columns:** {', '.join(result['affected_columns'])}")
                st.info(result["recommendation"])

                # Explainability
                if explain.get("summary"):
                    st.markdown("**💡 Why this matters:**")
                    st.write(explain["summary"])
                if explain.get("examples"):
                    for ex in explain["examples"]:
                        st.markdown(f"> {ex}")

                st.markdown("---")

                # Demographic charts
                if result["bias_type"] == "demographic_disparity":
                    for col, findings in evidence.items():
                        if not isinstance(findings, dict) or "observed" not in findings:
                            continue
                        observed = findings["observed"]
                        ref = findings.get("reference_distribution", {})
                        all_groups = sorted(set(list(observed.keys()) + list(ref.keys())))
                        rows = []
                        for g in all_groups:
                            if g in observed:
                                rows.append({"Group": g, "Rate": observed[g], "Source": "Observed"})
                            if g in ref:
                                rows.append({"Group": g, "Rate": ref[g], "Source": "Reference"})
                        if rows:
                            fig = px.bar(
                                pd.DataFrame(rows), x="Group", y="Rate", color="Source",
                                barmode="group", title=f"{col} — Observed vs Reference",
                                color_discrete_map={"Observed": "#4C9BE8", "Reference": "#E8854C"},
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        gaps = findings.get("representation_gaps", {})
                        if gaps:
                            st.dataframe(
                                pd.DataFrame([{"Group": k, "Gap": v} for k, v in gaps.items()]),
                                use_container_width=True
                            )
                        p_val = findings.get("p_value")
                        if p_val is not None:
                            st.caption(f"Chi-squared p={p_val} — {'Significant ✓' if findings.get('significant') else 'Not significant'}")

                elif result["bias_type"] == "temporal_bias":
                    drift_items = {k: v for k, v in evidence.items() if "_drift" in k}
                    if drift_items:
                        rows = []
                        for key, val in drift_items.items():
                            rows.append({
                                "Column": key.replace("_drift", ""),
                                "KS Stat": val.get("ks_stat"),
                                "P Value": val.get("p_value"),
                                "Early Mean": val.get("early_mean"),
                                "Late Mean": val.get("late_mean"),
                            })
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    sparse = evidence.get("sparse_time_periods", {})
                    if sparse:
                        st.warning(f"Sparse periods: {', '.join(sparse.get('periods', []))}")

                else:
                    for key, val in evidence.items():
                        if isinstance(val, dict) and "interpretation" in val:
                            st.write(f"• {val['interpretation']}")
                        elif isinstance(val, str):
                            st.write(f"• {val}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Impact Simulator
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if st.session_state.df is None:
        st.info("Upload a dataset first.")
    else:
        df = st.session_state.df
        valid_targets = get_valid_target_cols(df)
        all_cols_no_id = [c for c in df.columns if not is_id_col(c)]

        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox("Target column", options=valid_targets)
        with col2:
            protected_options = ["None"] + [c for c in all_cols_no_id if c != target_col]
            protected_col = st.selectbox("Protected attribute", options=protected_options)
            protected_col = None if protected_col == "None" else protected_col

        if target_col:
            dist = df[target_col].value_counts()
            st.caption("Class distribution: " + " | ".join([f"{k}: {v}" for k, v in dist.items()]))

        st.markdown("---")
        st.subheader("Step 1 — Balance")
        balance_method = st.radio("Method", ["SMOTE (numeric)", "Demographic group augmentation"])

        if balance_method == "SMOTE (numeric)":
            if df[target_col].nunique() != 2:
                st.warning(f"⚠️ SMOTE needs a binary target. '{target_col}' has {df[target_col].nunique()} values.")
            else:
                if st.button("Balance with SMOTE"):
                    with st.spinner("Running SMOTE..."):
                        try:
                            resp = requests.post(
                                f"{API_URL}/balance/smote",
                                params={"session_id": st.session_state.session_id, "target_col": target_col},
                                timeout=120,
                            )
                            resp.raise_for_status()
                            data = resp.json()
                            st.success(f"✅ Added {data['synthetic_rows_added']} synthetic rows.")
                            st.write(data["class_distribution"])
                        except Exception as e:
                            st.error(str(e))
        else:
            group_col = st.selectbox("Group column", [c for c in df.columns if not is_id_col(c)])
            target_group = st.selectbox("Underrepresented group", df[group_col].dropna().unique().tolist())
            current_count = int((df[group_col] == target_group).sum())
            st.caption(f"Current count: {current_count:,}")
            target_count = st.number_input("Target count", min_value=current_count + 1, value=current_count * 2)
            if st.button("Augment group"):
                with st.spinner("Synthesizing..."):
                    try:
                        resp = requests.post(
                            f"{API_URL}/balance/demographic",
                            params={
                                "session_id": st.session_state.session_id,
                                "group_col": group_col,
                                "target_group": str(target_group),
                                "target_count": int(target_count),
                            },
                            timeout=120,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        st.success(f"✅ {target_group}: {data['original_count']} → {data['new_count']} rows")
                    except Exception as e:
                        st.error(str(e))

        st.markdown("---")
        st.subheader("Step 2 — Simulate impact")
        if st.button("⚡ Run Impact Simulation", type="primary"):
            if not st.session_state.session_id:
                st.error("Run the bias scan first.")
            else:
                with st.spinner("Training models..."):
                    try:
                        resp = requests.post(
                            f"{API_URL}/simulate-impact",
                            params={
                                "session_id": st.session_state.session_id,
                                "target_col": target_col,
                                "protected_col": protected_col,
                            },
                            timeout=180,
                        )
                        resp.raise_for_status()
                        impact = resp.json()

                        rows = []
                        for label in ["biased", "balanced"]:
                            if "error" in impact.get(label, {}):
                                st.error(f"{label}: {impact[label]['error']}")
                                continue
                            row = {"Dataset": label.title()}
                            row.update(impact[label].get("overall", {}))
                            rows.append(row)

                        if rows:
                            st.subheader("Overall Metrics")
                            st.dataframe(pd.DataFrame(rows).set_index("Dataset"), use_container_width=True)
                            if len(rows) == 2:
                                f1_b = rows[0].get("f1", 0)
                                f1_bal = rows[1].get("f1", 0)
                                delta = round(f1_bal - f1_b, 4)
                                st.metric("F1 change after balancing", f"{f1_bal:.4f}",
                                          delta=f"{delta:+.4f}")

                        if protected_col:
                            st.subheader(f"Per-Group — {protected_col}")
                            for label in ["biased", "balanced"]:
                                gd = impact.get(label, {}).get("per_group", {})
                                if not gd:
                                    continue
                                gdf = pd.DataFrame(gd).T.reset_index().rename(columns={"index": "Group"})
                                fig = px.bar(
                                    gdf.melt("Group", var_name="Metric", value_name="Score"),
                                    x="Group", y="Score", color="Metric", barmode="group",
                                    title=f"{label.title()} — Per Group",
                                )
                                st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Bias Over Time
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📈 Bias Score Over Time")
    st.caption("Each time you scan a dataset it's saved. Upload multiple versions to track how bias changes.")

    try:
        ds_resp = requests.get(f"{API_URL}/history/datasets", timeout=10)
        ds_resp.raise_for_status()
        tracked = ds_resp.json().get("datasets", [])
    except Exception:
        tracked = []

    if not tracked:
        st.info("No scan history yet. Run a scan in the Upload & Scan tab first.")
    else:
        selected_ds = st.selectbox("Select dataset to view history", ["All"] + tracked)
        dataset_filter = None if selected_ds == "All" else selected_ds

        try:
            trend_resp = requests.get(
                f"{API_URL}/history/trend",
                params={"dataset_name": dataset_filter} if dataset_filter else {},
                timeout=10,
            )
            trend_resp.raise_for_status()
            trend_data = trend_resp.json()
            trend_df = pd.DataFrame(trend_data.get("trend", []))
        except Exception as e:
            st.error(str(e))
            trend_df = pd.DataFrame()

        if trend_df.empty:
            st.info("No trend data available.")
        else:
            # Severity score line chart
            st.markdown("#### Severity Score Over Time (3=High, 2=Medium, 1=Low)")
            fig = px.line(
                trend_df,
                x="label",
                y="severity_score",
                color="bias_type",
                markers=True,
                title="Bias Severity Trend",
                labels={"label": "Snapshot", "severity_score": "Severity Score", "bias_type": "Bias Type"},
            )
            fig.update_layout(yaxis=dict(tickvals=[1, 2, 3], ticktext=["Low", "Medium", "High"]))
            st.plotly_chart(fig, use_container_width=True)

            # Affected column count
            st.markdown("#### Affected Columns Over Time")
            fig2 = px.bar(
                trend_df,
                x="label",
                y="affected_column_count",
                color="bias_type",
                barmode="group",
                title="Number of Affected Columns per Snapshot",
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Raw history table
            with st.expander("Raw scan history"):
                display_cols = ["label", "timestamp", "bias_type", "severity", "detected", "affected_column_count"]
                st.dataframe(
                    trend_df[[c for c in display_cols if c in trend_df.columns]],
                    use_container_width=True
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Custom Reference Distributions
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("⚙️ Custom Reference Distributions")
    st.caption(
        "By default, demographic checks use US Census data. "
        "Override this with your own population (e.g. your customer base or regional data)."
    )

    # Show existing references
    try:
        ref_resp = requests.get(f"{API_URL}/references", timeout=10)
        ref_resp.raise_for_status()
        all_refs = ref_resp.json().get("references", {})
    except Exception:
        all_refs = {}

    if all_refs:
        st.markdown("#### Active Reference Distributions")
        for ref_name, ref_data in all_refs.items():
            is_custom = ref_data.get("custom", False)
            badge = "🟢 Custom" if is_custom else "🔵 Built-in"
            with st.expander(f"{badge} — {ref_name}"):
                st.write(f"**Column hint:** `{ref_data.get('col_hint', '—')}`")
                dist = ref_data.get("distribution", {})
                dist_df = pd.DataFrame([
                    {"Group": k, "Share": f"{v*100:.1f}%"}
                    for k, v in dist.items()
                ])
                st.dataframe(dist_df, use_container_width=True)
                fig = px.pie(dist_df, names="Group", title=f"{ref_name} distribution")
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Add Custom Reference Distribution")
    st.caption("Define the *expected* distribution for a column in your domain.")

    ref_name = st.text_input("Reference name", placeholder="e.g. Our Customer Base — Gender")
    col_hint = st.text_input(
        "Column name hint",
        placeholder="e.g. gender, race, age_group",
        help="The detector will match this against column names in your dataset"
    )

    st.markdown("**Define group proportions** (will be auto-normalized to sum to 100%)")
    n_groups = st.number_input("Number of groups", min_value=2, max_value=10, value=2)

    group_inputs = {}
    cols = st.columns(2)
    for i in range(int(n_groups)):
        with cols[i % 2]:
            g_name = st.text_input(f"Group {i+1} name", key=f"gname_{i}", placeholder="e.g. female")
            g_pct = st.number_input(f"Group {i+1} %", min_value=0.0, max_value=100.0,
                                     value=round(100.0 / n_groups, 1), key=f"gpct_{i}")
            if g_name:
                group_inputs[g_name.lower().strip()] = g_pct / 100.0

    if st.button("💾 Save Reference Distribution", type="primary"):
        if not ref_name or not col_hint:
            st.error("Please fill in both name and column hint.")
        elif len(group_inputs) < 2:
            st.error("Please define at least 2 groups with names.")
        else:
            try:
                resp = requests.post(
                    f"{API_URL}/references/custom",
                    params={"name": ref_name, "col_hint": col_hint},
                    json=group_inputs,
                    timeout=10,
                )
                resp.raise_for_status()
                st.success(f"✅ Saved '{ref_name}'. It will be used in all future scans automatically.")
                st.rerun()
            except Exception as e:
                st.error(str(e))