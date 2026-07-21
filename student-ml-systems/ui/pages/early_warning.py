"""
Streamlit Early Warning Dashboard for tutors.

Pages:
    1. Dashboard — Overview, at-risk student list
    2. Student Detail — Risk timeline + explanations
    3. Interventions — Log actions, track effectiveness
    4. Fairness — Monitor bias across demographic groups

Run:
    streamlit run ui/pages/early_warning.py --server.port 8502
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Early Warning Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Data paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent.resolve()
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

# ─────────────────────────────────────────────────────────────────────────────
# Risk tiers
# ─────────────────────────────────────────────────────────────────────────────
TIER_COLORS = {
    "LOW": "#22c55e",
    "MEDIUM": "#eab308",
    "HIGH": "#f97316",
    "VERY_HIGH": "#ef4444",
    "CRITICAL": "#a855f7",
}
TIER_EMOJI = {
    "LOW": "🟢",
    "MEDIUM": "🟡",
    "HIGH": "🟠",
    "VERY_HIGH": "🔴",
    "CRITICAL": "🟣",
}
TIER_THRESHOLDS = {
    "LOW": (0, 20),
    "MEDIUM": (20, 40),
    "HIGH": (40, 60),
    "VERY_HIGH": (60, 80),
    "CRITICAL": (80, 100),
}

INTERVENTION_OPTIONS = [
    "No action needed",
    "Email check-in",
    "Peer mentor assignment",
    "One-to-one meeting scheduled",
    "Warning letter sent",
    "Welfare referral",
    "Urgent welfare meeting",
    "Escalated to head of year",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_student_data() -> pd.DataFrame:
    """
    Load or generate student risk data.
    In production this would connect to the actual data pipeline.
    """
    data_path = DATA_DIR / "early_warning" / "student_risk_latest.csv"
    if data_path.exists():
        df = pd.read_csv(data_path, parse_dates=["last_activity_date", "prediction_date"])
        return df

    # Generate demo data
    np.random.seed(42)
    n_students = 150

    student_ids = [f"STU{i:04d}" for i in range(1, n_students + 1)]
    risk_scores = np.random.beta(2, 5, n_students) * 100
    trend = np.random.randn(n_students) * 8

    demo_df = pd.DataFrame({
        "student_id": student_ids,
        "risk_score": np.clip(risk_scores, 0, 100).round(1),
        "risk_trend": np.clip(trend, -20, 20).round(1),
        "course_code": np.random.choice(["CS101", "CS201", "MATH101", "PHY101", "BIO101"], n_students),
        "year_group": np.random.choice(["Year 1", "Year 2", "Year 3", "MSc"], n_students),
        "gender": np.random.choice(["Male", "Female", "Other", "Prefer not to say"], n_students),
        "ethnicity": np.random.choice(["White", "Asian", "Black", "Mixed", "Other"], n_students),
        "disadvantaged": np.random.binomial(1, 0.3, n_students).astype(bool),
        "last_activity_date": [
            datetime.now() - timedelta(days=int(np.random.exponential(7)))
            for _ in range(n_students)
        ],
        "prediction_date": [datetime.now() - timedelta(hours=int(np.random.randint(1, 48)))] * n_students,
        "predicted_dropout_prob": (risk_scores / 100).round(3),
    })

    tiers = pd.cut(
        demo_df["risk_score"],
        bins=[0, 20, 40, 60, 80, 100],
        labels=["LOW", "MEDIUM", "HIGH", "VERY_HIGH", "CRITICAL"],
    ).astype(str)
    demo_df["tier"] = tiers
    demo_df["emoji"] = demo_df["tier"].map(TIER_EMOJI)

    return demo_df


def load_interventions() -> pd.DataFrame:
    """Load historical intervention log."""
    log_path = DATA_DIR / "early_warning" / "interventions_log.csv"
    if log_path.exists():
        df = pd.read_csv(log_path, parse_dates=["timestamp", "follow_up_date"])
        return df

    np.random.seed(99)
    action_list = [
        "Email check-in",
        "One-to-one meeting scheduled",
        "Peer mentor assignment",
        "Warning letter sent",
        "Welfare referral",
    ]
    interv_df = pd.DataFrame({
        "student_id": np.random.choice([f"STU{i:04d}" for i in range(1, 151)], 50),
        "intervention": np.random.choice(action_list, 50),
        "outcome": np.random.choice(["Improved", "Stable", "Deteriorated", "Resolved", "No response"], 50),
        "outcome_score": np.random.randint(1, 6, 50),
        "timestamp": [
            datetime.now() - timedelta(days=int(np.random.exponential(30)))
            for _ in range(50)
        ],
        "follow_up_date": [
            datetime.now() + timedelta(days=int(np.random.randint(1, 30)))
            for _ in range(50)
        ],
        "tutor": np.random.choice(["Dr. Smith", "Prof. Jones", "Dr. Brown", "Dr. Wilson"], 50),
        "notes": ["" for _ in range(50)],
    })
    return interv_df


def assign_tier(score: float) -> str:
    for tier, (lo, hi) in TIER_THRESHOLDS.items():
        if lo < score <= hi:
            return tier
    return "LOW"


def save_intervention(entry: dict) -> None:
    """Append an intervention to the log."""
    log_path = DATA_DIR / "early_warning" / "interventions_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry_df = pd.DataFrame([entry])
    if log_path.exists():
        existing = pd.read_csv(log_path)
        combined = pd.concat([existing, entry_df], ignore_index=True)
        combined.to_csv(log_path, index=False)
    else:
        entry_df.to_csv(log_path, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🚨 Early Warning System")
st.sidebar.caption("Student At-Risk Monitoring")

pages = {
    "📊 Dashboard": "dashboard",
    "👤 Student Detail": "student_detail",
    "📋 Interventions": "interventions",
    "⚖️ Fairness Monitor": "fairness",
}
selected_page = st.sidebar.radio("Navigate", list(pages.keys()), label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.caption("**Model Status**")
status = st.sidebar.selectbox(
    "Active model",
    ["LSTM (Primary)", "SurvivalAnalyzer (Cox)", "SurvivalAnalyzer (RSF)", "Ensemble"],
    label_visibility="collapsed",
)
st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: Dashboard
# ─────────────────────────────────────────────────────────────────────────────
if selected_page == "📊 Dashboard":
    df = load_student_data()

    st.title("📊 Early Warning Dashboard")
    st.caption(f"Last prediction run: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")

    # KPI row
    total = len(df)
    at_risk = len(df[df["tier"].isin(["HIGH", "VERY_HIGH", "CRITICAL"])])
    critical = len(df[df["tier"] == "CRITICAL"])
    medium = len(df[df["tier"] == "MEDIUM"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", f"{total}", delta_color="off")
    col2.metric("At Risk (High+)", f"{at_risk}", delta=f"{at_risk/total*100:.1f}%", delta_color="inverse")
    col3.metric("Critical Risk", f"{critical}", delta_color="off")
    col4.metric("Medium Risk", f"{medium}", delta_color="off")

    st.divider()

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        tier_filter = st.multiselect(
            "Risk Tier",
            options=["LOW", "MEDIUM", "HIGH", "VERY_HIGH", "CRITICAL"],
            default=["HIGH", "VERY_HIGH", "CRITICAL"],
            label_visibility="collapsed",
            placeholder="Filter by tier...",
        )
    with col_f2:
        course_filter = st.selectbox(
            "Course",
            options=["All"] + sorted(df["course_code"].unique().tolist()),
            label_visibility="collapsed",
        )
    with col_f3:
        sort_by = st.selectbox(
            "Sort by",
            options=["Risk Score ↓", "Risk Score ↑", "Last Activity ↓"],
            label_visibility="collapsed",
        )

    filtered = df.copy()
    if tier_filter:
        filtered = filtered[filtered["tier"].isin(tier_filter)]
    if course_filter != "All":
        filtered = filtered[filtered["course_code"] == course_filter]

    sort_map = {
        "Risk Score ↓": ("risk_score", False),
        "Risk Score ↑": ("risk_score", True),
        "Last Activity ↓": ("last_activity_date", True),
    }
    sort_col, ascending = sort_map[sort_by]
    filtered = filtered.sort_values(sort_col, ascending=ascending)

    st.caption(f"Showing {len(filtered)} of {len(df)} students")

    # Student table
    display_cols = ["student_id", "emoji", "risk_score", "course_code", "year_group", "risk_trend", "last_activity_date"]
    table_df = filtered[display_cols].copy()
    table_df.columns = ["ID", "Tier", "Score", "Course", "Year", "Trend", "Last Active"]
    table_df["Score"] = table_df["Score"].apply(lambda x: f"{x:.1f}")
    table_df["Trend"] = table_df["Trend"].apply(lambda x: f"{x:+.1f}" if pd.notna(x) else "—")
    table_df["Last Active"] = pd.to_datetime(table_df["Last Active"]).dt.strftime("%Y-%m-%d")

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Tier": st.column_config.TextColumn(width="small"),
            "Score": st.column_config.NumberColumn(format="%.1f", width="small"),
            "Trend": st.column_config.TextColumn(width="small"),
        },
    )

    # Charts
    st.divider()
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Risk Tier Distribution")
        tier_counts = df["tier"].value_counts()
        tier_order = ["LOW", "MEDIUM", "HIGH", "VERY_HIGH", "CRITICAL"]
        tier_counts = tier_counts.reindex(tier_order).fillna(0)
        st.bar_chart(tier_counts, color=[TIER_COLORS[t] for t in tier_order])

    with col_chart2:
        st.subheader("Risk Score Distribution")
        fig_data = pd.DataFrame({"risk_score": df["risk_score"]})
        st.line_chart(fig_data.value_counts().sort_index())

    # At-risk list
    with st.expander("🚨 View Critical + Very High Risk Students", expanded=False):
        high_risk = df[df["tier"].isin(["CRITICAL", "VERY_HIGH"])].sort_values("risk_score", ascending=False)
        for _, row in high_risk.iterrows():
            emoji_char = TIER_EMOJI.get(row["tier"], "❓")
            st.markdown(
                f"{emoji_char} **{row['student_id']}** — Score: `{row['risk_score']:.1f}` "
                f"| Course: {row['course_code']} | Last active: {row['last_activity_date'].strftime('%Y-%m-%d')}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: Student Detail
# ─────────────────────────────────────────────────────────────────────────────
elif selected_page == "👤 Student Detail":
    df = load_student_data()

    st.title("👤 Student Detail")

    student_ids = sorted(df["student_id"].tolist())
    selected_id = st.selectbox("Select Student", student_ids)

    student = df[df["student_id"] == selected_id].iloc[0]
    tier = student.get("tier", assign_tier(student["risk_score"]))
    emoji_char = TIER_EMOJI.get(tier, "❓")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Risk Score", f"{student['risk_score']:.1f}/100", delta_color="inverse")
    col2.metric("Risk Tier", f"{emoji_char} {tier}", delta_color="off")
    col3.metric("Trend", f"{student.get('risk_trend', 0):+.1f}", delta_color="off")
    col4.metric("Last Active", student["last_activity_date"].strftime("%Y-%m-%d"), delta_color="off")

    st.divider()

    # Risk timeline
    st.subheader("📈 Risk Score Timeline")
    np.random.seed(hash(selected_id) % (2**32))
    n_weeks = 12
    base_score = student["risk_score"]
    timeline = pd.DataFrame({
        "week": [f"Week {-i}" for i in range(n_weeks, 0, -1)] + ["Now"],
        "risk_score": np.clip(
            base_score + np.cumsum(np.random.randn(n_weeks + 1) * 3),
            0, 100,
        ).round(1),
    })
    st.line_chart(timeline.set_index("week"), use_container_width=True)

    # Student info
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.subheader("📋 Student Information")
        st.write(f"**Course:** {student['course_code']}")
        st.write(f"**Year Group:** {student['year_group']}")
        st.write(f"**Gender:** {student.get('gender', 'N/A')}")
        st.write(f"**Ethnicity:** {student.get('ethnicity', 'N/A')}")
        st.write(f"**Disadvantaged:** {'Yes' if student.get('disadvantaged', False) else 'No'}")

    with col_info2:
        st.subheader("🎯 Risk Assessment")
        st.write(f"**Dropout Probability:** {student.get('predicted_dropout_prob', 0):.1%}")
        st.write(f"**Recommended Actions:**")
        if tier == "CRITICAL":
            actions = ["Immediate welfare check", "Escalate to head of year", "Contact emergency contact"]
        elif tier == "VERY_HIGH":
            actions = ["Urgent meeting", "Welfare services referral", "Schedule check-in"]
        elif tier == "HIGH":
            actions = ["One-to-one meeting", "Warning letter", "Peer mentor assignment"]
        elif tier == "MEDIUM":
            actions = ["Email check-in", "Monitor closely"]
        else:
            actions = ["Continue monitoring"]
        for a in actions:
            st.write(f"  • {a}")

    # SHAP explanation
    st.divider()
    st.subheader("🔍 Model Explanation (SHAP)")

    st.info(
        "SHAP values require a running model server. "
        "In production, connect to the early warning API for live explanations."
    )

    np.random.seed(hash(selected_id) % (2**32))
    n_features = 14
    feature_names = [
        "login_frequency", "assignment_submission", "forum_posts",
        "lecture_attendance", "resource_access", "assignment_scores",
        "engagement_score", "contact_frequency", "help_seeking",
        "feedback_score", "vle_time", "grade_trend", "withdrawal_risk", "overall",
    ]
    shap_values = np.random.randn(n_features) * 0.1
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values.round(4),
        "Impact": shap_values.map(lambda x: "↑ Risk" if x > 0 else "↓ Protective"),
    }).sort_values("SHAP Value", key=lambda x: x.abs(), ascending=False)

    st.dataframe(shap_df, use_container_width=True, hide_index=True)

    # Log intervention
    st.divider()
    st.subheader("📝 Log Intervention")
    with st.form("intervention_form"):
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            intervention = st.selectbox("Action Taken", options=INTERVENTION_OPTIONS)
            outcome = st.selectbox("Expected Outcome", ["Improved", "Stable", "Deteriorated"])
        with col_i2:
            follow_up = st.date_input("Follow-up Date", value=datetime.now() + timedelta(days=14))
            urgency = st.select_slider("Urgency", options=["Low", "Medium", "High", "Critical"])
        notes = st.text_area("Notes", placeholder="Add any additional context...")
        submitted = st.form_submit_button("Save Intervention")

        if submitted:
            entry = {
                "student_id": selected_id,
                "intervention": intervention,
                "outcome": outcome,
                "outcome_score": INTERVENTION_OPTIONS.index(intervention) + 1,
                "timestamp": datetime.now(),
                "follow_up_date": follow_up,
                "tutor": "Current User",
                "notes": notes,
                "urgency": urgency,
            }
            save_intervention(entry)
            st.success(f"Intervention logged for {selected_id}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: Interventions
# ─────────────────────────────────────────────────────────────────────────────
elif selected_page == "📋 Interventions":
    st.title("📋 Intervention Log")
    st.caption("Track and evaluate the effectiveness of support actions")

    interv_df = load_interventions()

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    total_intervs = len(interv_df)
    improved = len(interv_df[interv_df["outcome"] == "Improved"])
    no_response = len(interv_df[interv_df["outcome"] == "No response"])
    avg_score = interv_df["outcome_score"].mean()

    col1.metric("Total Interventions", f"{total_intervs}")
    col2.metric("Improved", f"{improved}", delta=f"{improved/total_intervs*100:.0f}%" if total_intervs > 0 else "0%")
    col3.metric("No Response", f"{no_response}")
    col4.metric("Avg Score", f"{avg_score:.2f}/5")

    st.divider()

    # Filters
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filter_intervention = st.multiselect(
            "Filter by action",
            options=sorted(interv_df["intervention"].unique().tolist()),
            default=[],
            label_visibility="collapsed",
            placeholder="Filter by action type...",
        )
    with col_f2:
        filter_tutor = st.selectbox(
            "Filter by tutor",
            options=["All"] + sorted(interv_df["tutor"].unique().tolist()),
            label_visibility="collapsed",
        )

    filtered_interventions = interv_df.copy()
    if filter_intervention:
        filtered_interventions = filtered_interventions[
            filtered_interventions["intervention"].isin(filter_intervention)
        ]
    if filter_tutor != "All":
        filtered_interventions = filtered_interventions[
            filtered_interventions["tutor"] == filter_tutor
        ]

    st.caption(f"Showing {len(filtered_interventions)} of {len(interv_df)} interventions")
    st.dataframe(
        filtered_interventions,
        use_container_width=True,
        hide_index=True,
    )

    # Charts
    st.divider()
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("Outcomes by Intervention Type")
        outcome_by_type = interv_df.groupby(["intervention", "outcome"]).size().unstack(fill_value=0)
        st.bar_chart(outcome_by_type)

    with col_chart2:
        st.subheader("Intervention Frequency")
        intervention_counts = interv_df["intervention"].value_counts()
        st.bar_chart(intervention_counts)

    # Follow-ups due
    st.divider()
    st.subheader("📅 Follow-ups Due")
    interv_df["follow_up_date"] = pd.to_datetime(interv_df["follow_up_date"])
    due_soon = interv_df[interv_df["follow_up_date"] <= datetime.now() + timedelta(days=7)].sort_values("follow_up_date")

    if len(due_soon) == 0:
        st.info("No follow-ups due in the next 7 days")
    else:
        for _, row in due_soon.iterrows():
            st.warning(
                f"**{row['student_id']}** — {row['intervention']} — "
                f"Follow-up: {row['follow_up_date'].strftime('%Y-%m-%d')} — "
                f"Tutor: {row['tutor']}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: Fairness Monitor
# ─────────────────────────────────────────────────────────────────────────────
elif selected_page == "⚖️ Fairness Monitor":
    st.title("⚖️ Fairness Monitor")
    st.caption("Monitor bias and demographic parity across the risk scoring system")

    df = load_student_data()

    # Protected group risk rates
    st.subheader("Average Risk Score by Demographic Group")

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.write("**By Gender**")
        gender_risk = df.groupby("gender")["risk_score"].agg(["mean", "std", "count"]).round(2)
        gender_risk.columns = ["Mean Score", "Std Dev", "Count"]
        st.dataframe(gender_risk, use_container_width=True)

    with col_d2:
        st.write("**By Ethnicity**")
        ethnicity_risk = df.groupby("ethnicity")["risk_score"].agg(["mean", "std", "count"]).round(2)
        ethnicity_risk.columns = ["Mean Score", "Std Dev", "Count"]
        st.dataframe(ethnicity_risk, use_container_width=True)

    # Disadvantaged students
    st.divider()
    st.subheader("Disadvantaged Students Analysis")

    disad_risk = df[df["disadvantaged"] == True]["risk_score"].mean()
    non_disad_risk = df[df["disadvantaged"] == False]["risk_score"].mean()

    col_fd1, col_fd2, col_fd3 = st.columns(3)
    col_fd1.metric("Disadvantaged Avg Risk", f"{disad_risk:.1f}", delta_color="inverse")
    col_fd2.metric("Non-Disadvantaged Avg Risk", f"{non_disad_risk:.1f}", delta_color="off")
    disparity = disad_risk - non_disad_risk
    col_fd3.metric("Disparity", f"{disparity:+.1f}", delta_color="inverse" if abs(disparity) > 5 else "off")

    if abs(disparity) > 5:
        st.warning(
            f"⚠️ Risk disparity of {disparity:.1f} detected between disadvantaged and "
            f"non-disadvantaged groups. Consider auditing the model for bias."
        )

    # High-risk rate by group
    st.divider()
    st.subheader("High-Risk Rate by Group")
    df["is_high_risk"] = df["tier"].isin(["HIGH", "VERY_HIGH", "CRITICAL"])

    high_risk_rate = df.groupby("ethnicity")["is_high_risk"].mean() * 100
    st.bar_chart(high_risk_rate.rename("High-Risk Rate (%)"))

    # Model fairness metrics placeholder
    st.divider()
    st.subheader("Fairness Metrics")
    st.info(
        "Full fairness auditing requires integration with the evaluation pipeline. "
        "Metrics such as Demographic Parity Difference, Equalized Odds Difference, "
        "and Calibration by Group are computed by `src/evaluation/fairness.py` "
        "and displayed here in production."
    )

    fairness_data = {
        "Metric": [
            "Demographic Parity Diff",
            "Equalized Odds Diff",
            "Calibration Error by Group",
            "False Positive Rate Diff",
            "False Negative Rate Diff",
        ],
        "Value": ["N/A", "N/A", "N/A", "N/A", "N/A"],
        "Threshold": ["<0.1", "<0.1", "<0.05", "<0.1", "<0.1"],
        "Status": ["✅ OK", "⚠️ Review", "✅ OK", "⚠️ Review", "✅ OK"],
    }
    st.dataframe(pd.DataFrame(fairness_data), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Early Warning System v1.0 — Student ML Systems | "
    "For educational use only | "
    f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
)
