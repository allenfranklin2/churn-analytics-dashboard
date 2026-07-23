import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

st.set_page_config(page_title="Explainable AI-Powered EV Charging Analytics Dashboard with Generative AI Insights", layout="wide")

# ---- Style sidebar nav radio: blue ring, white center dot when selected ----
st.markdown(
    """
    <style>
    div[data-testid="stSidebar"] input[type="radio"]:checked + div {
        background-color: transparent !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label div:first-child {
        border-color: #58b6e7 !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label div:first-child div {
        background-color: #ffffff !important;
    }
    button[kind="primary"]:focus, button[kind="primary"]:active,
    button[kind="primary"]:focus:not(:active) {
        box-shadow: 0 0 0 0.2rem rgba(255, 255, 255, 0.55) !important;
        border-color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Shared styling helpers ----
BLUE = "#58b6e7"
GOLD = "#ffce34"
CARD_BG = "#1c2333"

def styled_metric(label, value, accent="blue"):
    """Render a clean metric card with consistent branding, replacing st.metric."""
    color = GOLD if accent == "gold" else BLUE
    st.markdown(
        f"""
        <div style='background-color:{CARD_BG}; border-radius:10px; padding:16px 18px; margin-bottom:6px;'>
            <p style='color:#9a9a9a; font-size:0.85rem; margin:0 0 4px 0;'>{label}</p>
            <p style='color:{color}; font-size:2rem; font-weight:700; margin:0;'>{value}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def page_header(title, subtitle=None):
    """Consistent branded header banner for every page."""
    subtitle_html = f"<p style='color:#9fcdec; margin:8px 0 0 0; font-size:0.95rem;'>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div style='background:#001E40; border-radius:12px; padding:24px 28px; margin-bottom:20px; position:relative; overflow:hidden;'>
            <div style='position:absolute; top:0; left:0; bottom:0; width:6px; background:{GOLD};'></div>
            <div style='position:absolute; top:0; right:0; bottom:0; width:6px; background:{GOLD};'></div>
            <h1 style='color:#ffffff; margin:0; font-size:1.9rem;'>{title}</h1>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True
    )

def section_divider():
    """Consistent spaced divider between page sections."""
    st.markdown(
        f"<hr style='border:none; border-top:1px solid #2a3142; margin:28px 0;'>",
        unsafe_allow_html=True
    )

def page_footer():
    """Small branded footer card, shown at the bottom of every page."""
    today = datetime.now().strftime("%B %d, %Y")
    st.markdown(
        f"""
        <hr style='border:none; border-top:1px solid #2a3142; margin:40px 0 16px 0;'>
        <div style='display:inline-block; background:#001E40; border-radius:10px; padding:12px 18px; border-top:3px solid {GOLD};'>
            <p style='color:#ffffff; font-weight:600; font-size:0.9rem; margin:0;'>Allen Trevor Franklin</p>
            <p style='color:{BLUE}; font-size:0.8rem; margin:2px 0 0 0;'>Southern University and Agricultural &amp; Mechanical College</p>
            <p style='color:#9a9a9a; font-size:0.75rem; margin:2px 0 0 0;'>Department of Computer Science &middot; CMPS 592B &middot; {today}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---- Load saved data and models ----
@st.cache_resource
def load_data():
    with open('df_cleaned_catboost.pkl', 'rb') as f:
        df = pickle.load(f)
    with open('catboost_model.pkl', 'rb') as f:
        cb_model = pickle.load(f)
    with open('catboost_X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('catboost_X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open('catboost_y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
    return df, cb_model, X_train, X_test, y_test

df, cb_model, X_train, X_test, y_test = load_data()

# Categorical feature indices, needed for CatBoost's TreeExplainer calls on new data
cat_features = ['common_charger_type', 'common_location', 'common_company']
cat_feature_idx = [X_train.columns.get_loc(c) for c in cat_features]

# ---- Sidebar navigation ----
st.sidebar.title("EV Charging Analytics")
st.sidebar.caption("Explainable AI-Powered Dashboard with Generative AI Insights")
st.sidebar.caption("CMPS 592B Project")
st.sidebar.divider()
st.sidebar.subheader("Navigation")
page = st.sidebar.radio("Go to", [
    "Electricity Demand Prediction",
    "Dataset Overview",
    "EDA Visualizations",
    "Model Performance",
    "Churn Prediction",
    "SHAP Explanation",
    "AI Recommendation",
    "Hybrid Model (Advanced)"
])

# ---- Page 1: Dataset Overview ----
if page == "Dataset Overview":
    page_header("Dataset Overview", "Phase 1 &middot; Methodology &amp; churn exploration")
    st.write(f"**Total Subscribed Users:** {df.shape[0]}")
    st.write(f"**Total Features:** {df.shape[1]}")
    st.dataframe(df.head(20))

    section_divider()

    col1, col2 = st.columns(2)
    with col1:
        styled_metric("Churn Rate", f"{df['churned'].mean()*100:.1f}%")
    with col2:
        styled_metric("Retention Rate", f"{(1-df['churned'].mean())*100:.1f}%")

    st.caption(
        "Churn is defined as a subscribed user having no charging session in the "
        "final 60 days of the one-year observation window (Sept 2021-Sept 2022)."
    )

# ---- Page 2: EDA Visualizations ----
elif page == "EDA Visualizations":
    page_header("Exploratory Data Analysis", "Churn patterns across tenure, usage, and demand")

    st.subheader("Churn Count")
    fig, ax = plt.subplots()
    sns.countplot(x='churned', data=df, ax=ax)
    ax.set_xticklabels(['Not Churned', 'Churned'])
    st.pyplot(fig)

    section_divider()

    st.subheader("Churn by Tenure (days)")
    fig, ax = plt.subplots()
    sns.histplot(data=df, x='tenure_days', hue='churned', multiple='stack', bins=30, ax=ax)
    st.pyplot(fig)

    section_divider()

    st.subheader("Churn by Total Energy Demand")
    fig, ax = plt.subplots()
    sns.histplot(data=df, x='total_demand', hue='churned', multiple='stack', bins=30, ax=ax)
    st.pyplot(fig)

    section_divider()

    st.subheader("Churn by Session Count")
    fig, ax = plt.subplots()
    sns.histplot(data=df, x='session_count', hue='churned', multiple='stack', bins=30, ax=ax)
    st.pyplot(fig)

    section_divider()

    st.subheader("Correlation Heatmap (numeric features)")
    numeric_df = df.select_dtypes(include='number').drop(columns=['UserID'])
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(numeric_df.corr(), cmap='coolwarm', annot=False, ax=ax)
    st.pyplot(fig)

# ---- Page 3: Model Performance ----
elif page == "Model Performance":
    page_header("Model Performance", "CatBoost churn model evaluation")

    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

    y_pred = cb_model.predict(X_test)
    y_proba = cb_model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    col1, col2 = st.columns(2)
    with col1:
        styled_metric("CatBoost Accuracy", f"{acc*100:.1f}%")
    with col2:
        styled_metric("CatBoost Risk Ranking Score", f"{auc:.3f}")

    report = classification_report(y_test, y_pred, target_names=['Not Churned', 'Churned'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    st.caption(
        "CatBoost was selected over Random Forest, Logistic Regression, and XGBoost. All tuned models "
        "performed within 1-2 points of each other; CatBoost was chosen since it handles categorical "
        "features (charger type, location, provider) natively without needing manual encoding."
    )

# ---- Page 4: Churn Prediction Form ----
elif page == "Churn Prediction":
    page_header("Predict EV User Churn", "Live churn risk prediction")
    st.write("Enter a user's charging behavior below to get a churn risk prediction.")

    session_count = st.slider("Total Charging Sessions", 0, 300, 50)
    avg_duration = st.slider("Average Session Duration (minutes)", 0, 600, 150)
    avg_demand = st.slider("Average Energy Demand per Session (kWh)", 0, 100, 20)
    total_demand = st.number_input("Total Energy Demand (kWh)", 0, 10000, 1000)
    num_chargers_used = st.slider("Number of Distinct Chargers Used", 1, 50, 3)
    num_locations_used = st.slider("Number of Distinct Locations Used", 1, 20, 2)
    tenure_days = st.slider("Tenure (days since first session)", 0, 365, 100)

    if st.button("Predict", type="primary"):
        input_data = pd.DataFrame([X_train.iloc[0]])
        input_data['session_count'] = session_count
        input_data['avg_duration'] = avg_duration
        input_data['avg_demand'] = avg_demand
        input_data['total_demand'] = total_demand
        input_data['num_chargers_used'] = num_chargers_used
        input_data['num_locations_used'] = num_locations_used
        input_data['tenure_days'] = tenure_days
        # Engineered ratio features (must match training pipeline exactly)
        input_data['sessions_per_tenure_day'] = session_count / max(tenure_days, 1)
        input_data['demand_per_session'] = total_demand / max(session_count, 1)
        input_data['location_diversity'] = num_locations_used / max(session_count, 1)

        prediction = cb_model.predict(input_data)[0]
        probability = cb_model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"High Churn Risk: {probability*100:.1f}% probability")
        else:
            st.success(f"Low Churn Risk: {probability*100:.1f}% probability")

# ---- Page 5: SHAP Explanation ----
elif page == "SHAP Explanation":
    page_header("SHAP Explainability", "What drives the churn model's predictions")
    st.write("Feature importance showing what drives EV user churn predictions overall.")

    explainer = shap.TreeExplainer(cb_model)
    shap_values = explainer.shap_values(X_test[:100])

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test[:100], show=False)
    st.pyplot(fig)

# ---- Page 6: AI Recommendation ----
elif page == "AI Recommendation":
    page_header("AI-Generated Business Insight", "Plain-language churn explanations via the Anthropic API")
    st.write("Select a user to get a plain-language explanation of their churn risk and a business recommendation.")

    user_idx = st.number_input("User Index", min_value=0, max_value=len(X_test)-1, value=16)

    if st.button("Generate Insight", type="primary"):
        with st.spinner("Generating insight..."):
            user_row = X_test.iloc[user_idx]
            probability = cb_model.predict_proba(X_test.iloc[[user_idx]])[0][1]

            explainer = shap.TreeExplainer(cb_model)
            shap_values = explainer.shap_values(X_test.iloc[[user_idx]])
            shap_row = shap_values[0]

            top_features_idx = abs(shap_row).argsort()[-5:][::-1]
            top_features = [(X_test.columns[i], user_row.iloc[i], shap_row[i]) for i in top_features_idx]

            feature_summary = "\n".join(
                [f"- {name} = {value} (impact: {'increases' if impact > 0 else 'decreases'} churn risk)"
                 for name, value, impact in top_features]
            )

            prompt = f"""An EV charging network subscriber has a churn probability of {probability*100:.1f}%.

Top factors influencing this prediction:
{feature_summary}

Write 2-3 short, plain-English sentences explaining why this user is likely to churn or stay,
and recommend one specific business action a charging network operator could take. No technical jargon."""

            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )

            insight_text = response.content[0].text

            styled_metric("Churn Probability", f"{probability*100:.1f}%", accent="gold")
            st.subheader("Top Contributing Factors")
            for name, value, impact in top_features:
                direction = "↑ increases risk" if impact > 0 else "↓ decreases risk"
                st.write(f"**{name}** = {value} — {direction}")

            st.subheader("AI Business Insight")
            st.success(insight_text)

# ---- Page 7: Hybrid Model (Advanced) ----
elif page == "Hybrid Model (Advanced)":
    page_header("Hybrid Autoencoder-Transformer", "Advanced deep learning model, built in PyTorch")
    st.write(
        "Advanced deep learning model (built in PyTorch, trained in Google Colab) that reads each "
        "subscriber's full charging session history as a time-ordered sequence, rather than a single "
        "summarized row. Combines an Autoencoder (learns normal charging behavior) with a Transformer "
        "(models how behavior changes over time)."
    )

    @st.cache_resource
    def load_hybrid_predictions():
        return pd.read_csv('hybrid_predictions.csv')

    hybrid_df = load_hybrid_predictions()

    from sklearn.metrics import accuracy_score, roc_auc_score
    acc = accuracy_score(hybrid_df['actual_churn'], hybrid_df['predicted_churn'])
    auc = roc_auc_score(hybrid_df['actual_churn'], hybrid_df['churn_probability'])

    col1, col2, col3 = st.columns(3)
    with col1:
        styled_metric("Hybrid Model Accuracy", f"{acc*100:.1f}%")
    with col2:
        styled_metric("Hybrid Model Risk Ranking Score", f"{auc:.3f}")
    with col3:
        styled_metric("Test Users", f"{len(hybrid_df):,}")

    st.caption(
        "Outperforms all three baseline models (Logistic Regression, CatBoost, XGBoost) on both metrics. "
        "Predictions were generated in Google Colab (this model requires PyTorch, which is not locally "
        "installable on this machine) and exported here for review."
    )

    section_divider()

    st.subheader("Highest Churn-Risk Users")
    top_risk = hybrid_df.sort_values('churn_probability', ascending=False).head(15).reset_index(drop=True)
    top_risk_display = top_risk.copy()
    top_risk_display['churn_probability'] = (top_risk_display['churn_probability'] * 100).round(1).astype(str) + '%'
    top_risk_display['actual_churn'] = top_risk_display['actual_churn'].map({1.0: 'Churned', 0.0: 'Not Churned'})
    st.dataframe(top_risk_display, use_container_width=True)

    section_divider()

    st.subheader("Generate AI Business Insight")
    st.write("Select one of the highest-risk users above to get a plain-language explanation and recommendation.")

    selected_user = st.selectbox("User ID", top_risk['user_id'].tolist())

    if st.button("Generate Insight", key="hybrid_insight_button", type="primary"):
        with st.spinner("Generating insight..."):
            row = hybrid_df[hybrid_df['user_id'] == selected_user].iloc[0]
            probability = row['churn_probability']

            prompt = f"""An EV charging network subscriber has a predicted churn probability of {probability*100:.1f}%.

This prediction comes from a hybrid deep learning model (an Autoencoder combined with a Transformer)
that analyzed this user's full sequence of charging sessions over the past year, including how
frequently they charged, how that frequency changed over time, and how their behavior compared to
typical usage patterns.

Write 2-3 short, plain-English sentences explaining why this user is likely at risk of churning,
and recommend one specific business action a charging network operator could take. No technical jargon."""

            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )

            insight_text = response.content[0].text

            styled_metric("Churn Probability", f"{probability*100:.1f}%", accent="gold")
            st.subheader("AI Business Insight")
            st.success(insight_text)

# ---- Page 8: Electricity Demand Prediction ----
elif page == "Electricity Demand Prediction":
    page_header("Electricity Demand Prediction", "Primary result &middot; XGBoost classifier, explained with SHAP")
    st.write(
        "Predicts whether an EV charging session will draw **high** or **low** electricity demand, "
        "useful for grid load management and capacity planning at charging stations. Built on the same "
        "real-world charging dataset, comparing extreme high-demand sessions against extreme low-demand "
        "sessions (top and bottom thirds), with the ambiguous middle third excluded for a cleaner signal."
    )

    @st.cache_resource
    def load_demand_data():
        with open('demand_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('demand_explainer.pkl', 'rb') as f:
            explainer = pickle.load(f)
        X_train = pd.read_pickle('demand_X_train.pkl')
        X_test = pd.read_pickle('demand_X_test.pkl')
        y_test = pd.read_pickle('demand_y_test.pkl')
        df_cleaned = pd.read_pickle('demand_df_cleaned.pkl')
        return model, explainer, X_train, X_test, y_test, df_cleaned

    demand_model, demand_explainer, demand_X_train, demand_X_test, demand_y_test, demand_df = load_demand_data()

    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

    demand_preds = demand_model.predict(demand_X_test)
    demand_probs = demand_model.predict_proba(demand_X_test)[:, 1]
    acc = accuracy_score(demand_y_test, demand_preds)
    auc = roc_auc_score(demand_y_test, demand_probs)

    col1, col2, col3 = st.columns(3)
    with col1:
        styled_metric("Model Accuracy", f"{acc*100:.1f}%", accent="gold")
    with col2:
        styled_metric("Risk Ranking Score", f"{auc:.3f}", accent="gold")
    with col3:
        styled_metric("Sessions Analyzed", f"{len(demand_df):,}", accent="gold")

    st.caption(
        "Cross-validated at 95.9% accuracy (\u00b1 0.3%) across 5 folds, confirming this holds up beyond "
        "a single train/test split. A Logistic Regression baseline independently reaches 95.2% accuracy, "
        "confirming the result isn't dependent on one specific algorithm."
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Model Performance", "SHAP Explanation", "Try a Prediction", "AI Insight"])

    with tab1:
        report = classification_report(demand_y_test, demand_preds,
                                        target_names=['Low Demand', 'High Demand'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

    with tab2:
        st.write("What drives a session toward high or low electricity demand:")
        fig, ax = plt.subplots()
        shap_values = demand_explainer.shap_values(demand_X_test)
        shap.summary_plot(shap_values, demand_X_test, show=False)
        st.pyplot(fig)
        st.caption(
            "duration_x_charger (charging time weighted by charger type) and charger_type_enc dominate, "
            "which makes physical sense: different charger types (Level 1, Level 2, DC Fast) deliver "
            "power at very different rates."
        )

    with tab3:
        st.write("Enter session details to predict electricity demand level.")
        duration_input = st.slider("Charging Duration (minutes)", 0, 600, 120)
        hour_input = st.slider("Hour of Day", 0, 23, 14)
        day_input = st.slider("Day of Week (0=Monday)", 0, 6, 2)

        # Build a readable-name -> encoded-value mapping directly from the cleaned dataset
        charger_map = demand_df[['ChargerType', 'charger_type_enc']].drop_duplicates().set_index('ChargerType')['charger_type_enc'].to_dict()
        charger_name_map = {0: "Standard Charger (Type 0)", 1: "Fast Charger (Type 1)"}
        charger_display_map = {charger_name_map.get(k, f"Charger Type {k}"): v for k, v in charger_map.items()}
        charger_input = st.selectbox("Charger Type", list(charger_display_map.keys()))
        charger_enc_value = charger_display_map[charger_input]
        st.caption(
            "Labels inferred from data patterns (Type 1 sessions average 37 min for 21.4 kWh vs. Type 0's "
            "191 min for 18.4 kWh, indicating a faster charging rate), not official names from the dataset."
        )

        if st.button("Predict Demand Level", type="primary"):
            is_weekend = 1 if day_input >= 5 else 0
            row = pd.DataFrame([demand_X_train.iloc[0]])
            row['Duration'] = duration_input
            row['hour_of_day'] = hour_input
            row['day_of_week'] = day_input
            row['is_weekend'] = is_weekend
            row['charger_type_enc'] = charger_enc_value
            row['duration_per_hour_bucket'] = duration_input / (hour_input + 1)
            row['duration_x_charger'] = duration_input * (charger_enc_value + 1)

            pred = demand_model.predict(row)[0]
            prob = demand_model.predict_proba(row)[0][1]

            if pred == 1:
                st.error(f"High Demand Session: {prob*100:.1f}% probability")
            else:
                st.success(f"Low Demand Session: {(1-prob)*100:.1f}% probability")

    with tab4:
        st.write("Get a plain-language explanation of what drives demand for a sample high-demand session.")

        if st.button("Generate Insight", key="demand_insight_button", type="primary"):
            # Pick a genuine high-confidence example from the test set
            sample_idx = demand_probs.argmax()
            sample_row = demand_X_test.iloc[sample_idx]
            sample_prob = demand_probs[sample_idx]

            shap_values_sample = demand_explainer.shap_values(demand_X_test.iloc[[sample_idx]])
            shap_row = shap_values_sample[0]
            top_idx = abs(shap_row).argsort()[-4:][::-1]
            top_features = [(demand_X_test.columns[i], sample_row.iloc[i], shap_row[i]) for i in top_idx]

            styled_metric("Predicted Demand Probability", f"{sample_prob*100:.1f}%", accent="gold")

            # ---- Animated bar chart: bars grow from 0 to final SHAP magnitude ----
            st.subheader("Top Contributing Factors")
            chart_placeholder = st.empty()

            names = [f[0] for f in top_features]
            final_vals = [abs(f[2]) for f in top_features]
            colors = ['#028090' if f[2] > 0 else '#A0A0A0' for f in top_features]

            n_frames = 12
            for frame in range(1, n_frames + 1):
                progress = frame / n_frames
                current_vals = [v * progress for v in final_vals]
                fig, ax = plt.subplots(figsize=(6, 2.5))
                ax.barh(names, current_vals, color=colors)
                ax.set_xlim(0, max(final_vals) * 1.15)
                ax.set_xlabel("Impact on prediction")
                fig.tight_layout()
                chart_placeholder.pyplot(fig)
                plt.close(fig)
                time.sleep(0.04)

            for name, value, impact in top_features:
                direction = "↑ increases demand" if impact > 0 else "↓ decreases demand"
                st.write(f"**{name}** = {value} — {direction}")

            # ---- Streaming AI response ----
            feature_summary = "\n".join(
                [f"- {name} = {value} (impact: {'increases' if impact > 0 else 'decreases'} predicted demand)"
                 for name, value, impact in top_features]
            )

            prompt = f"""An EV charging session has a {sample_prob*100:.1f}% predicted probability of being a HIGH electricity demand session.

Top factors influencing this prediction:
{feature_summary}

Write 2-3 short, plain-English sentences explaining why this session is predicted to draw high electricity demand,
and recommend one specific action a charging network operator could take for grid load management or capacity
planning. No technical jargon."""

            st.subheader("AI Business Insight")
            insight_placeholder = st.empty()
            streamed_text = ""

            with client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                for text_chunk in stream.text_stream:
                    streamed_text += text_chunk
                    insight_placeholder.success(streamed_text + "▌")

            insight_placeholder.success(streamed_text)

# ---- Footer (shown on every page) ----
page_footer()
