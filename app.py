import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

st.set_page_config(page_title="Churn Analytics Dashboard", layout="wide")

# ---- Load saved data and models ----
@st.cache_resource
def load_data():
    with open('df_cleaned.pkl', 'rb') as f:
        df = pickle.load(f)
    with open('rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open('y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
    return df, rf_model, X_train, X_test, y_test

df, rf_model, X_train, X_test, y_test = load_data()

# ---- Sidebar navigation ----
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Dataset Overview",
    "EDA Visualizations",
    "Model Performance",
    "Churn Prediction",
    "SHAP Explanation",
    "AI Recommendation"
])

# ---- Page 1: Dataset Overview ----
if page == "Dataset Overview":
    st.title("Dataset Overview")
    st.write(f"**Total Customers:** {df.shape[0]}")
    st.write(f"**Total Features:** {df.shape[1]}")
    st.dataframe(df.head(20))

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Churn Rate", f"{df['Churn'].mean()*100:.1f}%")
    with col2:
        st.metric("Retention Rate", f"{(1-df['Churn'].mean())*100:.1f}%")

# ---- Page 2: EDA Visualizations ----
elif page == "EDA Visualizations":
    st.title("Exploratory Data Analysis")

    st.subheader("Churn Count")
    fig, ax = plt.subplots()
    sns.countplot(x='Churn', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Churn by Tenure")
    fig, ax = plt.subplots()
    sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', bins=30, ax=ax)
    st.pyplot(fig)

    st.subheader("Churn by Monthly Charges")
    fig, ax = plt.subplots()
    sns.histplot(data=df, x='MonthlyCharges', hue='Churn', multiple='stack', bins=30, ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(14,10))
    sns.heatmap(df.corr(), cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# ---- Page 3: Model Performance ----
elif page == "Model Performance":
    st.title("Model Performance")

    from sklearn.metrics import accuracy_score, classification_report

    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.metric("Random Forest Accuracy", f"{acc*100:.1f}%")

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

# ---- Page 4: Churn Prediction Form ----
elif page == "Churn Prediction":
    st.title("Predict Customer Churn")
    st.write("Enter customer details below to get a churn prediction.")

    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 0, 150, 70)
    total_charges = st.number_input("Total Charges ($)", 0, 10000, 1000)
    contract_two_year = st.selectbox("Two Year Contract?", ["No", "Yes"])
    fiber = st.selectbox("Fiber Optic Internet?", ["No", "Yes"])

    if st.button("Predict"):
        input_data = pd.DataFrame([X_train.iloc[0]])
        input_data['tenure'] = tenure
        input_data['MonthlyCharges'] = monthly_charges
        input_data['TotalCharges'] = total_charges
        input_data['Contract_Two year'] = 1 if contract_two_year == "Yes" else 0
        input_data['InternetService_Fiber optic'] = 1 if fiber == "Yes" else 0

        prediction = rf_model.predict(input_data)[0]
        probability = rf_model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"High Churn Risk: {probability*100:.1f}% probability")
        else:
            st.success(f"Low Churn Risk: {probability*100:.1f}% probability")

# ---- Page 5: SHAP Explanation ----
elif page == "SHAP Explanation":
    st.title("SHAP Explainability")
    st.write("Feature importance showing what drives churn predictions overall.")

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test[:100])

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values[:,:,1], X_test[:100], show=False)
    st.pyplot(fig)

# ---- Page 6: AI Recommendation ----
elif page == "AI Recommendation":
    st.title("AI-Generated Business Insight")
    st.write("Select a customer to get a plain-language explanation of their churn risk and a business recommendation.")

    customer_idx = st.number_input("Customer Index (0 to 1406)", min_value=0, max_value=len(X_test)-1, value=16)

    if st.button("Generate Insight"):
        with st.spinner("Generating insight..."):
            customer_row = X_test.iloc[customer_idx]
            probability = rf_model.predict_proba(X_test.iloc[[customer_idx]])[0][1]

            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(X_test.iloc[[customer_idx]])
            shap_row = shap_values[0, :, 1]

            top_features_idx = abs(shap_row).argsort()[-5:][::-1]
            top_features = [(X_test.columns[i], customer_row[i], shap_row[i]) for i in top_features_idx]

            feature_summary = "\n".join(
                [f"- {name} = {value} (impact: {'increases' if impact > 0 else 'decreases'} churn risk)"
                 for name, value, impact in top_features]
            )

            prompt = f"""A customer has a churn probability of {probability*100:.1f}%.

Top factors influencing this prediction:
{feature_summary}

Write 2-3 short, plain-English sentences explaining why this customer is likely to churn or stay,
and recommend one specific business action a manager could take. No technical jargon."""

            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )

            insight_text = response.content[0].text

            st.metric("Churn Probability", f"{probability*100:.1f}%")
            st.subheader("Top Contributing Factors")
            for name, value, impact in top_features:
                direction = "⬆️ increases risk" if impact > 0 else "⬇️ decreases risk"
                st.write(f"**{name}** = {value} — {direction}")

            st.subheader("AI Business Insight")
            st.success(insight_text)
