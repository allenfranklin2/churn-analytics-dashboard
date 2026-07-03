# Explainable AI-Powered Customer Churn Analytics Dashboard

**CMPS 592B — Advanced Topics in Information Systems**
Southern University and Agricultural & Mechanical College
Allen Trevor Franklin | Summer 2026

---

## Project Overview

This project builds an end-to-end explainable AI system that predicts whether a telecom customer is likely to churn and explains the reasoning behind each prediction in plain business language. The system combines machine learning, explainable AI (SHAP), an interactive Streamlit dashboard, and a live Generative AI module powered by the Anthropic API.

---

## Features

- Data cleaning, encoding, and preprocessing of the IBM Telco Customer Churn dataset
- Comparison of four machine learning models: Logistic Regression, Decision Tree, Random Forest, and XGBoost
- SHAP-based explainability showing global feature importance and individual customer-level explanations
- Interactive Streamlit dashboard with six pages
- Live Generative AI module that converts technical SHAP output into plain English business recommendations using the Anthropic Claude API
- Robustness testing covering training set size, class-weight balancing, and decision threshold tuning

---

## Dashboard Pages

1. **Dataset Overview** — summary statistics and a preview of the cleaned dataset
2. **EDA Visualizations** — churn by contract type, tenure, monthly charges, internet service, and a correlation heatmap
3. **Model Performance** — accuracy and full classification report for the Random Forest model
4. **Churn Prediction** — interactive form to enter customer attributes and receive a live prediction
5. **SHAP Explanation** — global feature importance summary plot
6. **AI Recommendation** — select any customer to generate a live Claude-powered business insight and retention recommendation

---

## Results

| Model | Test Accuracy |
|---|---|
| Logistic Regression | 79.0% |
| Random Forest | 79.0% |
| XGBoost | 76.0% |
| Decision Tree | 72.0% |

Random Forest was selected as the primary model for SHAP explainability and dashboard integration.

**Key findings from robustness testing:**
- Model accuracy remained stable between 77-79% regardless of training set size, indicating the churn patterns are learnable from a relatively small sample
- Class-weight balancing did not improve minority-class recall for this dataset
- Lowering the decision threshold from 0.50 to 0.35 improved churn recall from 48% to 68%, a meaningful improvement for real-world retention use cases

---

## Dataset

**IBM Telco Customer Churn Dataset** — available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

- 7,032 customer records after cleaning
- 31 encoded features
- 26.6% churn rate

---

## Project Structure

```
churn-analytics-dashboard/
├── app.py                        # Streamlit dashboard application
├── churn_analysis.ipynb          # Data cleaning, EDA, and model building
├── churn_testing.ipynb           # Robustness testing experiments
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset
├── .gitignore
└── README.md
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/allenfranklin2/churn-analytics-dashboard.git
cd churn-analytics-dashboard
```

### 2. Install dependencies

```bash
pip install pandas scikit-learn xgboost shap streamlit anthropic python-dotenv matplotlib seaborn
```

### 3. Run the notebook to generate model files

Open `churn_analysis.ipynb` in Jupyter and run all cells. This will generate the `.pkl` model and data files needed by the dashboard.

### 4. Set up your API key

Create a `.env` file in the project folder:

```
ANTHROPIC_API_KEY=your_api_key_here
```

### 5. Launch the dashboard

```bash
streamlit run app.py
```

---

## Technologies Used

- **Python** — pandas, scikit-learn, XGBoost, SHAP, matplotlib, seaborn
- **Streamlit** — interactive dashboard framework
- **Anthropic API** — Claude model for generative AI business insights
- **Jupyter Notebook** — data exploration and model development
- **GitHub** — version control and project hosting

---

## Author

Allen Trevor Franklin
Department of Computer Science
Southern University and Agricultural & Mechanical College
Baton Rouge, Louisiana
