import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# --- Load user-level data ---
user_agg = pd.read_csv('user_level_data.csv')

categorical_cols = ['common_charger_type', 'common_location', 'common_company']
for col in categorical_cols:
    le = LabelEncoder()
    user_agg[col] = le.fit_transform(user_agg[col])

drop_cols = ['UserID', 'last_session', 'first_session', 'days_since_last_session', 'churned']
X = user_agg.drop(columns=drop_cols)
y = user_agg['churned']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train Random Forest (best AUC model) ---
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# --- SHAP explainability ---
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# shap_values for binary classification: shap_values[1] = contributions toward "Churned" class
# (some shap versions return a single 3D array instead of a list -- handle both)
if isinstance(shap_values, list):
    shap_vals_churn = shap_values[1]
else:
    shap_vals_churn = shap_values[:, :, 1]

# --- Global feature importance (summary bar plot) ---
plt.figure()
shap.summary_plot(shap_vals_churn, X_test, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig('shap_summary_bar.png', dpi=150)
plt.close()

# --- Full summary plot (beeswarm) ---
plt.figure()
shap.summary_plot(shap_vals_churn, X_test, show=False)
plt.tight_layout()
plt.savefig('shap_summary_beeswarm.png', dpi=150)
plt.close()

# --- Print average absolute SHAP value per feature (ranked) ---
mean_abs_shap = pd.DataFrame({
    'feature': X_test.columns,
    'mean_abs_shap': np.abs(shap_vals_churn).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

print("Feature importance ranked by mean |SHAP value|:")
print(mean_abs_shap.to_string(index=False))

# --- Save everything needed for the Streamlit dashboard ---
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
with open('explainer.pkl', 'wb') as f:
    pickle.dump(explainer, f)
with open('shap_values.pkl', 'wb') as f:
    pickle.dump(shap_vals_churn, f)
X_train.to_pickle('X_train.pkl')
X_test.to_pickle('X_test.pkl')
y_train.to_pickle('y_train.pkl')
y_test.to_pickle('y_test.pkl')

print("\nSaved: rf_model.pkl, explainer.pkl, shap_values.pkl, X_train.pkl, X_test.pkl, y_train.pkl, y_test.pkl")
print("Saved plots: shap_summary_bar.png, shap_summary_beeswarm.png")
