import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import pickle

# --- Build dataset from raw sessions (60-day churn threshold) ---
df = pd.read_csv('ChargingRecords.csv')
df['StartDatetime'] = pd.to_datetime(df['StartDatetime'])
sub = df[df['UserID'] != 0].copy()
end_date = sub['StartDatetime'].max()

user_agg = sub.groupby('UserID').agg(
    last_session=('StartDatetime', 'max'),
    first_session=('StartDatetime', 'min'),
    session_count=('StartDatetime', 'count'),
    avg_duration=('Duration', 'mean'),
    avg_demand=('Demand', 'mean'),
    total_demand=('Demand', 'sum'),
    num_chargers_used=('ChargerID', 'nunique'),
    num_locations_used=('Location', 'nunique'),
).reset_index()

user_agg['tenure_days'] = (user_agg['last_session'] - user_agg['first_session']).dt.days
user_agg['days_since_last_session'] = (end_date - user_agg['last_session']).dt.days
user_agg['churned'] = (user_agg['days_since_last_session'] > 60).astype(int)

mode_charger = sub.groupby('UserID')['ChargerType'].agg(lambda x: x.mode()[0])
mode_location = sub.groupby('UserID')['Location'].agg(lambda x: x.mode()[0])
mode_company = sub.groupby('UserID')['ChargerCompany'].agg(lambda x: x.mode()[0])
user_agg = user_agg.merge(mode_charger.rename('common_charger_type'), on='UserID')
user_agg = user_agg.merge(mode_location.rename('common_location'), on='UserID')
user_agg = user_agg.merge(mode_company.rename('common_company'), on='UserID')

user_agg['sessions_per_tenure_day'] = user_agg['session_count'] / user_agg['tenure_days'].replace(0, 1)
user_agg['demand_per_session'] = user_agg['total_demand'] / user_agg['session_count'].replace(0, 1)
user_agg['location_diversity'] = user_agg['num_locations_used'] / user_agg['session_count'].replace(0, 1)

# CatBoost handles categoricals natively -- keep as strings, no LabelEncoder
cat_features = ['common_charger_type', 'common_location', 'common_company']
for col in cat_features:
    user_agg[col] = user_agg[col].astype(str)

drop_cols = ['UserID', 'last_session', 'first_session', 'days_since_last_session', 'churned']
X = user_agg.drop(columns=drop_cols)
y = user_agg['churned']
cat_feature_idx = [X.columns.get_loc(c) for c in cat_features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Tuned CatBoost params (from GridSearchCV)
model = CatBoostClassifier(depth=6, iterations=200, learning_rate=0.01, random_state=42, verbose=0)
model.fit(X_train, y_train, cat_features=cat_feature_idx)

from sklearn.metrics import accuracy_score, roc_auc_score
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]
print("Accuracy:", accuracy_score(y_test, preds))
print("AUC:", roc_auc_score(y_test, probs))

# SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig('shap_summary_beeswarm_catboost.png', dpi=150)
plt.close()

# Save everything app.py needs
with open('catboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)
X_train.to_pickle('catboost_X_train.pkl')
X_test.to_pickle('catboost_X_test.pkl')
y_train.to_pickle('catboost_y_train.pkl')
y_test.to_pickle('catboost_y_test.pkl')

# df_cleaned for the Dataset Overview / EDA pages
df_cleaned_cb = user_agg.drop(columns=['last_session', 'first_session', 'days_since_last_session'])
with open('df_cleaned_catboost.pkl', 'wb') as f:
    pickle.dump(df_cleaned_cb, f)

print("Saved: catboost_model.pkl, catboost_X_train.pkl, catboost_X_test.pkl,")
print("       catboost_y_train.pkl, catboost_y_test.pkl, df_cleaned_catboost.pkl")
print("Saved plot: shap_summary_beeswarm_catboost.png")
