"""
LEAKAGE TEST — CatBoost churn model
====================================
This is a modified COPY of save_catboost_pickles.py. It is NOT part of the
real pipeline and does not overwrite any of your real project files.

The only change: `days_since_last_session` is deliberately LEFT IN as a
feature (normally it's excluded to prevent data leakage, since churn is
literally defined as `days_since_last_session > 60`).

Purpose: show that accuracy spikes when this feature is included, proving
the exclusion in your real model was the correct call. Use the printed
Accuracy/AUC on a "Leakage Check" slide, compared against your honest
CatBoost result (69.4% accuracy / 0.700 AUC).

Run from the same folder as ChargingRecords.csv:
    python3 leakage_test_catboost.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

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

# ============================================================
# ONLY CHANGE vs. the real script: days_since_last_session is
# NOT dropped here, so it stays in as a feature (leakage test).
# ============================================================
drop_cols = ['UserID', 'last_session', 'first_session', 'churned']
X = user_agg.drop(columns=drop_cols)
y = user_agg['churned']
cat_feature_idx = [X.columns.get_loc(c) for c in cat_features]

print("Features used in this LEAKY run:", list(X.columns))
print("(compare to your real model's features -- this list has ONE extra: days_since_last_session)\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Same tuned CatBoost params as the real model, for a fair comparison
model = CatBoostClassifier(depth=6, iterations=200, learning_rate=0.01, random_state=42, verbose=0)
model.fit(X_train, y_train, cat_features=cat_feature_idx)

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]
leaky_acc = accuracy_score(y_test, preds)
leaky_auc = roc_auc_score(y_test, probs)

print("=== LEAKY RUN (days_since_last_session included) ===")
print(f"Accuracy: {leaky_acc:.4f}")
print(f"AUC:      {leaky_auc:.4f}")

# Your real, honest, reported CatBoost numbers (from the actual project)
honest_acc = 0.6944
honest_auc = 0.7000

print("\n=== COMPARISON (copy this into your slide) ===")
print(f"{'':20s}{'Accuracy':>12s}{'AUC':>10s}")
print(f"{'Honest (excluded)':20s}{honest_acc*100:>11.1f}%{honest_auc:>10.3f}")
print(f"{'Leaky (included)':20s}{leaky_acc*100:>11.1f}%{leaky_auc:>10.3f}")
print(f"{'Jump':20s}{(leaky_acc-honest_acc)*100:>+11.1f}pp{(leaky_auc-honest_auc):>+10.3f}")

# NOTE: outputs are intentionally NOT saved as catboost_model.pkl / catboost_X_train.pkl /
# etc. -- this script only prints results, so it can never overwrite your real pipeline's
# pickles even if run from the same folder.
