import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- Load raw session-level data ---
df = pd.read_csv('ChargingRecords.csv')
df['StartDatetime'] = pd.to_datetime(df['StartDatetime'])

# Exclude UserID 0 -- these are pooled non-subscribed sessions, not individual users,
# so they can't be part of a per-user churn analysis
sub = df[df['UserID'] != 0].copy()

# --- Define churn ---
# Churn = no charging session within the last N days of the observation window
CHURN_THRESHOLD_DAYS = 60
end_date = sub['StartDatetime'].max()

# --- Aggregate to one row per user ---
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

# Churn label (target) -- NOT included as a feature
user_agg['churned'] = (user_agg['days_since_last_session'] > CHURN_THRESHOLD_DAYS).astype(int)

# Most common charger type / location per user (behavioral preference features)
mode_charger = sub.groupby('UserID')['ChargerType'].agg(lambda x: x.mode()[0])
mode_location = sub.groupby('UserID')['Location'].agg(lambda x: x.mode()[0])
mode_company = sub.groupby('UserID')['ChargerCompany'].agg(lambda x: x.mode()[0])

user_agg = user_agg.merge(mode_charger.rename('common_charger_type'), on='UserID')
user_agg = user_agg.merge(mode_location.rename('common_location'), on='UserID')
user_agg = user_agg.merge(mode_company.rename('common_company'), on='UserID')

print("User-level dataset shape:", user_agg.shape)
print()
print("Churn distribution:")
print(user_agg['churned'].value_counts())
print(user_agg['churned'].value_counts(normalize=True).round(3))
print()

# --- Build model-ready feature matrix ---
# Drop leakage-prone / non-feature columns
drop_cols = ['UserID', 'last_session', 'first_session', 'days_since_last_session', 'churned']
X = user_agg.drop(columns=drop_cols)
y = user_agg['churned']

# Encode categoricals
categorical_cols = ['common_charger_type', 'common_location', 'common_company']
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

print("Feature matrix shape:", X.shape)
print("Features:", X.columns.tolist())
print()
print(X.head())

# Save the user-level dataset for reuse in modeling script
user_agg.to_csv('user_level_data.csv', index=False)
print()
print("Saved user_level_data.csv for the modeling step.")
