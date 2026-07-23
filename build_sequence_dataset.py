import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

MAX_SEQ_LEN = 50   # cap each user's session history at 50 most recent sessions
CHURN_THRESHOLD_DAYS = 60

# --- Load raw session-level data ---
df = pd.read_csv('ChargingRecords.csv')
df['StartDatetime'] = pd.to_datetime(df['StartDatetime'])
sub = df[df['UserID'] != 0].copy()
sub = sub.sort_values(['UserID', 'StartDatetime'])
end_date = sub['StartDatetime'].max()

# --- Per-session features ---
sub['hour_of_day'] = sub['StartDatetime'].dt.hour
sub['day_of_week'] = sub['StartDatetime'].dt.dayofweek

# Encode categoricals as integers (embeddings will handle them in the models)
charger_le = LabelEncoder()
location_le = LabelEncoder()
company_le = LabelEncoder()
sub['charger_type_enc'] = charger_le.fit_transform(sub['ChargerType'].astype(str))
sub['location_enc'] = location_le.fit_transform(sub['Location'].astype(str))
sub['company_enc'] = company_le.fit_transform(sub['ChargerCompany'].astype(str))

# Gap since previous session (hours) -- key sequential signal
sub['prev_session_time'] = sub.groupby('UserID')['StartDatetime'].shift(1)
sub['gap_hours'] = (sub['StartDatetime'] - sub['prev_session_time']).dt.total_seconds() / 3600
sub['gap_hours'] = sub['gap_hours'].fillna(0)  # first session per user has no gap

# Scale continuous features
scaler = StandardScaler()
continuous_cols = ['Duration', 'Demand', 'gap_hours', 'hour_of_day', 'day_of_week']
sub[continuous_cols] = scaler.fit_transform(sub[continuous_cols])

# --- Build churn label per user (same definition as before) ---
last_session = sub.groupby('UserID')['StartDatetime'].max()
days_since_last = (end_date - last_session).dt.days
churn_label = (days_since_last > CHURN_THRESHOLD_DAYS).astype(int)

# --- Build padded sequences per user ---
feature_cols = ['Duration', 'Demand', 'gap_hours', 'hour_of_day', 'day_of_week',
                 'charger_type_enc', 'location_enc', 'company_enc']

user_ids = sorted(sub['UserID'].unique())
n_features = len(feature_cols)

sequences = np.zeros((len(user_ids), MAX_SEQ_LEN, n_features), dtype=np.float32)
seq_lengths = np.zeros(len(user_ids), dtype=np.int64)
labels = np.zeros(len(user_ids), dtype=np.int64)

for i, uid in enumerate(user_ids):
    user_sessions = sub[sub['UserID'] == uid][feature_cols].values
    # Keep the most recent MAX_SEQ_LEN sessions
    if len(user_sessions) > MAX_SEQ_LEN:
        user_sessions = user_sessions[-MAX_SEQ_LEN:]
    seq_len = len(user_sessions)
    sequences[i, :seq_len, :] = user_sessions
    seq_lengths[i] = seq_len
    labels[i] = churn_label.loc[uid]

print("Sequence tensor shape:", sequences.shape)
print("Users:", len(user_ids))
print("Feature columns:", feature_cols)
print("Churn rate:", labels.mean())
print("Session count stats -- min:", seq_lengths.min(), "max:", seq_lengths.max(), "mean:", seq_lengths.mean().round(1))

# --- Save everything needed for model building ---
with open('sequence_data.pkl', 'wb') as f:
    pickle.dump({
        'sequences': sequences,
        'seq_lengths': seq_lengths,
        'labels': labels,
        'user_ids': user_ids,
        'feature_cols': feature_cols,
        'max_seq_len': MAX_SEQ_LEN,
        'n_charger_types': len(charger_le.classes_),
        'n_locations': len(location_le.classes_),
        'n_companies': len(company_le.classes_),
    }, f)

print("\nSaved sequence_data.pkl")
