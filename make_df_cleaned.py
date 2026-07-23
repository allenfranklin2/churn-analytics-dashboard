import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

user_agg = pd.read_csv('user_level_data.csv')

categorical_cols = ['common_charger_type', 'common_location', 'common_company']
for col in categorical_cols:
    le = LabelEncoder()
    user_agg[col] = le.fit_transform(user_agg[col])

# Engineered ratio features (must match the tuned model's training pipeline)
user_agg['sessions_per_tenure_day'] = user_agg['session_count'] / user_agg['tenure_days'].replace(0, 1)
user_agg['demand_per_session'] = user_agg['total_demand'] / user_agg['session_count'].replace(0, 1)
user_agg['location_diversity'] = user_agg['num_locations_used'] / user_agg['session_count'].replace(0, 1)

df_cleaned = user_agg.drop(columns=['last_session', 'first_session', 'days_since_last_session'])

with open('df_cleaned.pkl', 'wb') as f:
    pickle.dump(df_cleaned, f)

print("Saved df_cleaned.pkl:", df_cleaned.shape)
