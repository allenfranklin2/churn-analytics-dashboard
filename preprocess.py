import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data
df = pd.read_csv('ev_charging_patterns.csv')

# Parse datetime columns
df['Charging Start Time'] = pd.to_datetime(df['Charging Start Time'])
df['Charging End Time'] = pd.to_datetime(df['Charging End Time'])

# Handle missing values (median imputation for the 3 numeric cols with gaps)
for col in ['Energy Consumed (kWh)', 'Charging Rate (kW)', 'Distance Driven (since last charge) (km)']:
    df[col] = df[col].fillna(df[col].median())

# Drop high-cardinality ID columns not useful as model features
df_model = df.drop(columns=['User ID', 'Charging Station ID', 'Charging Start Time', 'Charging End Time'])

# Encode categorical features
categorical_cols = ['Vehicle Model', 'Charging Station Location', 'Time of Day', 'Day of Week', 'Charger Type']
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    le_dict[col] = le

# Encode target
target_le = LabelEncoder()
df_model['User Type'] = target_le.fit_transform(df_model['User Type'])

# Split features/target
X = df_model.drop(columns=['User Type'])
y = df_model['User Type']

print("Feature matrix shape:", X.shape)
print("Target classes:", dict(zip(target_le.classes_, target_le.transform(target_le.classes_))))
print(X.head())
