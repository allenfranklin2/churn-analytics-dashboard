import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# --- Build dataset (60-day threshold, with engineered features) ---
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

for col in ['common_charger_type', 'common_location', 'common_company']:
    le = LabelEncoder()
    user_agg[col] = le.fit_transform(user_agg[col])

user_agg['sessions_per_tenure_day'] = user_agg['session_count'] / user_agg['tenure_days'].replace(0, 1)
user_agg['demand_per_session'] = user_agg['total_demand'] / user_agg['session_count'].replace(0, 1)
user_agg['location_diversity'] = user_agg['num_locations_used'] / user_agg['session_count'].replace(0, 1)

drop_cols = ['UserID', 'last_session', 'first_session', 'days_since_last_session', 'churned']
X = user_agg.drop(columns=drop_cols)
y = user_agg['churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("=" * 70)
print("Tuning Logistic Regression")
print("=" * 70)

lr_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}
grid_lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), lr_grid, cv=skf, scoring='roc_auc', n_jobs=-1)
grid_lr.fit(X_train_scaled, y_train)
print(f"  Best params: {grid_lr.best_params_}")
print(f"  Best CV AUC: {grid_lr.best_score_:.4f}")

best_lr = grid_lr.best_estimator_
lr_preds = best_lr.predict(X_test_scaled)
lr_probs = best_lr.predict_proba(X_test_scaled)[:, 1]
print(f"  Test set -> Accuracy={accuracy_score(y_test, lr_preds):.4f}  AUC={roc_auc_score(y_test, lr_probs):.4f}")

print("\n" + "=" * 70)
print("FINAL HEAD-TO-HEAD (best tuned version of each model, engineered features)")
print("=" * 70)
print(f"{'Model':<25}{'CV AUC':<12}{'Test Acc':<12}{'Test AUC':<12}")
print(f"{'Logistic Regression':<25}{grid_lr.best_score_:<12.4f}{accuracy_score(y_test, lr_preds):<12.4f}{roc_auc_score(y_test, lr_probs):<12.4f}")
print(f"{'Random Forest (tuned)':<25}{0.7407:<12.4f}{0.6923:<12.4f}{0.7046:<12.4f}")
print(f"{'XGBoost (tuned)':<25}{0.7445:<12.4f}{0.6902:<12.4f}{0.7016:<12.4f}")

# --- Feature importance for Logistic Regression (coefficients) ---
print("\n" + "=" * 70)
print("Logistic Regression coefficients (standardized features, sorted by magnitude)")
print("=" * 70)
coef_df = pd.DataFrame({
    'feature': X.columns,
    'coefficient': best_lr.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)
print(coef_df.to_string(index=False))
