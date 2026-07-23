import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# --- Load and build base dataset (60-day threshold) ---
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

# --- NEW engineered features (behavioral ratios, not raw leakage) ---
# Sessions per day of tenure -- captures engagement intensity, not just raw counts
user_agg['sessions_per_tenure_day'] = user_agg['session_count'] / user_agg['tenure_days'].replace(0, 1)
# Demand per session -- captures typical usage intensity
user_agg['demand_per_session'] = user_agg['total_demand'] / user_agg['session_count'].replace(0, 1)
# Location diversity ratio -- how spread out a user's charging locations are relative to session count
user_agg['location_diversity'] = user_agg['num_locations_used'] / user_agg['session_count'].replace(0, 1)

drop_cols = ['UserID', 'last_session', 'first_session', 'days_since_last_session', 'churned']
X_base = user_agg.drop(columns=drop_cols + ['sessions_per_tenure_day', 'demand_per_session', 'location_diversity'])
X_engineered = user_agg.drop(columns=drop_cols)
y = user_agg['churned']

X_train_b, X_test_b, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=42, stratify=y)
X_train_e, X_test_e, _, _ = train_test_split(X_engineered, y, test_size=0.2, random_state=42, stratify=y)

print("=" * 70)
print("TEST A: Baseline vs Engineered Features (Random Forest, default params)")
print("=" * 70)

for label, (Xtr, Xte) in [('Baseline features', (X_train_b, X_test_b)),
                           ('+ Engineered ratios', (X_train_e, X_test_e))]:
    rf = RandomForestClassifier(random_state=42)
    rf.fit(Xtr, y_train)
    preds = rf.predict(Xte)
    probs = rf.predict_proba(Xte)[:, 1]
    print(f"  {label:<25} Accuracy={accuracy_score(y_test, preds):.4f}  AUC={roc_auc_score(y_test, probs):.4f}")

print("\n" + "=" * 70)
print("TEST B: Class-weight balancing (Random Forest, engineered features)")
print("=" * 70)

for weight in [None, 'balanced']:
    rf = RandomForestClassifier(random_state=42, class_weight=weight)
    rf.fit(X_train_e, y_train)
    preds = rf.predict(X_test_e)
    probs = rf.predict_proba(X_test_e)[:, 1]
    label = 'None (default)' if weight is None else 'balanced'
    print(f"  class_weight={label:<15} Accuracy={accuracy_score(y_test, preds):.4f}  AUC={roc_auc_score(y_test, probs):.4f}")

print("\n" + "=" * 70)
print("TEST C: Hyperparameter tuning (GridSearchCV, Random Forest, engineered features)")
print("=" * 70)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4],
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=skf, scoring='roc_auc', n_jobs=-1
)
grid.fit(X_train_e, y_train)

print(f"  Best params: {grid.best_params_}")
print(f"  Best CV AUC: {grid.best_score_:.4f}")

best_rf = grid.best_estimator_
preds = best_rf.predict(X_test_e)
probs = best_rf.predict_proba(X_test_e)[:, 1]
print(f"  Test set -> Accuracy={accuracy_score(y_test, preds):.4f}  AUC={roc_auc_score(y_test, probs):.4f}")

print("\n" + "=" * 70)
print("TEST D: Hyperparameter tuning (GridSearchCV, XGBoost, engineered features)")
print("=" * 70)

xgb_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
}
grid_xgb = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss'),
    xgb_grid, cv=skf, scoring='roc_auc', n_jobs=-1
)
grid_xgb.fit(X_train_e, y_train)

print(f"  Best params: {grid_xgb.best_params_}")
print(f"  Best CV AUC: {grid_xgb.best_score_:.4f}")

best_xgb = grid_xgb.best_estimator_
preds = best_xgb.predict(X_test_e)
probs = best_xgb.predict_proba(X_test_e)[:, 1]
print(f"  Test set -> Accuracy={accuracy_score(y_test, preds):.4f}  AUC={roc_auc_score(y_test, probs):.4f}")

print("\n" + "=" * 70)
print("SUMMARY: Original vs Best Improved Model")
print("=" * 70)
print(f"  Original Random Forest (default, base features): Accuracy=0.6410  AUC=0.7027")
print(f"  Tuned Random Forest (engineered features):        Accuracy={accuracy_score(y_test, best_rf.predict(X_test_e)):.4f}  AUC={roc_auc_score(y_test, best_rf.predict_proba(X_test_e)[:,1]):.4f}")
print(f"  Tuned XGBoost (engineered features):              Accuracy={accuracy_score(y_test, preds):.4f}  AUC={roc_auc_score(y_test, probs):.4f}")
