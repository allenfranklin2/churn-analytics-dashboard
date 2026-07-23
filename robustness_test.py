import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# --- Load raw session-level data ---
df = pd.read_csv('ChargingRecords.csv')
df['StartDatetime'] = pd.to_datetime(df['StartDatetime'])
sub = df[df['UserID'] != 0].copy()
end_date = sub['StartDatetime'].max()

def build_dataset(churn_threshold_days):
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
    user_agg['churned'] = (user_agg['days_since_last_session'] > churn_threshold_days).astype(int)

    mode_charger = sub.groupby('UserID')['ChargerType'].agg(lambda x: x.mode()[0])
    mode_location = sub.groupby('UserID')['Location'].agg(lambda x: x.mode()[0])
    mode_company = sub.groupby('UserID')['ChargerCompany'].agg(lambda x: x.mode()[0])
    user_agg = user_agg.merge(mode_charger.rename('common_charger_type'), on='UserID')
    user_agg = user_agg.merge(mode_location.rename('common_location'), on='UserID')
    user_agg = user_agg.merge(mode_company.rename('common_company'), on='UserID')

    for col in ['common_charger_type', 'common_location', 'common_company']:
        le = LabelEncoder()
        user_agg[col] = le.fit_transform(user_agg[col])

    drop_cols = ['UserID', 'last_session', 'first_session', 'days_since_last_session', 'churned']
    X = user_agg.drop(columns=drop_cols)
    y = user_agg['churned']
    return X, y

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
}

# ============================================================
# TEST 1: 5-fold cross-validation at the 60-day threshold
# ============================================================
print("=" * 70)
print("TEST 1: 5-fold cross-validation (60-day churn threshold)")
print("=" * 70)

X, y = build_dataset(60)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = {name: {'accuracy': [], 'auc': []} for name in models}

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

        cv_results[name]['accuracy'].append(accuracy_score(y_test, preds))
        cv_results[name]['auc'].append(roc_auc_score(y_test, probs))

print(f"\n{'Model':<22}{'Mean Acc':<12}{'Std Acc':<12}{'Mean AUC':<12}{'Std AUC':<12}")
for name, res in cv_results.items():
    acc_arr = np.array(res['accuracy'])
    auc_arr = np.array(res['auc'])
    print(f"{name:<22}{acc_arr.mean():<12.4f}{acc_arr.std():<12.4f}{auc_arr.mean():<12.4f}{auc_arr.std():<12.4f}")

# ============================================================
# TEST 2: Churn threshold sensitivity (30 / 60 / 90 days)
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: Churn threshold sensitivity")
print("=" * 70)

for threshold in [30, 60, 90]:
    X, y = build_dataset(threshold)
    churn_rate = y.mean()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n--- Threshold: {threshold} days | Churn rate: {churn_rate*100:.1f}% ---")
    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        print(f"  {name:<22} Accuracy={acc:.4f}  AUC={auc:.4f}")

# ============================================================
# TEST 3: Random seed sensitivity (60-day threshold, Random Forest only)
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: Random seed sensitivity (Random Forest, 60-day threshold)")
print("=" * 70)

X, y = build_dataset(60)
seed_accs, seed_aucs = [], []
for seed in [1, 7, 21, 42, 99]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    probs = rf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    seed_accs.append(acc)
    seed_aucs.append(auc)
    print(f"  Seed {seed:<5} Accuracy={acc:.4f}  AUC={auc:.4f}")

print(f"\n  Mean Accuracy: {np.mean(seed_accs):.4f} (+/- {np.std(seed_accs):.4f})")
print(f"  Mean AUC:      {np.mean(seed_aucs):.4f} (+/- {np.std(seed_aucs):.4f})")
