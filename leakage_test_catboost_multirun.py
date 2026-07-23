"""
LEAKAGE TEST (multi-run) — CatBoost churn model
=================================================
Same purpose as leakage_test_catboost.py, but runs BOTH the honest model
(days_since_last_session excluded, as in your real pipeline) and the leaky
model (feature included) across several random train/test splits, so you
can report mean +/- std on the slide -- consistent with the multi-seed
stability testing already done elsewhere in this project.

Does NOT overwrite any real project files -- only prints results.

Run from the same folder as ChargingRecords.csv:
    python3 leakage_test_catboost_multirun.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

SEEDS = [42, 7, 123, 2024, 99]  # 5 runs, same spirit as your existing CV/stability checks

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

cat_features = ['common_charger_type', 'common_location', 'common_company']
for col in cat_features:
    user_agg[col] = user_agg[col].astype(str)

y = user_agg['churned']

# Honest feature set (matches your real, reported model)
honest_drop_cols = ['UserID', 'last_session', 'first_session', 'days_since_last_session', 'churned']
X_honest = user_agg.drop(columns=honest_drop_cols)
cat_idx_honest = [X_honest.columns.get_loc(c) for c in cat_features]

# Leaky feature set (days_since_last_session left in)
leaky_drop_cols = ['UserID', 'last_session', 'first_session', 'churned']
X_leaky = user_agg.drop(columns=leaky_drop_cols)
cat_idx_leaky = [X_leaky.columns.get_loc(c) for c in cat_features]


def run_once(X, cat_idx, seed):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    model = CatBoostClassifier(depth=6, iterations=200, learning_rate=0.01,
                                random_state=seed, verbose=0)
    model.fit(X_train, y_train, cat_features=cat_idx)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    return accuracy_score(y_test, preds), roc_auc_score(y_test, probs)


honest_accs, honest_aucs = [], []
leaky_accs, leaky_aucs = [], []

print(f"Running {len(SEEDS)} seeds for HONEST and LEAKY models...\n")

for seed in SEEDS:
    h_acc, h_auc = run_once(X_honest, cat_idx_honest, seed)
    l_acc, l_auc = run_once(X_leaky, cat_idx_leaky, seed)
    honest_accs.append(h_acc); honest_aucs.append(h_auc)
    leaky_accs.append(l_acc); leaky_aucs.append(l_auc)
    print(f"  seed={seed:5d}  honest: acc={h_acc:.4f} auc={h_auc:.4f}   "
          f"leaky: acc={l_acc:.4f} auc={l_auc:.4f}")

honest_accs, honest_aucs = np.array(honest_accs), np.array(honest_aucs)
leaky_accs, leaky_aucs = np.array(leaky_accs), np.array(leaky_aucs)

print("\n=== SUMMARY ACROSS {} SEEDS (copy into slide) ===".format(len(SEEDS)))
print(f"{'':20s}{'Accuracy':>18s}{'AUC':>16s}")
print(f"{'Honest (excluded)':20s}"
      f"{honest_accs.mean()*100:>10.2f}% +/- {honest_accs.std()*100:.2f}pp"
      f"{honest_aucs.mean():>10.3f} +/- {honest_aucs.std():.3f}")
print(f"{'Leaky (included)':20s}"
      f"{leaky_accs.mean()*100:>10.2f}% +/- {leaky_accs.std()*100:.2f}pp"
      f"{leaky_aucs.mean():>10.3f} +/- {leaky_aucs.std():.3f}")
print(f"{'Jump (mean)':20s}"
      f"{(leaky_accs.mean()-honest_accs.mean())*100:>+10.2f}pp"
      f"{'':>8s}{(leaky_aucs.mean()-honest_aucs.mean()):>+10.3f}")

# NOTE: this script never saves catboost_model.pkl / X_train.pkl / etc. --
# it only prints results, so your real pipeline's files are never touched.
