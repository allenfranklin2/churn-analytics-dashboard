import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# --- Load user-level data (produced by preprocess_v2.py) ---
user_agg = pd.read_csv('user_level_data.csv')

# Re-encode categoricals (saved CSV has raw string values, not the encoded ones)
categorical_cols = ['common_charger_type', 'common_location', 'common_company']
for col in categorical_cols:
    le = LabelEncoder()
    user_agg[col] = le.fit_transform(user_agg[col])

drop_cols = ['UserID', 'last_session', 'first_session', 'days_since_last_session', 'churned']
X = user_agg.drop(columns=drop_cols)
y = user_agg['churned']

# --- Train/test split (stratified since churn is 56/44) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
}

results = {}
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
    results[name] = {'accuracy': acc, 'auc': auc}

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, preds, target_names=['Not Churned', 'Churned']))

print("\n=== Summary ===")
for name, metrics in results.items():
    print(f"{name}: Accuracy={metrics['accuracy']:.4f}  ROC-AUC={metrics['auc']:.4f}")
