import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('ChargingRecords.csv')
df['StartDatetime'] = pd.to_datetime(df['StartDatetime'])
df['hour_of_day'] = df['StartDatetime'].dt.hour
df['day_of_week'] = df['StartDatetime'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# --- Target: extreme high vs extreme low electricity demand (tertile split) ---
q33 = df['Demand'].quantile(0.33)
q67 = df['Demand'].quantile(0.67)
df_model = df[(df['Demand'] <= q33) | (df['Demand'] >= q67)].copy()
df_model['high_demand'] = (df_model['Demand'] >= q67).astype(int)

le_charger = LabelEncoder()
le_location = LabelEncoder()
le_company = LabelEncoder()
df_model['charger_type_enc'] = le_charger.fit_transform(df_model['ChargerType'].astype(str))
df_model['location_enc'] = le_location.fit_transform(df_model['Location'].astype(str))
df_model['company_enc'] = le_company.fit_transform(df_model['ChargerCompany'].astype(str))
df_model['duration_per_hour_bucket'] = df_model['Duration'] / (df_model['hour_of_day'] + 1)
df_model['duration_x_charger'] = df_model['Duration'] * (df_model['charger_type_enc'] + 1)

features = ['Duration', 'charger_type_enc', 'location_enc', 'company_enc',
            'hour_of_day', 'day_of_week', 'is_weekend', 'duration_per_hour_bucket', 'duration_x_charger']
X = df_model[features]
y = df_model['high_demand']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Baseline: Logistic Regression ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_preds = lr.predict(X_test_scaled)
lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
print(f"Logistic Regression -- Accuracy: {accuracy_score(y_test, lr_preds):.4f}  AUC: {roc_auc_score(y_test, lr_probs):.4f}")

# --- Primary model: XGBoost (tuned params from earlier testing) ---
xgb_model = XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=300,
                            max_depth=6, learning_rate=0.05, n_jobs=-1)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, xgb_preds)
auc = roc_auc_score(y_test, xgb_probs)
print(f"XGBoost (primary model) -- Accuracy: {acc:.4f}  AUC: {auc:.4f}")
print()
print(classification_report(y_test, xgb_preds, target_names=['Low Demand', 'High Demand']))

# --- 5-fold cross-validation for robustness ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_accs, cv_aucs = [], []
for train_idx, test_idx in skf.split(X, y):
    Xt, Xv = X.iloc[train_idx], X.iloc[test_idx]
    yt, yv = y.iloc[train_idx], y.iloc[test_idx]
    m = XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=300, max_depth=6, learning_rate=0.05, n_jobs=-1)
    m.fit(Xt, yt)
    p = m.predict(Xv)
    pr = m.predict_proba(Xv)[:, 1]
    cv_accs.append(accuracy_score(yv, p))
    cv_aucs.append(roc_auc_score(yv, pr))
print(f"\n5-Fold CV -- Accuracy: {np.mean(cv_accs):.4f} (+/- {np.std(cv_accs):.4f})  AUC: {np.mean(cv_aucs):.4f} (+/- {np.std(cv_aucs):.4f})")

# --- SHAP explainability ---
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

mean_abs_shap = pd.DataFrame({
    'feature': X_test.columns,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)
print("\nFeature importance ranked by mean |SHAP value|:")
print(mean_abs_shap.to_string(index=False))

plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig('shap_demand_beeswarm.png', dpi=150)
plt.close()

plt.figure()
shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig('shap_demand_bar.png', dpi=150)
plt.close()

# --- Save everything for the dashboard ---
with open('demand_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
with open('demand_explainer.pkl', 'wb') as f:
    pickle.dump(explainer, f)
with open('demand_shap_values.pkl', 'wb') as f:
    pickle.dump(shap_values, f)
X_train.to_pickle('demand_X_train.pkl')
X_test.to_pickle('demand_X_test.pkl')
y_train.to_pickle('demand_y_train.pkl')
y_test.to_pickle('demand_y_test.pkl')
df_model.to_pickle('demand_df_cleaned.pkl')

print("\nSaved: demand_model.pkl, demand_explainer.pkl, demand_shap_values.pkl,")
print("       demand_X_train.pkl, demand_X_test.pkl, demand_y_train.pkl, demand_y_test.pkl, demand_df_cleaned.pkl")
print("Saved plots: shap_demand_beeswarm.png, shap_demand_bar.png")
