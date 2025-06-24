import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# 1. Load data
df = pd.read_csv('train.csv')

# 2. Preprocessing: One-hot encode categoricals
df = pd.get_dummies(df, columns=['sourcing_channel', 'residence_area_type'])

# Impute missing values
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col].fillna(df[col].median(skipna=True), inplace=True)
    else:
        df[col].fillna(df[col].mode(dropna=True).iloc[0], inplace=True)

# 3. Split features/target
X = df.drop(['target', 'id'], axis=1)
y = df['target']

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 5. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
resampled = smote.fit_resample(X_train_scaled, y_train)
X_train_bal, y_train_bal = resampled  # type: ignore

# 7. Train RandomForest with class_weight
clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
clf.fit(X_train_bal, y_train_bal)

# 8. Evaluate
preds = clf.predict(X_test_scaled)
probs = clf.predict_proba(X_test_scaled)[:, 1]  # type: ignore
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)
roc = roc_auc_score(y_test, probs)
print('Accuracy:', acc)
print('F1 Score:', f1)
print('ROC-AUC:', roc)
print(classification_report(y_test, preds))

# 9. Save model and scaler
joblib.dump(clf, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print('Model and scaler saved.') 