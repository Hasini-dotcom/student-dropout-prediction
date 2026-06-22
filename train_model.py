# train_model.py
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
df = pd.read_csv('data/student_dropout_dataset.csv')
print(f"✅ Loaded {len(df)} rows")

# ─────────────────────────────────────────
# 2. CLEAN & ENCODE
# ─────────────────────────────────────────
# Drop columns the model doesn't need
df = df.drop(columns=['student_id', 'dropout_risk_score', 'dropout_risk_label'])

# Convert text columns to numbers
label_encoders = {}
text_columns = ['gender', 'state', 'family_income_category', 'parents_education']

for col in text_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # save for later use in dashboard

# ─────────────────────────────────────────
# 3. SPLIT INTO FEATURES & TARGET
# ─────────────────────────────────────────
X = df.drop(columns=['dropout'])
y = df['dropout']

print(f"Features: {list(X.columns)}")
print(f"Dropout cases: {y.sum()} / {len(y)}")

# 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────
# 4. TRAIN XGBOOST MODEL
# ─────────────────────────────────────────
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=5,   # handles class imbalance (fewer dropouts)
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)
print("✅ Model trained!")

# ─────────────────────────────────────────
# 5. EVALUATE
# ─────────────────────────────────────────
y_pred = model.predict(X_test)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Dropout', 'Dropout']))

# Confusion matrix chart
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Dropout', 'Dropout'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('model/confusion_matrix.png')
plt.show()
print("✅ Confusion matrix saved to model/")

# ─────────────────────────────────────────
# 6. SHAP EXPLANATIONS
# ─────────────────────────────────────────
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global feature importance chart
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig('model/shap_summary.png')
plt.close()
print("✅ SHAP summary chart saved to model/")

# ─────────────────────────────────────────
# 7. SAVE MODEL & ENCODERS
# ─────────────────────────────────────────
joblib.dump(model, 'model/dropout_model.pkl')
joblib.dump(label_encoders, 'model/label_encoders.pkl')
joblib.dump(list(X.columns), 'model/feature_names.pkl')
joblib.dump(explainer, 'model/shap_explainer.pkl')

print("\n✅ All files saved in model/ folder:")
print("   - dropout_model.pkl")
print("   - label_encoders.pkl")
print("   - feature_names.pkl")
print("   - shap_explainer.pkl")
print("   - confusion_matrix.png")
print("   - shap_summary.png")
print("\n🎉 Training complete!")