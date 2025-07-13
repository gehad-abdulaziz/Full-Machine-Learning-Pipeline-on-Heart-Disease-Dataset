import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_csv('./Data/processed_heart.csv') 

X = df.drop('target', axis=1)
y = df['target']

print("Features columns:")
print(X.columns.tolist()) 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(
        max_depth=20,
        min_samples_split=5,
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    ))
])

smote = SMOTE(random_state=42)

X_train_scaled = pipeline.named_steps['scaler'].fit_transform(X_train)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

pipeline.named_steps['model'].fit(X_train_resampled, y_train_resampled)

X_test_scaled = pipeline.named_steps['scaler'].transform(X_test)
y_pred = pipeline.named_steps['model'].predict(X_test_scaled)

print("✅ Final Model Evaluation on Test Set:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
print(f"Precision (macro): {precision_score(y_test, y_pred, average='macro'):.4f}")
print(f"Recall (macro): {recall_score(y_test, y_pred, average='macro'):.4f}")
print(f"F1 Score (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")


pipeline.fit(X_train, y_train)

joblib.dump(pipeline, './models/best_model_pipeline.pkl')
print("✅ Pipeline model saved successfully!")

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision (macro): {precision_score(y_test, y_pred, average='macro'):.4f}")
print(f"Recall (macro): {recall_score(y_test, y_pred, average='macro'):.4f}")
print(f"F1 Score (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
