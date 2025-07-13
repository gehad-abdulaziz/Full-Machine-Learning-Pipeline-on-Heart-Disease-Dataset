# superpy.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# ============================================================
# 📌 Function: load_data
# ============================================================
def load_data(path):
    df = pd.read_csv(path)
    X = df.drop('target', axis=1)
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ============================================================
# 📌 Function: tune_logistic_regression
# ============================================================
def tune_logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr', class_weight='balanced')

    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['lbfgs']
    }

    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    return evaluate_model(best_model, X_test, y_test, y_train)

# ============================================================
# 📌 Function: tune_decision_tree
# ============================================================
def tune_decision_tree(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier(random_state=42, class_weight='balanced')

    param_grid = {
        'max_depth': [None, 3, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    return evaluate_model(best_model, X_test, y_test, y_train)

# ============================================================
# 📌 Function: tune_random_forest
# ============================================================
def tune_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(random_state=42, class_weight='balanced')

    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    random_search = RandomizedSearchCV(model, param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42)
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    return evaluate_model(best_model, X_test, y_test, y_train)

# ============================================================
# 📌 Function: tune_svm
# ============================================================
def tune_svm(X_train, y_train, X_test, y_test):
    model = SVC(probability=True, class_weight='balanced', random_state=42)

    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }

    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    return evaluate_model(best_model, X_test, y_test, y_train)

# ============================================================
# 📌 Function: evaluate_model
# ============================================================
def evaluate_model(model, X_test, y_test, y_train):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    n_classes = len(np.unique(y_train))
    y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        auc_score = roc_auc_score(y_test_bin, y_proba, multi_class='ovr')
    else:
        auc_score = None

    # Print results
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    if auc_score is not None:
        print(f"AUC: {auc_score:.4f}")

    return model, acc, prec, rec, f1, auc_score


if __name__ == "__main__":
    # تحميل البيانات
    X_train, X_test, y_train, y_test = load_data('./Data/selected_features_scaled.csv')

    print("\n🔶 Logistic Regression Results")
    tune_logistic_regression(X_train, y_train, X_test, y_test)

    print("\n🔶 Decision Tree Results")
    tune_decision_tree(X_train, y_train, X_test, y_test)

    print("\n🔶 Random Forest Results")
    tune_random_forest(X_train, y_train, X_test, y_test)

    print("\n🔶 SVM Results")
    tune_svm(X_train, y_train, X_test, y_test)