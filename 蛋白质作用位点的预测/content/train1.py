"""train.py -- 训练随机森林模型"""
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
from dataset import build_training_set
from features import seq_to_windowed_matrix

def train():
    # 1. 拿数据
    X, y, seq = build_training_set()

    # 2. 切分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 3. 训练
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight='balanced',  # 81正 vs 1287负 严重不平衡
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 4. 评估
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cv = cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()

    print(f"准确率: {acc:.3f}")
    print(f"AUC: {auc:.3f}")
    print(f"5折交叉验证 AUC: {cv:.3f}")
    print(f"特征重要性: {model.feature_importances_}")

    # 5. 保存
    joblib.dump(model, "model.pkl")
    print("模型已保存到 model.pkl")

    return model, (acc, auc)

if __name__ == "__main__":
    train()
