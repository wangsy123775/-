"""visualize.py -- 画图"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']

def plot_feature_importance(model, names, save_path="importance.png"):
    imp = model.feature_importances_
    idx = np.argsort(imp)
    plt.figure(figsize=(8, 4))
    plt.barh(range(len(imp)), imp[idx])
    plt.yticks(range(len(imp)), [names[i] for i in idx])
    plt.xlabel("特征重要性")
    plt.title("随机森林特征重要性排序")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_prediction(seq, proba, known_sites=None, save_path="prediction.png"):
    """在序列上画互作概率分布"""
    plt.figure(figsize=(14, 3))
    x = np.arange(1, len(seq) + 1)
    plt.plot(x, proba, color='#2196F3', linewidth=0.5)

    # 高置信度互作位点（概率>0.5）
    high = np.where(proba > 0.5)[0]
    if len(high) > 0:
        plt.scatter(high + 1, proba[high], c='red', s=3, label='预测互作位点')

    # 已知互作位点（如果有）
    if known_sites:
        for s, e, name in known_sites:
            mid = (s + e) / 2
            plt.axvspan(s, e, alpha=0.15, color='green')
            plt.text(mid, 0.8, name, ha='center', fontsize=6, rotation=90)

    plt.xlabel("残基位置")
    plt.ylabel("互作概率")
    plt.title("蛋白互作位点预测")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    # 训练模型后跑这个
    import joblib
    from dataset import build_training_set, INTERFACE_SITES

    model = joblib.load("model.pkl")
    X, y, seq = build_training_set()
    proba = model.predict_proba(X)[:, 1]

    names = ["疏水性", "体积", "电荷", "极性",
             "邻居疏水", "邻居体积", "邻居电荷", "邻居极性"]
    plot_feature_importance(model, names)
    plot_prediction(seq, proba, INTERFACE_SITES)
    print("图片已保存")
