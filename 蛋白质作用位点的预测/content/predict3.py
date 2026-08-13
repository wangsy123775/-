"""predict.py -- 预测新Cas9蛋白的互作位点"""
import sys, joblib, numpy as np
from features import seq_to_windowed_matrix

def predict_from_string(seq):
    """输入一条蛋白序列字符串 → 返回预测结果"""
    model = joblib.load("model.pkl")
    X = seq_to_windowed_matrix(seq)
    proba = model.predict_proba(X)[:, 1]        # 互作概率
    pred = model.predict(X)                      # 0/1预测

    sites = [(i+1, round(p, 3)) for i, (pr, p) in enumerate(zip(pred, proba)) if pr == 1]

    return sites, proba

def main():
    if len(sys.argv) != 2:
        print("用法: python predict.py <序列文件路径>")
        print("文件内容：单行氨基酸序列（纯文本，无FASTA头）")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        seq = f.read().strip().upper()

    sites, proba = predict_from_string(seq)

    print(f"序列长度: {len(seq)}")
    print(f"预测互作位点: {len(sites)} 个\n")

    print("位置\t概率")
    print("-" * 20)
    for pos, p in sites:
        aa = seq[pos - 1]
        bar = int(p * 20) * "#"
        print(f"{pos:>5}\t{p:.3f}  {bar}")

if __name__ == "__main__":
    main()
