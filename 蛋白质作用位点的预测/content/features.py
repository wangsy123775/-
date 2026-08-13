"""features.py -- 氨基酸特征工程"""
import numpy as np

FEAT = {
    # [疏水值, 体积(Å³), 电荷, 极性]
    'A': [1.8,  88.6,  0, 0],
    'C': [2.5,  108.5, 0, 0],
    'D': [-3.5, 111.1, -1, 1],
    'E': [-3.5, 138.4, -1, 1],
    'F': [2.8,  189.9, 0, 0],
    'G': [-0.4, 60.1,  0, 0],
    'H': [-3.2, 153.2, 0, 1],
    'I': [4.5,  166.7, 0, 0],
    'K': [-3.9, 168.6, 1, 1],
    'L': [3.8,  166.7, 0, 0],
    'M': [1.9,  162.9, 0, 0],
    'N': [-3.5, 114.1, 0, 1],
    'P': [-1.6, 112.7, 0, 0],
    'Q': [-3.5, 143.8, 0, 1],
    'R': [-4.5, 173.4, 1, 1],
    'S': [-0.8, 89.0,  0, 1],
    'T': [-0.7, 116.1, 0, 1],
    'V': [4.2,  140.0, 0, 0],
    'W': [-0.9, 227.8, 0, 0],
    'Y': [-1.3, 193.6, 0, 0],
}

def seq_to_matrix(seq: str):
    """一条氨基酸序列 → N*4 特征矩阵"""
    return np.array([FEAT[aa] for aa in seq])

def seq_to_windowed_matrix(seq: str, w: int = 3):
    """加滑动窗口:每个残基取前后±w个邻居的特征均值,拼成 12维"""
    raw = seq_to_matrix(seq)
    n = len(seq)
    out = np.zeros((n, raw.shape[1] * 2))
    for i in range(n):
        lo = max(0, i - w)
        hi = min(n, i + w + 1)
        local_avg = raw[lo:hi].mean(axis=0)
        out[i] = np.hstack([raw[i], local_avg])
    return out
