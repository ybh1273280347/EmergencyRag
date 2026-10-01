import numpy as np


def normalize_vectors(vectors: np.ndarray, dimension: int | None = None) -> np.ndarray:
    """FAISS 的余弦检索要求文档与查询均为单位 float32 向量。"""
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or not array.shape[1]:
        raise ValueError("向量必须为二维矩阵且维度非零")
    if dimension is not None and array.shape[1] != dimension:
        raise ValueError(f"向量维度不匹配：期望 {dimension}，实际 {array.shape[1]}")
    if not np.isfinite(array).all():
        raise ValueError("向量包含非有限数值")
    norms = np.linalg.norm(array.astype(np.float64), axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("零向量不能用于余弦检索")
    return np.ascontiguousarray(array / norms, dtype=np.float32)
