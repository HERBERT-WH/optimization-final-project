"""
投影算子实现

包含常用的投影算子，用于约束优化问题
"""

import numpy as np
from typing import Union, Tuple

def project_box(x: np.ndarray, lower: Union[float, np.ndarray] = None, 
                upper: Union[float, np.ndarray] = None) -> np.ndarray:
    """
    投影到箱约束集 [lower, upper]
    
    Args:
        x: 输入向量
        lower: 下界（标量或向量）
        upper: 上界（标量或向量）
        
    Returns:
        投影后的向量
    """
    result = x.copy()
    
    if lower is not None:
        result = np.maximum(result, lower)
    if upper is not None:
        result = np.minimum(result, upper)
        
    return result

def project_simplex(x: np.ndarray) -> np.ndarray:
    """
    投影到概率单纯形: {x: x >= 0, sum(x) = 1}
    
    使用O(n log n)算法
    
    Args:
        x: 输入向量
        
    Returns:
        投影后的向量
    """
    n = len(x)
    
    # 排序（降序）
    x_sorted = np.sort(x)[::-1]
    
    # 计算阈值
    cumsum = np.cumsum(x_sorted)
    idx = np.arange(1, n + 1)
    cond = (x_sorted - (cumsum - 1) / idx) > 0
    
    if np.any(cond):
        k = np.where(cond)[0][-1]
        theta = (cumsum[k] - 1) / (k + 1)
    else:
        theta = (cumsum[-1] - 1) / n
    
    return np.maximum(x - theta, 0)

def project_l2_ball(x: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """
    投影到L2球: {x: ||x||_2 <= radius}
    
    Args:
        x: 输入向量
        radius: 球的半径
        
    Returns:
        投影后的向量
    """
    norm_x = np.linalg.norm(x)
    if norm_x <= radius:
        return x
    else:
        return radius * x / norm_x

def project_l1_ball(x: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """
    投影到L1球: {x: ||x||_1 <= radius}
    
    Args:
        x: 输入向量
        radius: 球的半径
        
    Returns:
        投影后的向量
    """
    if np.sum(np.abs(x)) <= radius:
        return x
    
    # 使用对偶排序方法
    abs_x = np.abs(x)
    signs = np.sign(x)
    
    # 对绝对值排序
    sorted_abs = np.sort(abs_x)[::-1]
    
    # 找到阈值
    cumsum = np.cumsum(sorted_abs)
    idx = np.arange(1, len(x) + 1)
    cond = (sorted_abs - (cumsum - radius) / idx) > 0
    
    if np.any(cond):
        k = np.where(cond)[0][-1]
        theta = (cumsum[k] - radius) / (k + 1)
    else:
        theta = (cumsum[-1] - radius) / len(x)
    
    return signs * np.maximum(abs_x - theta, 0)

def project_linf_ball(x: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """
    投影到L∞球 {x: ||x||_∞ <= radius}
    
    Args:
        x: 输入向量
        radius: 球的半径
        
    Returns:
        投影后的向量
    """
    return np.clip(x, -radius, radius)

def project_positive_orthant(x: np.ndarray) -> np.ndarray:
    """
    投影到正象限: {x: x >= 0}
    
    Args:
        x: 输入向量
        
    Returns:
        投影后的向量
    """
    return np.maximum(x, 0)

def project_affine_set(x: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    投影到仿射集 {x: Ax = b}
    
    Args:
        x: 输入向量
        A: 约束矩阵
        b: 约束向量
        
    Returns:
        投影后的向量
    """
    # 投影公式: x - A^T(AA^T)^{-1}(Ax - b)
    try:
        AAT_inv = np.linalg.inv(A @ A.T)
        return x - A.T @ AAT_inv @ (A @ x - b)
    except np.linalg.LinAlgError:
        # 如果矩阵奇异，使用伪逆
        AAT_pinv = np.linalg.pinv(A @ A.T)
        return x - A.T @ AAT_pinv @ (A @ x - b)

def project_hyperplane(x: np.ndarray, a: np.ndarray, b: float) -> np.ndarray:
    """
    投影到超平面: {x: a^T x = b}
    
    Args:
        x: 输入向量
        a: 法向量
        b: 常数项
        
    Returns:
        投影后的向量
    """
    a_norm_sq = np.dot(a, a)
    if a_norm_sq == 0:
        return x
    
    return x - ((np.dot(a, x) - b) / a_norm_sq) * a

def project_halfspace(x: np.ndarray, a: np.ndarray, b: float) -> np.ndarray:
    """
    投影到半空间: {x: a^T x <= b}
    
    Args:
        x: 输入向量
        a: 法向量
        b: 常数项
        
    Returns:
        投影后的向量
    """
    if np.dot(a, x) <= b:
        return x
    else:
        return project_hyperplane(x, a, b)

def project_psd_cone(X: np.ndarray) -> np.ndarray:
    """
    投影到半正定锥
    
    Args:
        X: 输入矩阵
        
    Returns:
        投影后的矩阵
    """
    # 特征值分解
    eigenvals, eigenvecs = np.linalg.eigh(X)
    
    # 保留非负特征值
    eigenvals_proj = np.maximum(eigenvals, 0)
    
    # 重构矩阵
    return eigenvecs @ np.diag(eigenvals_proj) @ eigenvecs.T 