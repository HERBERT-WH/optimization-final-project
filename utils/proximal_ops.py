"""
各种近端算子的实现
"""
import numpy as np
from scipy.optimize import minimize_scalar

def prox_l1(x, t):
    """
    L1范数的近端算子（软阈值算子）
    prox_{t||·||_1}(x) = sign(x) * max(|x| - t, 0)
    
    参数:
        x: 输入向量
        t: 步长参数
        
    返回:
        近端算子的输出
    """
    return np.sign(x) * np.maximum(np.abs(x) - t, 0)

def prox_l2(x, t):
    """
    L2范数的近端算子
    prox_{t||·||_2}(x) = max(1 - t/||x||_2, 0) * x
    
    参数:
        x: 输入向量
        t: 步长参数
        
    返回:
        近端算子的输出
    """
    norm_x = np.linalg.norm(x)
    if norm_x == 0:
        return x
    return np.maximum(1 - t / norm_x, 0) * x

def prox_l2_squared(x, t):
    """
    L2范数平方的近端算子
    prox_{t/2||·||_2^2}(x) = x / (1 + t)
    
    参数:
        x: 输入向量
        t: 步长参数
        
    返回:
        近端算子的输出
    """
    return x / (1 + t)

def prox_elastic_net(x, t, alpha, l1_ratio):
    """
    弹性网络正则化的近端算子
    
    参数:
        x: 输入向量
        t: 步长参数
        alpha: 正则化强度
        l1_ratio: L1正则化的比例
        
    返回:
        近端算子的输出
    """
    l1_reg = alpha * l1_ratio * t
    l2_reg = alpha * (1 - l1_ratio) * t
    
    # 先应用L2项，再应用L1项
    x_l2 = x / (1 + l2_reg)
    return prox_l1(x_l2, l1_reg / (1 + l2_reg))

def prox_group_lasso(x, t, groups):
    """
    组Lasso的近端算子
    
    参数:
        x: 输入向量
        t: 步长参数
        groups: 组的索引列表
        
    返回:
        近端算子的输出
    """
    result = x.copy()
    
    for group in groups:
        x_group = x[group]
        norm_group = np.linalg.norm(x_group)
        
        if norm_group > t:
            result[group] = (1 - t / norm_group) * x_group
        else:
            result[group] = 0
            
    return result

def prox_nuclear_norm(X, t):
    """
    核范数的近端算子（软阈值SVD）
    
    参数:
        X: 输入矩阵
        t: 步长参数
        
    返回:
        近端算子的输出
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_soft = np.maximum(s - t, 0)
    return U @ np.diag(s_soft) @ Vt

def prox_log_sum_exp(x, t):
    """
    log-sum-exp函数的近端算子
    
    参数:
        x: 输入向量
        t: 步长参数
        
    返回:
        近端算子的输出
    """
    # 使用数值稳定的计算方法
    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    sum_exp = np.sum(exp_x)
    
    # 近端算子需要求解非线性方程
    # 这里使用简化的近似
    return x - t * exp_x / sum_exp

def prox_indicator_box(x, t, lower=None, upper=None):
    """
    箱约束的示性函数的近端算子（投影算子）
    
    参数:
        x: 输入向量
        t: 步长参数（对于示性函数不起作用）
        lower: 下界
        upper: 上界
        
    返回:
        投影后的向量
    """
    from .projections import project_box
    return project_box(x, lower, upper)

def prox_indicator_simplex(x, t):
    """
    单纯形约束的示性函数的近端算子
    
    参数:
        x: 输入向量
        t: 步长参数（对于示性函数不起作用）
        
    返回:
        投影后的向量
    """
    from .projections import project_simplex
    return project_simplex(x)

def prox_huber(x, t, delta=1.0):
    """
    Huber损失的近端算子
    
    参数:
        x: 输入向量
        t: 步长参数
        delta: Huber参数
        
    返回:
        近端算子的输出
    """
    result = np.zeros_like(x)
    
    # 对每个元素分别处理
    for i in range(len(x)):
        if np.abs(x[i]) <= delta + t:
            result[i] = x[i] / (1 + t / delta)
        else:
            result[i] = x[i] - t * np.sign(x[i])
            
    return result

def prox_quantile(x, t, tau=0.5):
    """
    分位数损失的近端算子
    
    参数:
        x: 输入向量
        t: 步长参数
        tau: 分位数参数
        
    返回:
        近端算子的输出
    """
    result = np.zeros_like(x)
    
    for i in range(len(x)):
        if x[i] > t * (1 - tau):
            result[i] = x[i] - t * (1 - tau)
        elif x[i] < -t * tau:
            result[i] = x[i] + t * tau
        else:
            result[i] = 0
            
    return result 