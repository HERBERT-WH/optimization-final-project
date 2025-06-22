"""
基础优化问题定义
"""
from abc import ABC, abstractmethod
import numpy as np

class OptimizationProblem(ABC):
    """
    优化问题的抽象基类
    定义了所有优化问题必须实现的接口
    """
    
    def __init__(self):
        self.dim = None  # 问题维度
        
    @abstractmethod
    def objective(self, x):
        """
        计算目标函数值
        
        参数:
            x: 优化变量
            
        返回:
            目标函数值
        """
        pass
    
    @abstractmethod
    def subgradient(self, x):
        """
        计算次梯度
        
        参数:
            x: 优化变量
            
        返回:
            次梯度向量
        """
        pass
    
    def gradient(self, x):
        """
        计算梯度（如果存在）
        默认返回次梯度
        
        参数:
            x: 优化变量
            
        返回:
            梯度向量
        """
        return self.subgradient(x)
    
    def proximal_operator(self, x, t):
        """
        近端算子（如果可用）
        
        参数:
            x: 输入点
            t: 步长参数
            
        返回:
            近端算子的输出
        """
        raise NotImplementedError("该问题未实现近端算子")
    
    def projection(self, x):
        """
        投影到可行集（对于约束问题）
        
        参数:
            x: 输入点
            
        返回:
            投影后的点
        """
        return x  # 默认无约束
    
    def is_feasible(self, x):
        """
        检查点是否可行
        
        参数:
            x: 待检查的点
            
        返回:
            布尔值，表示是否可行
        """
        return True  # 默认无约束
    
    def distance_to_feasible_set(self, x):
        """
        计算到可行集的距离
        
        参数:
            x: 输入点
            
        返回:
            到可行集的距离
        """
        if self.is_feasible(x):
            return 0.0
        else:
            proj_x = self.projection(x)
            return np.linalg.norm(x - proj_x)
            
    def __str__(self):
        """
        返回问题的字符串表示
        """
        return f"{self.__class__.__name__}(维度={self.dim})" 