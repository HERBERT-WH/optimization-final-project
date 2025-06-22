"""
正则化优化问题
"""
import numpy as np
from .base import OptimizationProblem

class L1RegularizedProblem(OptimizationProblem):
    """
    L1正则化问题: min 1/2 ||Ax - b||^2 + λ||x||_1
    """
    
    def __init__(self, A, b, lam=1.0):
        super().__init__()
        self.A = A
        self.b = b
        self.lam = lam
        self.dim = A.shape[1]
        
    def objective(self, x):
        """计算目标函数值"""
        residual = self.A @ x - self.b
        return 0.5 * np.sum(residual**2) + self.lam * np.sum(np.abs(x))
    
    def subgradient(self, x):
        """计算次梯度"""
        # 二次项的梯度
        grad_quad = self.A.T @ (self.A @ x - self.b)
        
        # L1项的次梯度
        subgrad_l1 = np.zeros_like(x)
        subgrad_l1[x > 0] = self.lam
        subgrad_l1[x < 0] = -self.lam
        subgrad_l1[x == 0] = self.lam * np.random.uniform(-1, 1, np.sum(x == 0))
        
        return grad_quad + subgrad_l1
    
    def gradient_smooth(self, x):
        """返回光滑部分的梯度"""
        return self.A.T @ (self.A @ x - self.b)
    
    def proximal_operator(self, x, t):
        """L1范数的近端算子（软阈值算子）"""
        return np.sign(x) * np.maximum(np.abs(x) - t * self.lam, 0)

class L2RegularizedProblem(OptimizationProblem):
    """
    L2正则化问题: min 1/2 ||Ax - b||^2 + λ/2 ||x||^2
    """
    
    def __init__(self, A, b, lam=1.0):
        super().__init__()
        self.A = A
        self.b = b
        self.lam = lam
        self.dim = A.shape[1]
        
    def objective(self, x):
        """计算目标函数值"""
        residual = self.A @ x - self.b
        return 0.5 * np.sum(residual**2) + 0.5 * self.lam * np.sum(x**2)
    
    def subgradient(self, x):
        """计算梯度（L2正则化是光滑的）"""
        return self.A.T @ (self.A @ x - self.b) + self.lam * x
    
    def gradient(self, x):
        """计算梯度"""
        return self.subgradient(x)
    
    def proximal_operator(self, x, t):
        """L2范数的近端算子"""
        return x / (1 + t * self.lam)

class ElasticNetProblem(OptimizationProblem):
    """
    弹性网络问题: min 1/2 ||Ax - b||^2 + λ1||x||_1 + λ2/2 ||x||^2
    """
    
    def __init__(self, A, b, lam1=1.0, lam2=1.0):
        super().__init__()
        self.A = A
        self.b = b
        self.lam1 = lam1  # L1正则化参数
        self.lam2 = lam2  # L2正则化参数
        self.dim = A.shape[1]
        
    def objective(self, x):
        """计算目标函数值"""
        residual = self.A @ x - self.b
        return (0.5 * np.sum(residual**2) + 
                self.lam1 * np.sum(np.abs(x)) + 
                0.5 * self.lam2 * np.sum(x**2))
    
    def subgradient(self, x):
        """计算次梯度"""
        # 二次项和L2项的梯度
        grad_smooth = self.A.T @ (self.A @ x - self.b) + self.lam2 * x
        
        # L1项的次梯度
        subgrad_l1 = np.zeros_like(x)
        subgrad_l1[x > 0] = self.lam1
        subgrad_l1[x < 0] = -self.lam1
        subgrad_l1[x == 0] = self.lam1 * np.random.uniform(-1, 1, np.sum(x == 0))
        
        return grad_smooth + subgrad_l1
    
    def gradient_smooth(self, x):
        """返回光滑部分的梯度"""
        return self.A.T @ (self.A @ x - self.b) + self.lam2 * x
    
    def proximal_operator(self, x, t):
        """弹性网络的近端算子"""
        # 先处理L2项，再处理L1项
        y = x / (1 + t * self.lam2)
        return np.sign(y) * np.maximum(np.abs(y) - t * self.lam1 / (1 + t * self.lam2), 0)

class L1RegularizedLogisticRegression(OptimizationProblem):
    """
    L1正则化逻辑回归: min Σlog(1 + exp(-yi * xi^T β)) + λ||β||_1
    """
    
    def __init__(self, X, y, lam=1.0):
        super().__init__()
        self.X = X  # 特征矩阵
        self.y = y  # 标签 (+1 或 -1)
        self.lam = lam
        self.dim = X.shape[1]
        
    def objective(self, beta):
        """计算目标函数值"""
        z = self.y * (self.X @ beta)
        logistic_loss = np.sum(np.log(1 + np.exp(-z)))
        regularization = self.lam * np.sum(np.abs(beta))
        return logistic_loss + regularization
    
    def subgradient(self, beta):
        """计算次梯度"""
        z = self.y * (self.X @ beta)
        prob = 1 / (1 + np.exp(z))  # P(yi = -1 | xi, beta)
        
        # 逻辑损失的梯度
        grad_logistic = -self.X.T @ (self.y * prob)
        
        # L1项的次梯度
        subgrad_l1 = np.zeros_like(beta)
        subgrad_l1[beta > 0] = self.lam
        subgrad_l1[beta < 0] = -self.lam
        subgrad_l1[beta == 0] = self.lam * np.random.uniform(-1, 1, np.sum(beta == 0))
        
        return grad_logistic + subgrad_l1
    
    def gradient_smooth(self, beta):
        """返回光滑部分的梯度"""
        z = self.y * (self.X @ beta)
        prob = 1 / (1 + np.exp(z))
        return -self.X.T @ (self.y * prob)
    
    def proximal_operator(self, beta, t):
        """L1范数的近端算子"""
        return np.sign(beta) * np.maximum(np.abs(beta) - t * self.lam, 0) 