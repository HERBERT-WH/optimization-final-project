"""
分裂方法 (Splitting Methods) 实现

基于用户提供的算法描述:
- 将复杂问题分解为可并行处理的简单子问题
- 包含Douglas-Rachford、ADMM等经典分裂算法
"""

import numpy as np
import time
from typing import Tuple, List, Optional, Callable
from ..problems.base import OptimizationProblem

class DouglasRachford:
    """
    Douglas-Rachford分裂算法
    
    用于求解形式为 min_x f(x) + g(x) 的问题
    其中f和g都是凸函数，且它们的近端算子都易于计算
    """
    
    def __init__(self,
                 max_iter: int = 1000,
                 tolerance: float = 1e-6,
                 relaxation: float = 1.0,
                 verbose: bool = False,
                 record_history: bool = True):
        """
        初始化Douglas-Rachford算法
        
        Args:
            max_iter: 最大迭代次数
            tolerance: 收敛容差
            relaxation: 松弛参数 (通常为1.0)
            verbose: 是否打印详细信息
            record_history: 是否记录历史信息
        """
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.relaxation = relaxation
        self.verbose = verbose
        self.record_history = record_history
        
        self.name = "Douglas-Rachford分裂法"
    
    def solve(self, problem: OptimizationProblem, 
              x0: Optional[np.ndarray] = None,
              prox_f: Optional[Callable] = None,
              prox_g: Optional[Callable] = None) -> Tuple[np.ndarray, dict]:
        """
        使用Douglas-Rachford算法求解
        
        Args:
            problem: 优化问题
            x0: 初始点
            prox_f: f的近端算子函数
            prox_g: g的近端算子函数
            
        Returns:
            (最终解, 历史信息字典)
        """
        # 初始化
        if x0 is None:
            x = np.random.randn(problem.dimension)
        else:
            x = x0.copy()
        
        z = x.copy()  # 辅助变量
        
        # 如果没有提供近端算子，使用问题的默认近端算子
        if prox_g is None:
            prox_g = lambda x, alpha: problem.proximal_operator(x, alpha)
        
        if prox_f is None:
            # 对于没有显式分解的问题，使用梯度下降作为近似
            def prox_f_default(x, alpha):
                grad = problem.gradient(x)
                if grad is None:
                    grad = problem.subgradient(x)
                return x - alpha * grad
            prox_f = prox_f_default
        
        # 历史记录
        history = {
            'x_values': [],
            'z_values': [],
            'objective_values': [],
            'residuals': [],
            'iterations': 0,
            'converged': False,
            'convergence_reason': ''
        }
        
        if self.verbose:
            print(f"开始求解 - 算法: {self.name}")
        
        # 主循环
        for k in range(self.max_iter):
            z_old = z.copy()
            
            # Douglas-Rachford迭代
            # Step 1: x = prox_g(z)
            x = prox_g(z, 1.0)
            
            # Step 2: y = prox_f(2x - z)
            y = prox_f(2*x - z, 1.0)
            
            # Step 3: z = z + λ(y - x)
            z = z + self.relaxation * (y - x)
            
            # 记录历史
            if self.record_history:
                history['x_values'].append(x.copy())
                history['z_values'].append(z.copy())
                history['objective_values'].append(problem.objective(x))
                
                # 残差
                residual = np.linalg.norm(z - z_old)
                history['residuals'].append(residual)
            
            # 检查收敛
            residual = np.linalg.norm(z - z_old)
            if residual < self.tolerance:
                history['converged'] = True
                history['convergence_reason'] = '残差小于容差'
                break
            
            # 打印进度
            if self.verbose and (k + 1) % 100 == 0:
                print(f"迭代 {k+1:4d}: f(x) = {problem.objective(x):.6f}, 残差 = {residual:.6f}")
        
        history['iterations'] = k + 1
        
        if not history['converged']:
            history['convergence_reason'] = '达到最大迭代次数'
        
        if self.verbose:
            print(f"求解完成 - 迭代次数: {history['iterations']}")
            print(f"收敛原因: {history['convergence_reason']}")
            print(f"最终目标函数值: {problem.objective(x):.6f}")
        
        return x, history

class ADMM:
    """
    交替方向乘子法 (Alternating Direction Method of Multipliers)
    
    用于求解形式为:
    min_{x,y} f(x) + g(y)
    s.t.      Ax + By = c
    的问题
    """
    
    def __init__(self,
                 penalty_param: float = 1.0,
                 max_iter: int = 1000,
                 tolerance: float = 1e-6,
                 verbose: bool = False,
                 record_history: bool = True):
        """
        初始化ADMM算法
        
        Args:
            penalty_param: 惩罚参数 ρ
            max_iter: 最大迭代次数
            tolerance: 收敛容差
            verbose: 是否打印详细信息
            record_history: 是否记录历史信息
        """
        self.penalty_param = penalty_param
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.verbose = verbose
        self.record_history = record_history
        
        self.name = "ADMM"
    
    def solve_consensus(self, problems: List[OptimizationProblem],
                       x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """
        使用ADMM求解一致性问题
        
        min Σ f_i(x_i)
        s.t. x_i = z ∀i
        
        Args:
            problems: 问题列表
            x0: 初始点
            
        Returns:
            (最终解, 历史信息字典)
        """
        n_problems = len(problems)
        dimension = problems[0].dimension
        
        # 初始化变量
        if x0 is None:
            x = [np.random.randn(dimension) for _ in range(n_problems)]
        else:
            x = [x0.copy() for _ in range(n_problems)]
        
        z = np.mean(x, axis=0)  # 一致性变量
        u = [np.zeros(dimension) for _ in range(n_problems)]  # 对偶变量
        
        # 历史记录
        history = {
            'objective_values': [],
            'primal_residuals': [],
            'dual_residuals': [],
            'x_values': [],
            'z_values': [],
            'iterations': 0,
            'converged': False,
            'convergence_reason': ''
        }
        
        if self.verbose:
            print(f"开始求解 - 算法: {self.name} (一致性问题)")
        
        # 主循环
        for k in range(self.max_iter):
            x_old = [xi.copy() for xi in x]
            z_old = z.copy()
            
            # x-最小化步骤
            for i in range(n_problems):
                # 求解: min f_i(x_i) + (ρ/2)||x_i - z + u_i||^2
                def augmented_objective(xi):
                    return (problems[i].objective(xi) + 
                            0.5 * self.penalty_param * np.sum((xi - z + u[i])**2))
                
                # 使用近端算子或梯度步骤
                try:
                    # 尝试使用近端算子
                    target = z - u[i]
                    x[i] = problems[i].proximal_operator(target, 1.0/self.penalty_param)
                except:
                    # 如果近端算子不可用，使用梯度下降
                    grad = problems[i].subgradient(x[i])
                    aug_grad = grad + self.penalty_param * (x[i] - z + u[i])
                    x[i] = x[i] - 0.01 * aug_grad
            
            # z-更新（一致性变量）
            z = np.mean([x[i] + u[i] for i in range(n_problems)], axis=0)
            
            # u-更新（对偶变量）
            for i in range(n_problems):
                u[i] = u[i] + x[i] - z
            
            # 计算残差
            primal_residual = np.sqrt(sum(np.sum((x[i] - z)**2) for i in range(n_problems)))
            dual_residual = self.penalty_param * np.linalg.norm(z - z_old)
            
            # 记录历史
            if self.record_history:
                total_obj = sum(problems[i].objective(x[i]) for i in range(n_problems))
                history['objective_values'].append(total_obj)
                history['primal_residuals'].append(primal_residual)
                history['dual_residuals'].append(dual_residual)
                history['x_values'].append([xi.copy() for xi in x])
                history['z_values'].append(z.copy())
            
            # 检查收敛
            if primal_residual < self.tolerance and dual_residual < self.tolerance:
                history['converged'] = True
                history['convergence_reason'] = '原始和对偶残差都小于容差'
                break
            
            # 打印进度
            if self.verbose and (k + 1) % 50 == 0:
                total_obj = sum(problems[i].objective(x[i]) for i in range(n_problems))
                print(f"迭代 {k+1:4d}: 目标值 = {total_obj:.6f}, 原始残差 = {primal_residual:.6f}, 对偶残差 = {dual_residual:.6f}")
        
        history['iterations'] = k + 1
        
        if not history['converged']:
            history['convergence_reason'] = '达到最大迭代次数'
        
        if self.verbose:
            print(f"求解完成 - 迭代次数: {history['iterations']}")
            print(f"收敛原因: {history['convergence_reason']}")
        
        # 返回一致性解
        return z, history
    
    def solve_separable(self, problem: OptimizationProblem,
                       A: np.ndarray, B: np.ndarray, c: np.ndarray,
                       x0: Optional[np.ndarray] = None,
                       y0: Optional[np.ndarray] = None) -> Tuple[Tuple[np.ndarray, np.ndarray], dict]:
        """
        使用ADMM求解可分离约束问题
        
        min f(x) + g(y)
        s.t. Ax + By = c
        
        Args:
            problem: 优化问题（包含f和g）
            A, B: 约束矩阵
            c: 约束向量
            x0, y0: 初始点
            
        Returns:
            ((x, y), 历史信息字典)
        """
        # 这里提供一个简化的实现框架
        if x0 is None:
            x = np.random.randn(A.shape[1])
        else:
            x = x0.copy()
        
        if y0 is None:
            y = np.random.randn(B.shape[1])
        else:
            y = y0.copy()
        
        lambda_dual = np.zeros(A.shape[0])  # 对偶变量
        
        history = {
            'objective_values': [],
            'constraint_violations': [],
            'iterations': 0,
            'converged': False,
            'convergence_reason': ''
        }
        
        for k in range(self.max_iter):
            # x-最小化
            # 这里需要根据具体问题实现
            
            # y-最小化
            # 这里需要根据具体问题实现
            
            # λ-更新
            constraint_violation = A @ x + B @ y - c
            lambda_dual = lambda_dual + self.penalty_param * constraint_violation
            
            # 检查收敛
            if np.linalg.norm(constraint_violation) < self.tolerance:
                history['converged'] = True
                history['convergence_reason'] = '约束违反小于容差'
                break
            
            if self.record_history:
                history['objective_values'].append(problem.objective(x))  # 简化
                history['constraint_violations'].append(np.linalg.norm(constraint_violation))
        
        history['iterations'] = k + 1
        return (x, y), history

class ForwardBackwardSplitting:
    """
    前向-后向分裂算法
    
    用于求解 min_x f(x) + g(x) 形式的问题
    其中 f 光滑，g 非光滑但近端算子易计算
    """
    
    def __init__(self,
                 step_size: float = 0.01,
                 max_iter: int = 1000,
                 tolerance: float = 1e-6,
                 verbose: bool = False,
                 record_history: bool = True):
        """
        初始化前向-后向分裂算法
        
        Args:
            step_size: 步长
            max_iter: 最大迭代次数
            tolerance: 收敛容差
            verbose: 是否打印详细信息
            record_history: 是否记录历史信息
        """
        self.step_size = step_size
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.verbose = verbose
        self.record_history = record_history
        
        self.name = "前向-后向分裂法"
    
    def solve(self, problem: OptimizationProblem, 
              x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """
        使用前向-后向分裂算法求解
        
        这实际上等同于近端梯度法
        """
        # 初始化
        if x0 is None:
            x = np.random.randn(problem.dimension)
        else:
            x = x0.copy()
        
        history = {
            'x_values': [],
            'objective_values': [],
            'iterations': 0,
            'converged': False,
            'convergence_reason': ''
        }
        
        if self.verbose:
            print(f"开始求解 - 算法: {self.name}")
        
        for k in range(self.max_iter):
            x_old = x.copy()
            
            # 前向步：计算光滑部分的梯度
            grad = problem.gradient(x)
            if grad is None:
                grad = problem.subgradient(x)
            
            # 后向步：应用非光滑部分的近端算子
            y = x - self.step_size * grad
            x = problem.proximal_operator(y, self.step_size)
            
            # 记录历史
            if self.record_history:
                history['x_values'].append(x.copy())
                history['objective_values'].append(problem.objective(x))
            
            # 检查收敛
            if np.linalg.norm(x - x_old) < self.tolerance:
                history['converged'] = True
                history['convergence_reason'] = '迭代点差值小于容差'
                break
            
            if self.verbose and (k + 1) % 100 == 0:
                print(f"迭代 {k+1:4d}: f(x) = {problem.objective(x):.6f}")
        
        history['iterations'] = k + 1
        
        if not history['converged']:
            history['convergence_reason'] = '达到最大迭代次数'
        
        return x, history 