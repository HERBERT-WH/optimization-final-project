"""
近端梯度法 (Proximal Gradient Method) 实现

基于用户提供的算法描述:
- 专为复合问题 min f(x) + g(x) 设计
- 迭代公式: x^{k+1} = prox_{αg}(x^k - α∇f(x^k))
- 包含加速变体 FISTA
"""

import numpy as np
import time
from typing import Tuple, List, Optional, Union
from ..problems.base import OptimizationProblem

class ProximalGradientMethod:
    """
    近端梯度法实现
    
    适用于复合优化问题 min_x f(x) + g(x)
    其中 f 光滑凸，g 非光滑凸但近端算子易计算
    """
    
    def __init__(self,
                 step_size: Union[float, str] = 0.01,
                 max_iter: int = 1000,
                 tolerance: float = 1e-6,
                 line_search: bool = False,
                 backtrack_factor: float = 0.8,
                 verbose: bool = False,
                 record_history: bool = True):
        """
        初始化近端梯度法
        
        Args:
            step_size: 步长或步长策略
                - float: 固定步长
                - 'backtrack': 回溯线搜索
                - 'adaptive': 自适应步长
            max_iter: 最大迭代次数
            tolerance: 收敛容差
            line_search: 是否使用线搜索
            backtrack_factor: 回溯因子
            verbose: 是否打印详细信息
            record_history: 是否记录历史信息
        """
        self.step_size = step_size
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.line_search = line_search
        self.backtrack_factor = backtrack_factor
        self.verbose = verbose
        self.record_history = record_history
        
        self.name = "近端梯度法"
    
    def _backtrack_line_search(self, x: np.ndarray, grad: np.ndarray, 
                              problem: OptimizationProblem, 
                              initial_step: float = 1.0) -> float:
        """
        回溯线搜索确定步长
        
        Args:
            x: 当前点
            grad: 当前梯度
            problem: 优化问题
            initial_step: 初始步长
            
        Returns:
            合适的步长
        """
        alpha = initial_step
        beta = self.backtrack_factor
        c = 1e-4  # Armijo条件参数
        
        f_x = problem.objective(x)
        
        for _ in range(20):  # 最多回溯20次
            # 近端梯度步
            y = problem.proximal_operator(x - alpha * grad, alpha)
            f_y = problem.objective(y)
            
            # 检查Armijo条件的变体（适用于近端梯度）
            expected_decrease = c * np.dot(grad, x - y)
            if f_y <= f_x + expected_decrease:
                break
            
            alpha *= beta
        
        return alpha
    
    def solve(self, problem: OptimizationProblem, 
              x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """
        求解复合优化问题
        
        Args:
            problem: 优化问题实例
            x0: 初始点
            
        Returns:
            (最终解, 历史信息字典)
        """
        # 初始化
        if x0 is None:
            x = np.random.randn(problem.dimension)
        else:
            x = x0.copy()
        
        # 历史记录
        history = {
            'x_values': [],
            'objective_values': [],
            'step_sizes': [],
            'gradient_norms': [],
            'iterations': 0,
            'converged': False,
            'convergence_reason': ''
        }
        
        if self.verbose:
            print(f"开始求解 - 算法: {self.name}")
            print(f"初始目标函数值: {problem.objective(x):.6f}")
        
        # 主循环
        for k in range(self.max_iter):
            # 计算目标函数值和梯度
            f_val = problem.objective(x)
            
            # 尝试获取梯度，如果不可微则使用次梯度
            grad = problem.gradient(x)
            if grad is None:
                grad = problem.subgradient(x)
            
            # 记录历史
            if self.record_history:
                history['x_values'].append(x.copy())
                history['objective_values'].append(f_val)
                history['gradient_norms'].append(np.linalg.norm(grad))
            
            # 确定步长
            if self.line_search or self.step_size == 'backtrack':
                alpha = self._backtrack_line_search(x, grad, problem)
            elif isinstance(self.step_size, float):
                alpha = self.step_size
            elif self.step_size == 'adaptive':
                # 简单的自适应步长
                L = problem.lipschitz_constant()
                alpha = 1.0 / L if L is not None else 0.01
            else:
                alpha = 0.01
            
            if self.record_history:
                history['step_sizes'].append(alpha)
            
            # 近端梯度步
            y = x - alpha * grad
            x_new = problem.proximal_operator(y, alpha)
            
            # 检查收敛条件
            diff_norm = np.linalg.norm(x_new - x)
            if diff_norm < self.tolerance:
                history['converged'] = True
                history['convergence_reason'] = '迭代点差值小于容差'
                break
            
            x = x_new
            
            # 打印进度
            if self.verbose and (k + 1) % 100 == 0:
                print(f"迭代 {k+1:4d}: f(x) = {f_val:.6f}, ||∇f|| = {np.linalg.norm(grad):.6f}, α = {alpha:.6f}")
        
        history['iterations'] = k + 1
        
        if not history['converged']:
            history['convergence_reason'] = '达到最大迭代次数'
        
        if self.verbose:
            print(f"求解完成 - 迭代次数: {history['iterations']}")
            print(f"收敛原因: {history['convergence_reason']}")
            print(f"最终目标函数值: {problem.objective(x):.6f}")
        
        return x, history

class FISTA(ProximalGradientMethod):
    """
    FISTA: Fast Iterative Shrinkage-Thresholding Algorithm
    
    近端梯度法的加速版本，收敛速度 O(1/k^2)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "FISTA"
    
    def solve(self, problem: OptimizationProblem, 
              x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """
        FISTA算法求解
        
        使用动量项加速收敛
        """
        # 初始化
        if x0 is None:
            x = np.random.randn(problem.dimension)
        else:
            x = x0.copy()
        
        y = x.copy()  # 外推点
        t = 1.0  # 动量参数
        
        # 历史记录
        history = {
            'x_values': [],
            'objective_values': [],
            'step_sizes': [],
            'gradient_norms': [],
            'momentum_params': [],
            'iterations': 0,
            'converged': False,
            'convergence_reason': ''
        }
        
        if self.verbose:
            print(f"开始求解 - 算法: {self.name}")
            print(f"初始目标函数值: {problem.objective(x):.6f}")
        
        # 主循环
        for k in range(self.max_iter):
            x_prev = x.copy()
            
            # 在外推点计算梯度
            f_val = problem.objective(y)
            grad = problem.gradient(y)
            if grad is None:
                grad = problem.subgradient(y)
            
            # 记录历史
            if self.record_history:
                history['x_values'].append(x.copy())
                history['objective_values'].append(problem.objective(x))
                history['gradient_norms'].append(np.linalg.norm(grad))
                history['momentum_params'].append(t)
            
            # 确定步长
            if self.line_search or self.step_size == 'backtrack':
                alpha = self._backtrack_line_search(y, grad, problem)
            elif isinstance(self.step_size, float):
                alpha = self.step_size
            else:
                L = problem.lipschitz_constant()
                alpha = 1.0 / L if L is not None else 0.01
            
            if self.record_history:
                history['step_sizes'].append(alpha)
            
            # 近端梯度步
            z = y - alpha * grad
            x = problem.proximal_operator(z, alpha)
            
            # 更新动量参数
            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
            
            # 更新外推点
            beta = (t - 1) / t_new
            y = x + beta * (x - x_prev)
            
            t = t_new
            
            # 检查收敛条件
            diff_norm = np.linalg.norm(x - x_prev)
            if diff_norm < self.tolerance:
                history['converged'] = True
                history['convergence_reason'] = '迭代点差值小于容差'
                break
            
            # 打印进度
            if self.verbose and (k + 1) % 100 == 0:
                print(f"迭代 {k+1:4d}: f(x) = {problem.objective(x):.6f}, β = {beta:.4f}")
        
        history['iterations'] = k + 1
        
        if not history['converged']:
            history['convergence_reason'] = '达到最大迭代次数'
        
        if self.verbose:
            print(f"求解完成 - 迭代次数: {history['iterations']}")
            print(f"收敛原因: {history['convergence_reason']}")
            print(f"最终目标函数值: {problem.objective(x):.6f}")
        
        return x, history

class AcceleratedProximalGradient(FISTA):
    """
    加速近端梯度法的通用实现
    
    包含多种加速策略
    """
    
    def __init__(self, acceleration_type: str = 'nesterov', *args, **kwargs):
        """
        Args:
            acceleration_type: 加速类型
                - 'nesterov': Nesterov加速
                - 'fista': FISTA加速
                - 'adaptive': 自适应重启
        """
        super().__init__(*args, **kwargs)
        self.acceleration_type = acceleration_type
        self.name = f"加速近端梯度法 ({acceleration_type})"
    
    def solve(self, problem: OptimizationProblem, 
              x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """
        根据指定的加速类型求解
        """
        if self.acceleration_type == 'adaptive':
            return self._solve_with_adaptive_restart(problem, x0)
        else:
            return super().solve(problem, x0)
    
    def _solve_with_adaptive_restart(self, problem: OptimizationProblem, 
                                   x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """
        带有自适应重启的加速近端梯度法
        """
        # 基本的FISTA实现，但加入重启机制
        if x0 is None:
            x = np.random.randn(problem.dimension)
        else:
            x = x0.copy()
        
        y = x.copy()
        t = 1.0
        
        history = {
            'x_values': [],
            'objective_values': [],
            'step_sizes': [],
            'gradient_norms': [],
            'restart_points': [],
            'iterations': 0,
            'converged': False,
            'convergence_reason': ''
        }
        
        for k in range(self.max_iter):
            x_prev = x.copy()
            y_prev = y.copy()
            
            # 标准FISTA步骤
            grad = problem.gradient(y)
            if grad is None:
                grad = problem.subgradient(y)
            
            alpha = self.step_size if isinstance(self.step_size, float) else 0.01
            
            z = y - alpha * grad
            x = problem.proximal_operator(z, alpha)
            
            # 检查是否需要重启
            if np.dot(y - x, x - x_prev) > 0:
                # 重启：重置动量
                t = 1.0
                y = x.copy()
                if self.record_history:
                    history['restart_points'].append(k)
            else:
                # 正常更新
                t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
                beta = (t - 1) / t_new
                y = x + beta * (x - x_prev)
                t = t_new
            
            # 记录历史
            if self.record_history:
                history['x_values'].append(x.copy())
                history['objective_values'].append(problem.objective(x))
                history['gradient_norms'].append(np.linalg.norm(grad))
                history['step_sizes'].append(alpha)
            
            # 检查收敛
            if np.linalg.norm(x - x_prev) < self.tolerance:
                history['converged'] = True
                history['convergence_reason'] = '迭代点差值小于容差'
                break
        
        history['iterations'] = k + 1
        return x, history 