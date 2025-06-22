"""
次梯度法 (Subgradient Method) 实现

基于用户提供的算法描述:
- 无约束问题：x^{k+1} = x^k - α_k g_k
- 约束问题：x^{k+1} = P_C(x^k - α_k g_k)
- 特性：简单但收敛缓慢，函数值可能非单调递减
"""

import numpy as np
import time
from typing import Tuple, List, Optional, Union, Callable
from ..problems.base import OptimizationProblem

class SubgradientMethod:
    """
    次梯度法实现
    
    支持多种步长策略:
    - 常数步长
    - 递减步长 (平方可和不可和序列)
    - 自适应步长
    """
    
    def __init__(self, 
                 step_size: Union[float, str, Callable] = 0.01,
                 max_iter: int = 1000,
                 tolerance: float = 1e-6,
                 step_size_params: Optional[dict] = None,
                 verbose: bool = False,
                 record_history: bool = True):
        """
        初始化次梯度法
        
        Args:
            step_size: 步长策略
                - float: 常数步长
                - 'diminishing': 递减步长 α_k = α_0 / sqrt(k+1)
                - 'nonsummable': 非可和递减 α_k = α_0 / (k+1)
                - 'adaptive': 自适应步长
                - Callable: 自定义步长函数 f(k) -> α_k
            max_iter: 最大迭代次数
            tolerance: 收敛容差
            step_size_params: 步长参数字典
            verbose: 是否打印详细信息
            record_history: 是否记录历史信息
        """
        self.step_size = step_size
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.step_size_params = step_size_params or {}
        self.verbose = verbose
        self.record_history = record_history
        
        # 算法名称
        self.name = "次梯度法"
    
    def _get_step_size(self, k: int, x: np.ndarray, subgrad: np.ndarray, 
                       problem: OptimizationProblem) -> float:
        """
        根据步长策略计算当前步长
        
        Args:
            k: 当前迭代次数
            x: 当前点
            subgrad: 当前次梯度
            problem: 优化问题
            
        Returns:
            当前步长
        """
        if isinstance(self.step_size, float):
            return self.step_size
        
        elif self.step_size == 'diminishing':
            # α_k = α_0 / sqrt(k+1)
            alpha_0 = self.step_size_params.get('alpha_0', 1.0)
            return alpha_0 / np.sqrt(k + 1)
        
        elif self.step_size == 'nonsummable':
            # α_k = α_0 / (k+1)
            alpha_0 = self.step_size_params.get('alpha_0', 1.0)
            return alpha_0 / (k + 1)
        
        elif self.step_size == 'adaptive':
            # 自适应步长基于次梯度范数
            alpha_0 = self.step_size_params.get('alpha_0', 1.0)
            beta = self.step_size_params.get('beta', 0.5)
            subgrad_norm = np.linalg.norm(subgrad)
            if subgrad_norm > 0:
                return alpha_0 * beta / subgrad_norm
            else:
                return alpha_0
        
        elif callable(self.step_size):
            return self.step_size(k)
        
        else:
            raise ValueError(f"未知的步长策略: {self.step_size}")
    
    def solve(self, problem: OptimizationProblem, 
              x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """
        求解优化问题
        
        Args:
            problem: 优化问题实例
            x0: 初始点，如果为None则随机初始化
            
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
            'subgradient_norms': [],
            'iterations': 0,
            'converged': False,
            'convergence_reason': ''
        }
        
        if self.verbose:
            print(f"开始求解 - 算法: {self.name}")
            print(f"初始目标函数值: {problem.objective(x):.6f}")
        
        # 主循环
        for k in range(self.max_iter):
            # 计算目标函数值和次梯度
            f_val = problem.objective(x)
            subgrad = problem.subgradient(x)
            
            # 记录历史
            if self.record_history:
                history['x_values'].append(x.copy())
                history['objective_values'].append(f_val)
                history['subgradient_norms'].append(np.linalg.norm(subgrad))
            
            # 检查收敛条件
            subgrad_norm = np.linalg.norm(subgrad)
            if subgrad_norm < self.tolerance:
                history['converged'] = True
                history['convergence_reason'] = '次梯度范数小于容差'
                break
            
            # 计算步长
            alpha_k = self._get_step_size(k, x, subgrad, problem)
            
            if self.record_history:
                history['step_sizes'].append(alpha_k)
            
            # 次梯度步
            x_new = x - alpha_k * subgrad
            
            # 投影到可行域（如果有约束）
            x_new = problem.projection(x_new)
            
            # 更新
            x = x_new
            
            # 打印进度
            if self.verbose and (k + 1) % 100 == 0:
                print(f"迭代 {k+1:4d}: f(x) = {f_val:.6f}, ||g|| = {subgrad_norm:.6f}, α = {alpha_k:.6f}")
        
        history['iterations'] = k + 1
        
        if not history['converged']:
            history['convergence_reason'] = '达到最大迭代次数'
        
        if self.verbose:
            print(f"求解完成 - 迭代次数: {history['iterations']}")
            print(f"收敛原因: {history['convergence_reason']}")
            print(f"最终目标函数值: {problem.objective(x):.6f}")
        
        return x, history

class ProjectedSubgradientMethod(SubgradientMethod):
    """
    投影次梯度法
    
    专门用于约束优化问题的次梯度法变体
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "投影次梯度法"

class AdaptiveSubgradientMethod(SubgradientMethod):
    """
    自适应次梯度法
    
    基于历史信息自动调整步长
    """
    
    def __init__(self, *args, **kwargs):
        # 强制使用自适应步长
        kwargs['step_size'] = 'adaptive'
        super().__init__(*args, **kwargs)
        self.name = "自适应次梯度法"
    
    def solve(self, problem: OptimizationProblem, 
              x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """
        带有自适应步长调整的求解过程
        """
        # 初始化
        if x0 is None:
            x = np.random.randn(problem.dimension)
        else:
            x = x0.copy()
        
        # 自适应参数
        best_f = float('inf')
        no_improvement_count = 0
        step_adjustment_factor = 0.8
        
        # 历史记录
        history = {
            'x_values': [],
            'objective_values': [],
            'step_sizes': [],
            'subgradient_norms': [],
            'iterations': 0,
            'converged': False,
            'convergence_reason': ''
        }
        
        current_alpha = self.step_size_params.get('alpha_0', 0.1)
        
        if self.verbose:
            print(f"开始求解 - 算法: {self.name}")
        
        # 主循环
        for k in range(self.max_iter):
            f_val = problem.objective(x)
            subgrad = problem.subgradient(x)
            
            # 记录历史
            if self.record_history:
                history['x_values'].append(x.copy())
                history['objective_values'].append(f_val)
                history['subgradient_norms'].append(np.linalg.norm(subgrad))
                history['step_sizes'].append(current_alpha)
            
            # 自适应步长调整
            if f_val < best_f:
                best_f = f_val
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count > 10:  # 连续10次无改进
                    current_alpha *= step_adjustment_factor
                    no_improvement_count = 0
            
            # 检查收敛
            subgrad_norm = np.linalg.norm(subgrad)
            if subgrad_norm < self.tolerance:
                history['converged'] = True
                history['convergence_reason'] = '次梯度范数小于容差'
                break
            
            # 次梯度步
            x_new = x - current_alpha * subgrad
            x_new = problem.projection(x_new)
            
            x = x_new
            
            if self.verbose and (k + 1) % 100 == 0:
                print(f"迭代 {k+1:4d}: f(x) = {f_val:.6f}, α = {current_alpha:.6f}")
        
        history['iterations'] = k + 1
        
        if not history['converged']:
            history['convergence_reason'] = '达到最大迭代次数'
        
        return x, history 