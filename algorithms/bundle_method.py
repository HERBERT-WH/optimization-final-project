"""
束方法 (Bundle Methods) 实现

基于用户提供的算法描述:
- 利用历史信息构建目标函数的精确分段线性下界模型
- 求解主问题（最小化模型 + 二次正则项）
- 执行充分改进测试决定serious step或null step
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from scipy.optimize import minimize
import warnings
from ..problems.base import OptimizationProblem
import time

class BundleMethod:
    """
    束方法 (Bundle Method)
    
    特点：利用历史次梯度信息构建目标函数的分段线性下界模型
    收敛速度：线性收敛
    适用：需要高精度解的中等规模问题
    """
    
    def __init__(self, problem, nu=0.1, m=1e-3):
        """
        初始化束方法
        
        参数:
            problem: 优化问题实例
            nu: 下降参数 (0 < nu < 1)
            m: 稳定化参数
        """
        self.problem = problem
        self.nu = nu
        self.m = m
        
    def solve(self, x0, max_iter=100, tol=1e-6, max_bundle_size=50, verbose=False):
        """
        求解优化问题
        
        参数:
            x0: 初始点
            max_iter: 最大迭代次数
            tol: 收敛容差
            max_bundle_size: 最大束大小
            verbose: 是否输出详细信息
            
        返回:
            字典包含解、历史信息等
        """
        x = np.array(x0, dtype=float)
        n = len(x)
        
        # 束信息存储
        bundle_points = []  # 历史点
        bundle_values = []  # 历史函数值
        bundle_subgrads = []  # 历史次梯度
        
        # 历史记录
        history = {
            'x_vals': [x.copy()],
            'f_vals': [],
            'bundle_sizes': []
        }
        
        start_time = time.time()
        
        for k in range(max_iter):
            # 计算当前点的函数值和次梯度
            f_x = self.problem.objective(x)
            g_x = self.problem.subgradient(x)
            
            # 添加到束中
            bundle_points.append(x.copy())
            bundle_values.append(f_x)
            bundle_subgrads.append(g_x.copy())
            
            # 记录历史
            history['f_vals'].append(f_x)
            history['bundle_sizes'].append(len(bundle_points))
            
            if verbose and (k + 1) % 10 == 0:
                print(f"迭代 {k+1}: f = {f_x:.6e}, 束大小 = {len(bundle_points)}")
            
            # 求解子问题：构建和求解分段线性模型
            try:
                x_new, gap = self._solve_subproblem(x, bundle_points, bundle_values, 
                                                   bundle_subgrads)
            except Exception as e:
                if verbose:
                    print(f"子问题求解失败: {e}")
                break
            
            # 检查收敛
            if gap < tol:
                if verbose:
                    print(f"迭代 {k}: 对偶间隙 {gap:.2e} < {tol}, 收敛")
                break
            
            # 计算新点的函数值
            f_new = self.problem.objective(x_new)
            
            # 计算预期下降和实际下降
            predicted_decrease = self.nu * gap
            actual_decrease = f_x - f_new
            
            if actual_decrease >= predicted_decrease:
                # 严重步：接受新点
                x = x_new
                if verbose:
                    print(f"  严重步: 下降 {actual_decrease:.6e}")
            else:
                # 空步：不移动但保留束信息
                if verbose:
                    print(f"  空步: 下降不足 {actual_decrease:.6e} < {predicted_decrease:.6e}")
            
            history['x_vals'].append(x.copy())
            
            # 束管理：限制束大小
            if len(bundle_points) > max_bundle_size:
                # 移除最旧的束元素
                bundle_points.pop(0)
                bundle_values.pop(0)
                bundle_subgrads.pop(0)
        
        end_time = time.time()
        
        result = {
            'x': x,
            'f_opt': self.problem.objective(x),
            'n_iter': k + 1,
            'time': end_time - start_time,
            'converged': gap < tol if 'gap' in locals() else False,
            'history': history
        }
        
        if verbose:
            print(f"\n束方法完成:")
            print(f"  迭代次数: {result['n_iter']}")
            print(f"  最优值: {result['f_opt']:.6e}")
            print(f"  运行时间: {result['time']:.3f}秒")
        
        return result
    
    def _solve_subproblem(self, x_current, bundle_points, bundle_values, bundle_subgrads):
        """
        求解束方法的子问题
        
        min_d { max_i [f_i + g_i^T d] + m/2 ||d||^2 }
        
        参数:
            x_current: 当前点
            bundle_points: 束中的点
            bundle_values: 束中的函数值
            bundle_subgrads: 束中的次梯度
            
        返回:
            新点和对偶间隙
        """
        n = len(x_current)
        num_bundle = len(bundle_points)
        
        if num_bundle == 0:
            return x_current, 0.0
        
        # 定义子问题的目标函数
        def subproblem_objective(d):
            # 计算最大值 max_i [f_i + g_i^T d]
            max_val = -np.inf
            for i in range(num_bundle):
                val = bundle_values[i] + np.dot(bundle_subgrads[i], d)
                max_val = max(max_val, val)
            
            # 加上稳定化项
            return max_val + 0.5 * self.m * np.dot(d, d)
        
        # 使用BFGS求解子问题
        try:
            result = minimize(subproblem_objective, np.zeros(n), method='BFGS')
            d_opt = result.x
            gap = -result.fun + bundle_values[-1]  # 对偶间隙
            
            x_new = x_current + d_opt
            return x_new, gap
            
        except Exception as e:
            # 如果数值优化失败，使用简单的次梯度步
            g_current = bundle_subgrads[-1]
            step_size = 0.01
            d_opt = -step_size * g_current
            x_new = x_current + d_opt
            gap = np.linalg.norm(g_current)
            return x_new, gap

class ProximalBundleMethod:
    """
    近端束方法 (Proximal Bundle Method)
    
    结合近端算子的束方法，适用于复合问题
    """
    
    def __init__(self, problem, nu=0.1, m=1e-3):
        """
        初始化近端束方法
        
        参数:
            problem: 优化问题实例（需要实现proximal_operator）
            nu: 下降参数
            m: 稳定化参数
        """
        self.problem = problem
        self.nu = nu
        self.m = m
        
        if not hasattr(problem, 'proximal_operator'):
            raise ValueError("问题必须实现proximal_operator方法")
    
    def solve(self, x0, max_iter=100, tol=1e-6, max_bundle_size=50, verbose=False):
        """
        求解优化问题
        
        参数:
            x0: 初始点
            max_iter: 最大迭代次数
            tol: 收敛容差
            max_bundle_size: 最大束大小
            verbose: 是否输出详细信息
            
        返回:
            字典包含解、历史信息等
        """
        x = np.array(x0, dtype=float)
        n = len(x)
        
        # 束信息存储
        bundle_points = []
        bundle_values = []
        bundle_subgrads = []
        
        # 历史记录
        history = {
            'x_vals': [x.copy()],
            'f_vals': [],
            'bundle_sizes': []
        }
        
        start_time = time.time()
        
        for k in range(max_iter):
            # 计算当前点的函数值和次梯度
            f_x = self.problem.objective(x)
            g_x = self.problem.subgradient(x)
            
            # 添加到束中
            bundle_points.append(x.copy())
            bundle_values.append(f_x)
            bundle_subgrads.append(g_x.copy())
            
            # 记录历史
            history['f_vals'].append(f_x)
            history['bundle_sizes'].append(len(bundle_points))
            
            if verbose and (k + 1) % 10 == 0:
                print(f"迭代 {k+1}: f = {f_x:.6e}, 束大小 = {len(bundle_points)}")
            
            # 求解近端子问题
            try:
                x_new, gap = self._solve_proximal_subproblem(x, bundle_points, 
                                                           bundle_values, bundle_subgrads)
            except Exception as e:
                if verbose:
                    print(f"子问题求解失败: {e}")
                break
            
            # 检查收敛
            if gap < tol:
                if verbose:
                    print(f"迭代 {k}: 对偶间隙 {gap:.2e} < {tol}, 收敛")
                break
            
            # 计算新点的函数值
            f_new = self.problem.objective(x_new)
            
            # 下降测试
            predicted_decrease = self.nu * gap
            actual_decrease = f_x - f_new
            
            if actual_decrease >= predicted_decrease:
                # 严重步
                x = x_new
                if verbose:
                    print(f"  严重步: 下降 {actual_decrease:.6e}")
            else:
                # 空步
                if verbose:
                    print(f"  空步: 下降不足")
            
            history['x_vals'].append(x.copy())
            
            # 束管理
            if len(bundle_points) > max_bundle_size:
                bundle_points.pop(0)
                bundle_values.pop(0)
                bundle_subgrads.pop(0)
        
        end_time = time.time()
        
        result = {
            'x': x,
            'f_opt': self.problem.objective(x),
            'n_iter': k + 1,
            'time': end_time - start_time,
            'converged': gap < tol if 'gap' in locals() else False,
            'history': history
        }
        
        if verbose:
            print(f"\n近端束方法完成:")
            print(f"  迭代次数: {result['n_iter']}")
            print(f"  最优值: {result['f_opt']:.6e}")
            print(f"  运行时间: {result['time']:.3f}秒")
        
        return result
    
    def _solve_proximal_subproblem(self, x_current, bundle_points, bundle_values, 
                                 bundle_subgrads):
        """
        求解近端束方法的子问题
        
        涉及近端算子的组合优化问题
        """
        n = len(x_current)
        num_bundle = len(bundle_points)
        
        if num_bundle == 0:
            return x_current, 0.0
        
        # 简化处理：使用最新的次梯度信息结合近端算子
        g_current = bundle_subgrads[-1]
        
        # 近端梯度步
        step_size = 1.0 / self.m
        y = x_current - step_size * g_current
        x_new = self.problem.proximal_operator(y, step_size)
        
        # 估计对偶间隙
        gap = np.linalg.norm(x_new - x_current) / step_size
        
        return x_new, gap 