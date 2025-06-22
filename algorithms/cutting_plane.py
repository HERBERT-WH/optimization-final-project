"""
切割平面法 (Cutting Plane Methods) 实现

基于用户提供的算法描述:
- 通过添加线性不等式（"切割"）迭代细化可行集或目标函数下界模型
- 包含经典Kelley方法及其改进变体
"""

import numpy as np
import time
from typing import Tuple, List, Optional, Union
from scipy.optimize import linprog, minimize
from ..problems.base import OptimizationProblem

class KelleyMethod:
    """
    Kelley切割平面法实现
    
    通过构建目标函数的分段线性下界模型并最小化该模型来迭代求解
    """
    
    def __init__(self,
                 max_iter: int = 1000,
                 tolerance: float = 1e-6,
                 max_cuts: int = 100,
                 verbose: bool = False,
                 record_history: bool = True):
        """
        初始化Kelley方法
        
        Args:
            max_iter: 最大迭代次数
            tolerance: 收敛容差
            max_cuts: 最大切割数量
            verbose: 是否打印详细信息
            record_history: 是否记录历史信息
        """
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.max_cuts = max_cuts
        self.verbose = verbose
        self.record_history = record_history
        
        self.name = "Kelley切割平面法"
        
        # 切割信息存储
        self.cut_points = []      # 切割点
        self.cut_values = []      # 切割点函数值
        self.cut_subgrads = []    # 切割点次梯度
    
    def _add_cut(self, x: np.ndarray, f_val: float, subgrad: np.ndarray):
        """
        添加新的切割
        
        Args:
            x: 切割点
            f_val: 函数值
            subgrad: 次梯度
        """
        self.cut_points.append(x.copy())
        self.cut_values.append(f_val)
        self.cut_subgrads.append(subgrad.copy())
        
        # 如果切割太多，移除最旧的
        if len(self.cut_points) > self.max_cuts:
            self.cut_points.pop(0)
            self.cut_values.pop(0)
            self.cut_subgrads.pop(0)
    
    def _solve_linear_program(self, bounds: Optional[Tuple] = None) -> Tuple[np.ndarray, float]:
        """
        求解线性规划子问题
        
        min_x,t  t
        s.t.     f(x_i) + g_i^T(x - x_i) ≤ t  ∀i
                 bounds constraints
        
        Args:
            bounds: 变量界限
            
        Returns:
            (最优解, 最优值)
        """
        if not self.cut_points:
            # 如果没有切割，返回零向量
            return np.zeros(self.cut_points[0].shape[0] if self.cut_points else 1), 0.0
        
        n = len(self.cut_points[0])  # 变量维度
        m = len(self.cut_points)     # 切割数量
        
        # 变量: [x, t]，其中x是n维，t是1维
        c = np.zeros(n + 1)
        c[-1] = 1  # 目标函数：min t
        
        # 不等式约束: A_ub @ [x, t] <= b_ub
        # 每个切割: g_i^T @ x - t <= g_i^T @ x_i - f_i
        A_ub = np.zeros((m, n + 1))
        b_ub = np.zeros(m)
        
        for i, (x_i, f_i, g_i) in enumerate(zip(self.cut_points, 
                                                 self.cut_values, 
                                                 self.cut_subgrads)):
            A_ub[i, :n] = g_i      # x的系数
            A_ub[i, n] = -1        # t的系数
            b_ub[i] = np.dot(g_i, x_i) - f_i
        
        # 变量界限
        if bounds is None:
            bounds_list = [(None, None)] * n + [(None, None)]  # x无界限，t无界限
        else:
            bounds_list = [bounds] * n + [(None, None)]
        
        try:
            # 求解线性规划
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_list, method='highs')
            
            if result.success:
                x_opt = result.x[:n]
                t_opt = result.x[n]
                return x_opt, t_opt
            else:
                # 如果求解失败，返回最后一个切割点
                return self.cut_points[-1], self.cut_values[-1]
        except:
            # 如果出现数值问题，返回最后一个切割点
            return self.cut_points[-1], self.cut_values[-1]
    
    def solve(self, problem: OptimizationProblem, 
              x0: Optional[np.ndarray] = None,
              bounds: Optional[Tuple] = None) -> Tuple[np.ndarray, dict]:
        """
        使用Kelley方法求解优化问题
        
        Args:
            problem: 优化问题实例
            x0: 初始点
            bounds: 变量界限 (lower, upper)
            
        Returns:
            (最终解, 历史信息字典)
        """
        # 初始化
        if x0 is None:
            x = np.random.randn(problem.dimension)
        else:
            x = x0.copy()
        
        # 清空切割
        self.cut_points = []
        self.cut_values = []
        self.cut_subgrads = []
        
        # 历史记录
        history = {
            'x_values': [],
            'objective_values': [],
            'model_values': [],
            'cut_counts': [],
            'iterations': 0,
            'converged': False,
            'convergence_reason': ''
        }
        
        if self.verbose:
            print(f"开始求解 - 算法: {self.name}")
        
        # 计算初始点信息并添加第一个切割
        f_current = problem.objective(x)
        subgrad_current = problem.subgradient(x)
        self._add_cut(x, f_current, subgrad_current)
        
        if self.verbose:
            print(f"初始目标函数值: {f_current:.6f}")
        
        # 主循环
        for k in range(self.max_iter):
            # 记录历史
            if self.record_history:
                history['x_values'].append(x.copy())
                history['objective_values'].append(f_current)
                history['cut_counts'].append(len(self.cut_points))
            
            # 检查收敛条件
            subgrad_norm = np.linalg.norm(subgrad_current)
            if subgrad_norm < self.tolerance:
                history['converged'] = True
                history['convergence_reason'] = '次梯度范数小于容差'
                break
            
            # 求解线性规划子问题
            x_new, model_val = self._solve_linear_program(bounds)
            
            if self.record_history:
                history['model_values'].append(model_val)
            
            # 计算新点的函数值和次梯度
            f_new = problem.objective(x_new)
            subgrad_new = problem.subgradient(x_new)
            
            # 添加新的切割
            self._add_cut(x_new, f_new, subgrad_new)
            
            # 检查改进
            improvement = f_current - f_new
            if improvement < self.tolerance:
                history['converged'] = True
                history['convergence_reason'] = '函数值改进小于容差'
                break
            
            # 更新当前点
            x = x_new
            f_current = f_new
            subgrad_current = subgrad_new
            
            # 打印进度
            if self.verbose and (k + 1) % 10 == 0:
                print(f"迭代 {k+1:4d}: f(x) = {f_current:.6f}, 切割数 = {len(self.cut_points)}")
        
        history['iterations'] = k + 1
        
        if not history['converged']:
            history['convergence_reason'] = '达到最大迭代次数'
        
        if self.verbose:
            print(f"求解完成 - 迭代次数: {history['iterations']}")
            print(f"收敛原因: {history['convergence_reason']}")
            print(f"最终目标函数值: {f_current:.6f}")
            print(f"总切割数: {len(self.cut_points)}")
        
        return x, history

class StabilizedCuttingPlane:
    """
    稳定化切割平面法
    
    在经典Kelley方法基础上加入稳定化项以避免数值不稳定
    """
    
    def __init__(self,
                 stabilization: float = 1.0,
                 *args, **kwargs):
        """
        初始化稳定化切割平面法
        
        Args:
            stabilization: 稳定化参数
        """
        self.kelley = KelleyMethod(*args, **kwargs)
        self.stabilization = stabilization
        self.name = "稳定化切割平面法"
    
    def _solve_stabilized_subproblem(self, x_center: np.ndarray, 
                                   bounds: Optional[Tuple] = None) -> Tuple[np.ndarray, float]:
        """
        求解稳定化子问题
        
        min_x,t  t + (μ/2)||x - x_center||^2
        s.t.     f(x_i) + g_i^T(x - x_i) ≤ t  ∀i
        
        Args:
            x_center: 稳定化中心
            bounds: 变量界限
            
        Returns:
            (最优解, 最优值)
        """
        if not self.kelley.cut_points:
            return x_center, 0.0
        
        n = len(x_center)
        
        def objective(vars):
            x = vars[:n]
            t = vars[n]
            
            # 计算最大切割值
            max_cut_val = -np.inf
            for x_i, f_i, g_i in zip(self.kelley.cut_points, 
                                     self.kelley.cut_values, 
                                     self.kelley.cut_subgrads):
                cut_val = f_i + np.dot(g_i, x - x_i)
                max_cut_val = max(max_cut_val, cut_val)
            
            # 如果t小于最大切割值，返回大的惩罚值
            if t < max_cut_val - 1e-6:
                return 1e10
            
            # 目标函数：t + 稳定化项
            return t + 0.5 * self.stabilization * np.sum((x - x_center)**2)
        
        # 初始猜测
        x0_vars = np.concatenate([x_center, [0.0]])
        
        # 变量界限
        if bounds is None:
            bounds_list = [(None, None)] * n + [(None, None)]
        else:
            bounds_list = [bounds] * n + [(None, None)]
        
        try:
            result = minimize(objective, x0_vars, bounds=bounds_list, method='L-BFGS-B')
            if result.success:
                x_opt = result.x[:n]
                t_opt = result.x[n]
                return x_opt, t_opt
            else:
                return x_center, objective(x0_vars)
        except:
            return x_center, objective(x0_vars)
    
    def solve(self, problem: OptimizationProblem, 
              x0: Optional[np.ndarray] = None,
              bounds: Optional[Tuple] = None) -> Tuple[np.ndarray, dict]:
        """
        使用稳定化切割平面法求解
        """
        # 使用修改的求解策略
        if x0 is None:
            x = np.random.randn(problem.dimension)
        else:
            x = x0.copy()
        
        x_center = x.copy()  # 稳定化中心
        
        # 清空切割
        self.kelley.cut_points = []
        self.kelley.cut_values = []
        self.kelley.cut_subgrads = []
        
        history = {
            'x_values': [],
            'objective_values': [],
            'model_values': [],
            'cut_counts': [],
            'iterations': 0,
            'converged': False,
            'convergence_reason': ''
        }
        
        # 添加初始切割
        f_current = problem.objective(x)
        subgrad_current = problem.subgradient(x)
        self.kelley._add_cut(x, f_current, subgrad_current)
        
        for k in range(self.kelley.max_iter):
            if self.kelley.record_history:
                history['x_values'].append(x.copy())
                history['objective_values'].append(f_current)
                history['cut_counts'].append(len(self.kelley.cut_points))
            
            # 检查收敛
            if np.linalg.norm(subgrad_current) < self.kelley.tolerance:
                history['converged'] = True
                history['convergence_reason'] = '次梯度范数小于容差'
                break
            
            # 求解稳定化子问题
            x_new, model_val = self._solve_stabilized_subproblem(x_center, bounds)
            
            if self.kelley.record_history:
                history['model_values'].append(model_val)
            
            # 计算新点信息
            f_new = problem.objective(x_new)
            subgrad_new = problem.subgradient(x_new)
            
            # 添加切割
            self.kelley._add_cut(x_new, f_new, subgrad_new)
            
            # 更新
            if f_new < f_current:
                x = x_new
                f_current = f_new
                subgrad_current = subgrad_new
                x_center = x.copy()  # 更新稳定化中心
            
            if self.kelley.verbose and (k + 1) % 10 == 0:
                print(f"迭代 {k+1:4d}: f(x) = {f_current:.6f}")
        
        history['iterations'] = k + 1
        return x, history

class CuttingPlaneWithTrustRegion:
    """
    带信赖域的切割平面法
    
    结合信赖域策略来改善切割平面法的稳定性
    """
    
    def __init__(self,
                 initial_radius: float = 1.0,
                 max_radius: float = 10.0,
                 eta1: float = 0.1,
                 eta2: float = 0.75,
                 gamma1: float = 0.5,
                 gamma2: float = 2.0,
                 *args, **kwargs):
        """
        初始化带信赖域的切割平面法
        
        Args:
            initial_radius: 初始信赖域半径
            max_radius: 最大信赖域半径
            eta1, eta2: 信赖域更新参数
            gamma1, gamma2: 半径调整因子
        """
        self.kelley = KelleyMethod(*args, **kwargs)
        self.initial_radius = initial_radius
        self.max_radius = max_radius
        self.eta1 = eta1
        self.eta2 = eta2
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.name = "信赖域切割平面法"
    
    def solve(self, problem: OptimizationProblem, 
              x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """
        使用信赖域切割平面法求解
        """
        # 基本实现，结合信赖域约束的线性规划
        if x0 is None:
            x = np.random.randn(problem.dimension)
        else:
            x = x0.copy()
        
        radius = self.initial_radius
        
        # 使用修改的界限来实现信赖域
        for k in range(self.kelley.max_iter):
            # 设置信赖域界限
            bounds = (-radius + x, radius + x)
            
            # 使用Kelley方法的一步
            x_new, _ = self.kelley._solve_linear_program(bounds)
            
            # 计算实际减少和预测减少
            f_old = problem.objective(x)
            f_new = problem.objective(x_new)
            
            actual_reduction = f_old - f_new
            
            # 信赖域更新逻辑
            if actual_reduction > 0:
                x = x_new
                if actual_reduction > self.eta2 * abs(f_old):
                    radius = min(self.gamma2 * radius, self.max_radius)
            else:
                radius = self.gamma1 * radius
            
            # 添加切割
            subgrad = problem.subgradient(x)
            self.kelley._add_cut(x, problem.objective(x), subgrad)
        
        history = {'iterations': k + 1, 'converged': False, 'convergence_reason': '达到最大迭代次数'}
        return x, history 