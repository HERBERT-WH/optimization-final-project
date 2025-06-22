"""
算法比较框架

提供统一的接口来比较不同的非光滑凸优化算法
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from typing import Dict, List, Any, Optional, Tuple
from algorithms import *
from problems.base import OptimizationProblem
from algorithms.subgradient import SubgradientMethod
from algorithms.proximal_gradient import ProximalGradientMethod, FISTA
from algorithms.bundle_method import BundleMethod
from algorithms.cutting_plane import KelleyMethod
from algorithms.splitting import ForwardBackwardSplitting
from utils.visualization import plot_convergence, plot_performance_comparison

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class AlgorithmComparison:
    """
    算法比较类
    
    用于比较多个优化算法在相同问题上的性能
    """
    
    def __init__(self):
        """初始化比较框架"""
        self.algorithms = {}
        self.results = {}
    
    def add_algorithm(self, name: str, algorithm_class, **kwargs):
        """
        添加要比较的算法
        
        参数:
            name: 算法名称
            algorithm_class: 算法类
            **kwargs: 算法初始化参数
        """
        self.algorithms[name] = (algorithm_class, kwargs)
    
    def compare_algorithms(self, problem, x0, max_iter=1000, tol=1e-6, 
                          verbose=False, timeout=60):
        """
        比较所有添加的算法
        
        参数:
            problem: 优化问题实例
            x0: 初始点
            max_iter: 最大迭代次数
            tol: 收敛容差
            verbose: 是否输出详细信息
            timeout: 超时时间（秒）
            
        返回:
            包含所有算法结果的字典
        """
        results = {}
        
        print(f"开始算法比较，问题维度: {len(x0)}")
        print("=" * 60)
        
        for name, (algorithm_class, kwargs) in self.algorithms.items():
            print(f"\n运行 {name}...")
            
            try:
                # 创建算法实例
                algorithm = algorithm_class(problem, **kwargs)
                
                # 计时运行
                start_time = time.time()
                result = algorithm.solve(x0, max_iter=max_iter, tol=tol, verbose=verbose)
                elapsed_time = time.time() - start_time
                
                # 检查超时
                if elapsed_time > timeout:
                    print(f"  {name} 超时 ({elapsed_time:.1f}s > {timeout}s)")
                    results[name] = {'error': 'timeout', 'time': elapsed_time}
                    continue
                
                # 存储结果
                results[name] = result
                
                # 输出摘要
                print(f"  迭代次数: {result['n_iter']}")
                print(f"  最优值: {result['f_opt']:.6e}")
                print(f"  运行时间: {result['time']:.3f}秒")
                print(f"  收敛: {'是' if result['converged'] else '否'}")
                
            except Exception as e:
                print(f"  {name} 运行失败: {str(e)}")
                results[name] = {'error': str(e)}
        
        self.results = results
        print("\n" + "=" * 60)
        print("算法比较完成!")
        
        return results
    
    def compare_on_default_problems(self, x0_dim=100, max_iter=1000, verbose=False):
        """
        在默认的测试问题上比较算法
        
        参数:
            x0_dim: 初始点维度
            max_iter: 最大迭代次数
            verbose: 是否输出详细信息
            
        返回:
            比较结果字典
        """
        from problems.regularized import L1RegularizedProblem, L2RegularizedProblem
        
        # 生成测试数据
        np.random.seed(42)
        m, n = max(50, x0_dim//2), x0_dim
        A = np.random.randn(m, n)
        x_true = np.random.randn(n)
        x_true[np.random.rand(n) < 0.7] = 0  # 稀疏化
        b = A @ x_true + 0.1 * np.random.randn(m)
        x0 = np.random.randn(n)
        
        # 定义测试问题
        problems = {
            'L1正则化 (λ=0.1)': L1RegularizedProblem(A, b, 0.1),
            'L1正则化 (λ=0.01)': L1RegularizedProblem(A, b, 0.01),
            'L2正则化 (λ=0.1)': L2RegularizedProblem(A, b, 0.1)
        }
        
        all_results = {}
        
        for prob_name, problem in problems.items():
            print(f"\n{'='*20} {prob_name} {'='*20}")
            
            # 设置默认算法
            self._setup_default_algorithms(problem)
            
            # 运行比较
            results = self.compare_algorithms(problem, x0, max_iter, verbose=verbose)
            all_results[prob_name] = results
        
        return all_results
    
    def _setup_default_algorithms(self, problem):
        """设置默认的算法集合"""
        self.algorithms = {}
        
        # 次梯度法
        self.add_algorithm('次梯度法', SubgradientMethod, 
                          step_size='diminishing', step_size_param=1.0)
        
        # 近端梯度法（如果问题支持）
        if hasattr(problem, 'gradient_smooth') and hasattr(problem, 'proximal_operator'):
            self.add_algorithm('近端梯度法', ProximalGradientMethod,
                              step_size='constant', step_size_param=0.01)
            self.add_algorithm('FISTA', FISTA, 
                              step_size='constant', step_size_param=0.01)
            self.add_algorithm('前向-后向分裂', ForwardBackwardSplitting)
        
        # 束方法
        self.add_algorithm('束方法', BundleMethod, nu=0.1, m=1e-3)
        
        # 切割平面法
        self.add_algorithm('Kelley方法', KelleyMethod)
    
    def plot_convergence_comparison(self, results=None, title="算法收敛比较"):
        """
        绘制收敛曲线比较图
        
        参数:
            results: 结果字典，默认使用最近的比较结果
            title: 图表标题
        """
        if results is None:
            results = self.results
        
        if not results:
            print("没有可用的结果进行绘制")
            return
        
        # 提取收敛历史
        histories = {}
        for name, result in results.items():
            if 'error' not in result and 'history' in result:
                histories[name] = result['history']
        
        if histories:
            plot_convergence(histories, title=title)
        else:
            print("没有可用的收敛历史数据")
    
    def plot_performance_comparison(self, results=None, 
                                  metrics=['目标函数值', '迭代次数', '运行时间']):
        """
        绘制性能比较图
        
        参数:
            results: 结果字典，默认使用最近的比较结果
            metrics: 要比较的指标
        """
        if results is None:
            results = self.results
        
        if not results:
            print("没有可用的结果进行绘制")
            return
        
        # 过滤掉有错误的结果
        filtered_results = {name: result for name, result in results.items() 
                          if 'error' not in result}
        
        if filtered_results:
            plot_performance_comparison(filtered_results, metrics)
        else:
            print("没有可用的性能数据")
    
    def generate_report(self, results=None, save_path=None):
        """
        生成详细的比较报告
        
        参数:
            results: 结果字典，默认使用最近的比较结果
            save_path: 报告保存路径
        """
        if results is None:
            results = self.results
        
        if not results:
            print("没有可用的结果生成报告")
            return
        
        report = []
        report.append("非光滑凸优化算法比较报告")
        report.append("=" * 50)
        report.append("")
        
        # 成功运行的算法统计
        successful = [name for name, result in results.items() if 'error' not in result]
        failed = [name for name, result in results.items() if 'error' in result]
        
        report.append(f"成功运行的算法: {len(successful)}")
        report.append(f"失败的算法: {len(failed)}")
        report.append("")
        
        if successful:
            # 性能统计表
            report.append("性能统计:")
            report.append("-" * 30)
            report.append(f"{'算法名':<15} {'迭代次数':<8} {'最优值':<12} {'运行时间(s)':<12} {'收敛':<6}")
            report.append("-" * 60)
            
            for name in successful:
                result = results[name]
                convergence = "是" if result.get('converged', False) else "否"
                report.append(f"{name:<15} {result['n_iter']:<8} {result['f_opt']:<12.3e} "
                            f"{result['time']:<12.3f} {convergence:<6}")
            
            report.append("")
            
            # 找出最佳算法
            best_by_value = min(successful, key=lambda x: results[x]['f_opt'])
            best_by_time = min(successful, key=lambda x: results[x]['time'])
            best_by_iter = min(successful, key=lambda x: results[x]['n_iter'])
            
            report.append("最佳性能:")
            report.append(f"  最优值: {best_by_value} ({results[best_by_value]['f_opt']:.3e})")
            report.append(f"  运行时间: {best_by_time} ({results[best_by_time]['time']:.3f}s)")
            report.append(f"  迭代次数: {best_by_iter} ({results[best_by_iter]['n_iter']})")
            report.append("")
        
        if failed:
            report.append("失败的算法:")
            for name in failed:
                error = results[name].get('error', '未知错误')
                report.append(f"  {name}: {error}")
            report.append("")
        
        # 算法特性总结
        report.append("算法特性总结:")
        report.append("-" * 20)
        report.append("• 次梯度法: 简单实现，收敛慢 O(1/√k)")
        report.append("• 近端梯度法: 平衡的选择，收敛快 O(1/k)")
        report.append("• FISTA: 加速版本，收敛最快 O(1/k²)")
        report.append("• 束方法: 高精度，但内存消耗大")
        report.append("• 切割平面法: 中等性能，适合理论分析")
        report.append("• 分裂方法: 高可扩展性，适合大规模问题")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"报告已保存到: {save_path}")
        else:
            print(report_text)
        
        return report_text

def quick_comparison(problem, x0, algorithms=None, max_iter=1000, verbose=False):
    """
    快速比较函数
    
    参数:
        problem: 优化问题
        x0: 初始点
        algorithms: 要比较的算法列表，None表示使用默认算法
        max_iter: 最大迭代次数
        verbose: 是否详细输出
        
    返回:
        比较结果
    """
    comparison = AlgorithmComparison()
    
    if algorithms is None:
        comparison._setup_default_algorithms(problem)
    else:
        for name, (alg_class, kwargs) in algorithms.items():
            comparison.add_algorithm(name, alg_class, **kwargs)
    
    results = comparison.compare_algorithms(problem, x0, max_iter, verbose=verbose)
    
    # 自动生成可视化
    comparison.plot_convergence_comparison(results)
    comparison.plot_performance_comparison(results)
    
    return results 