"""
基本用法示例

演示如何使用非光滑凸优化算法库
"""

import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms import SubgradientMethod, ProximalGradientMethod, FISTA, BundleMethod
from problems import L1RegularizedProblem, L2RegularizedProblem, ElasticNetProblem
from comparison import AlgorithmComparison
from utils.visualization import plot_convergence, plot_sparsity_pattern

def main():
    print("=" * 60)
    print("非光滑凸优化算法库 - 基本用法示例")
    print("=" * 60)
    
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    
    # 1. 创建测试问题
    print("\n1. 创建L1正则化问题 (Lasso)")
    m, n = 100, 50  # 100个样本，50个特征
    A = np.random.randn(m, n)
    x_true = np.random.randn(n)
    x_true[np.random.choice(n, n//2, replace=False)] = 0  # 稀疏真值
    b = A @ x_true + 0.1 * np.random.randn(m)  # 添加噪声
    
    lambda_reg = 0.1
    problem = L1RegularizedProblem(A, b, lambda_reg)
    
    print(f"问题维度: {problem.dimension}")
    print(f"正则化参数: {lambda_reg}")
    
    # 2. 单独测试算法
    print("\n2. 单独测试次梯度法")
    x0 = np.random.randn(n)
    print(f"初始目标函数值: {problem.objective(x0):.6f}")
    
    # 次梯度法
    subgrad_solver = SubgradientMethod(
        step_size=0.01, 
        max_iter=500, 
        verbose=True,
        tolerance=1e-6
    )
    
    x_subgrad, history_subgrad = subgrad_solver.solve(problem, x0.copy())
    print(f"次梯度法最终目标值: {problem.objective(x_subgrad):.6f}")
    print(f"次梯度法迭代次数: {history_subgrad['iterations']}")
    
    # 3. 测试近端梯度法
    print("\n3. 测试近端梯度法")
    pgm_solver = ProximalGradientMethod(
        step_size=0.01,
        max_iter=500,
        verbose=True,
        tolerance=1e-6
    )
    
    x_pgm, history_pgm = pgm_solver.solve(problem, x0.copy())
    print(f"近端梯度法最终目标值: {problem.objective(x_pgm):.6f}")
    print(f"近端梯度法迭代次数: {history_pgm['iterations']}")
    
    # 4. 测试FISTA
    print("\n4. 测试FISTA加速算法")
    fista_solver = FISTA(
        step_size=0.01,
        max_iter=500,
        verbose=True,
        tolerance=1e-6
    )
    
    x_fista, history_fista = fista_solver.solve(problem, x0.copy())
    print(f"FISTA最终目标值: {problem.objective(x_fista):.6f}")
    print(f"FISTA迭代次数: {history_fista['iterations']}")
    
    # 5. 算法比较
    print("\n5. 算法性能比较")
    comparison = AlgorithmComparison()
    
    # 添加算法到比较
    comparison.add_algorithm('次梯度法', SubgradientMethod(step_size=0.01, max_iter=500, verbose=False))
    comparison.add_algorithm('近端梯度法', ProximalGradientMethod(step_size=0.01, max_iter=500, verbose=False))
    comparison.add_algorithm('FISTA', FISTA(step_size=0.01, max_iter=500, verbose=False))
    
    # 运行比较
    results = comparison.run_comparison(problem, x0.copy())
    
    # 显示结果摘要
    comparison.print_summary()
    
    # 绘制收敛曲线
    print("\n正在生成收敛曲线图...")
    comparison.plot_convergence(figsize=(10, 6), log_scale=True)
    
    # 绘制性能比较图
    print("正在生成性能比较图...")
    comparison.plot_performance_comparison(figsize=(12, 8))
    
    # 6. 不同问题类型测试
    print("\n6. 测试不同问题类型")
    
    # L2正则化问题
    print("L2正则化问题 (Ridge回归):")
    l2_problem = L2RegularizedProblem(A, b, lambda_reg)
    x_l2, _ = pgm_solver.solve(l2_problem, x0.copy())
    print(f"  最终目标值: {l2_problem.objective(x_l2):.6f}")
    
    # Elastic Net问题
    print("Elastic Net问题:")
    elastic_problem = ElasticNetProblem(A, b, lambda1=0.1, lambda2=0.1)
    x_elastic, _ = pgm_solver.solve(elastic_problem, x0.copy())
    print(f"  最终目标值: {elastic_problem.objective(x_elastic):.6f}")
    
    # 7. 解的质量分析
    print("\n7. 解的质量分析")
    print("真实解的稀疏性 (非零元素个数):", np.sum(np.abs(x_true) > 1e-6))
    print("L1解的稀疏性 (非零元素个数):", np.sum(np.abs(x_pgm) > 1e-6))
    print("L2解的稀疏性 (非零元素个数):", np.sum(np.abs(x_l2) > 1e-6))
    print("Elastic Net解的稀疏性 (非零元素个数):", np.sum(np.abs(x_elastic) > 1e-6))
    
    # 计算重构误差
    l1_error = np.linalg.norm(A @ x_pgm - b)
    l2_error = np.linalg.norm(A @ x_l2 - b)
    elastic_error = np.linalg.norm(A @ x_elastic - b)
    
    print(f"\n重构误差 ||Ax - b||:")
    print(f"  L1正则化: {l1_error:.6f}")
    print(f"  L2正则化: {l2_error:.6f}")
    print(f"  Elastic Net: {elastic_error:.6f}")
    
    print("\n=" * 60)
    print("示例完成！")
    print("=" * 60)

if __name__ == "__main__":
    main() 