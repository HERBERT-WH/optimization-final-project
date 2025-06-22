"""
全面算法比较示例
演示如何使用算法比较框架
"""
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comparison import AlgorithmComparison, quick_comparison
from problems.regularized import L1RegularizedProblem, L2RegularizedProblem

def comprehensive_comparison():
    """全面算法比较"""
    print("非光滑凸优化算法全面比较")
    print("=" * 60)
    
    # 创建比较实例
    comparison = AlgorithmComparison()
    
    # 在默认问题上进行比较
    results = comparison.compare_on_default_problems(x0_dim=50, max_iter=500, verbose=False)
    
    # 生成报告
    comparison.generate_report()
    
    return results

def custom_problem_comparison():
    """自定义问题比较"""
    print("\n自定义问题算法比较")
    print("=" * 60)
    
    # 生成自定义测试数据
    np.random.seed(123)
    m, n = 40, 80
    A = np.random.randn(m, n) / np.sqrt(m)
    x_true = np.random.randn(n)
    x_true[np.random.rand(n) < 0.8] = 0  # 80%稀疏
    b = A @ x_true + 0.05 * np.random.randn(m)
    x0 = np.random.randn(n)
    
    # 定义问题
    problem = L1RegularizedProblem(A, b, lam=0.05)
    
    print(f"问题规模: {m} x {n}")
    print(f"真实解稀疏度: {np.sum(x_true == 0) / n * 100:.1f}%")
    print(f"初始目标函数值: {problem.objective(x0):.3e}")
    
    # 快速比较
    results = quick_comparison(problem, x0, max_iter=300, verbose=False)
    
    return results

def main():
    """主函数"""
    try:
        # 运行全面比较
        results1 = comprehensive_comparison()
        
        # 运行自定义问题比较  
        results2 = custom_problem_comparison()
        
        print("\n" + "=" * 60)
        print("所有比较完成！")
        
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 