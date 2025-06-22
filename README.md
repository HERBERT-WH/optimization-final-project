# 非光滑凸优化算法库

## 算法概览

### 1. 次梯度法 (Subgradient Method)

**特点**: 梯度下降法的直接推广，简单易实现但收敛缓慢

- 适用: 无约束和约束非光滑凸优化问题
- 收敛速度: O(1/√k)
- 变体: 投影次梯度法、镜像下降法、自适应次梯度法

**关键代码示例**:

```python
from algorithms.subgradient import SubgradientMethod

solver = SubgradientMethod(problem, step_size='diminishing', step_size_param=1.0)
result = solver.solve(x0, max_iter=1000)
```

### 2. 近端梯度法 (Proximal Gradient Method)

**特点**: 专为复合问题设计，平衡了效率和实现复杂度

- 适用: f(x) + g(x) 形式的复合问题
- 收敛速度: O(1/k)
- 变体: FISTA加速算法、加速近端梯度法

**关键代码示例**:

```python
from algorithms.proximal_gradient import FISTA

solver = FISTA(problem, step_size='backtracking')
result = solver.solve(x0, max_iter=1000)
```

### 3. 束方法 (Bundle Methods)

**特点**: 利用历史信息构建精确下界模型，高精度但计算成本高

- 适用: 需要高精度解的问题
- 收敛速度: 线性收敛
- 变体: 近端束方法

**关键代码示例**:

```python
from algorithms.bundle_method import BundleMethod

solver = BundleMethod(problem, nu=0.1, m=1e-3)
result = solver.solve(x0, max_iter=100)
```

### 4. 切割平面法 (Cutting Plane Methods)

**特点**: 通过线性不等式迭代细化模型

- 适用: 凸函数优化问题
- 收敛速度: 线性收敛
- 变体: Kelley方法、稳定化切割平面法、信赖域切割平面法

**关键代码示例**:

```python
from algorithms.cutting_plane import KelleyMethod

solver = KelleyMethod(problem)
result = solver.solve(x0, max_iter=100)
```

### 5. 分裂方法 (Splitting Methods)

**特点**: 将复杂问题分解为可并行处理的简单子问题

- 适用: 大规模分布式优化
- 收敛速度: O(1/k)
- 变体: Douglas-Rachford、ADMM、前向-后向分裂

**关键代码示例**:

```python
from algorithms.splitting import ForwardBackwardSplitting

solver = ForwardBackwardSplitting(problem)
result = solver.solve(x0, max_iter=1000)
```

## 算法特性对比

| 算法       | 收敛速度 | 内存需求 | 计算复杂度 | 适用规模 | 实现难度 |
| ---------- | -------- | -------- | ---------- | -------- | -------- |
| 次梯度法   | O(1/√k) | 低       | 低         | 大       | 简单     |
| 近端梯度法 | O(1/k)   | 低       | 中         | 大       | 中等     |
| 束方法     | 线性     | 高       | 高         | 中       | 复杂     |
| 切割平面法 | 线性     | 中       | 中         | 中       | 中等     |
| 分裂方法   | O(1/k)   | 低       | 低         | 极大     | 中等     |

## 支持的问题类型

### 正则化问题

- **L1正则化问题** (Lasso): min 1/2 ||Ax - b||² + λ||x||₁
- **L2正则化问题** (Ridge): min 1/2 ||Ax - b||² + λ/2 ||x||²
- **弹性网络问题**: min 1/2 ||Ax - b||² + λ₁||x||₁ + λ₂/2 ||x||²
- **L1正则化逻辑回归**: min Σlog(1 + exp(-yᵢ xᵢᵀ β)) + λ||β||₁

### 工具函数

**投影算子**:

- 箱约束投影、单纯形投影、L1/L2球投影、正象限投影

**近端算子**:

- L1范数近端算子、L2范数近端算子、弹性网络近端算子、组Lasso近端算子

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 基本用法

```python
from algorithms.subgradient import SubgradientMethod
from problems.regularized import L1RegularizedProblem
import numpy as np

# 定义问题
problem = L1RegularizedProblem(A=np.random.randn(50, 100), 
                               b=np.random.randn(50), 
                               lam=0.1)

# 求解
solver = SubgradientMethod(problem)
result = solver.solve(x0=np.zeros(100), max_iter=1000)

print(f"最优解: {result['x']}")
print(f"最优值: {result['f_opt']}")
```

### 完整示例

```python
import numpy as np
from problems.regularized import L1RegularizedProblem
from algorithms.proximal_gradient import FISTA

# 生成数据
A = np.random.randn(50, 100)
b = np.random.randn(50)
x0 = np.zeros(100)

# 定义问题
problem = L1RegularizedProblem(A, b, lam=0.1)

# 求解
solver = FISTA(problem)
result = solver.solve(x0, max_iter=1000)

print(f"最优解: {result['x']}")
print(f"最优值: {result['f_opt']}")
print(f"迭代次数: {result['n_iter']}")
print(f"运行时间: {result['time']:.3f}秒")
```

## 可视化功能

项目提供了丰富的可视化工具：

- **收敛曲线**: `plot_convergence()` - 比较不同算法的收敛速度
- **2D优化轨迹**: `plot_trajectory_2d()` - 显示优化路径和等高线
- **性能比较**: `plot_performance_comparison()` - 柱状图比较各项指标
- **收敛动画**: `animate_convergence()` - 动态展示优化过程
- **稀疏模式**: `plot_sparsity_pattern()` - 显示解的稀疏结构

```python
from utils.visualization import plot_convergence, plot_sparsity_pattern

# 绘制收敛曲线
histories = {'FISTA': result['history']}
plot_convergence(histories, title="FISTA收敛曲线")

# 显示解的稀疏模式
plot_sparsity_pattern(result['x'], title="L1正则化解的稀疏模式")
```

## 核心特性

- ✅ **统一接口**: 所有算法都实现相同的 `solve()`方法接口
- ✅ **丰富的步长策略**: 支持固定、递减、自适应、回溯等多种步长
- ✅ **完整的历史记录**: 记录目标函数值、迭代点、收敛信息
- ✅ **算法比较框架**: 自动化的多算法性能比较
- ✅ **可视化支持**: 内置多种图表和动画功能
- ✅ **中文文档和注释**: 所有输出和文档都支持中文

## 示例文件

- `examples/basic_usage.py`: 基本用法演示
- `examples/algorithm_comparison.py`: 全面算法比较示例

运行示例：

```bash
python examples/basic_usage.py
python examples/algorithm_comparison.py
```
