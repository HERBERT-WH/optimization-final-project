"""
可视化工具

提供用于分析和展示优化算法结果的可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def plot_convergence(histories, labels=None, title="算法收敛比较"):
    """
    绘制收敛曲线
    
    参数:
        histories: 历史记录列表或字典
        labels: 算法标签
        title: 图表标题
    """
    plt.figure(figsize=(10, 6))
    
    if isinstance(histories, dict):
        for label, history in histories.items():
            if 'f_vals' in history:
                plt.semilogy(history['f_vals'], label=label, linewidth=2)
    else:
        if labels is None:
            labels = [f'算法 {i+1}' for i in range(len(histories))]
            
        for i, history in enumerate(histories):
            if 'f_vals' in history:
                plt.semilogy(history['f_vals'], label=labels[i], linewidth=2)
    
    plt.xlabel('迭代次数')
    plt.ylabel('目标函数值 (对数尺度)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_trajectory_2d(histories, problem=None, labels=None, title="优化轨迹"):
    """
    绘制2D优化轨迹
    
    参数:
        histories: 历史记录列表
        problem: 优化问题（用于绘制等高线）
        labels: 算法标签
        title: 图表标题
    """
    plt.figure(figsize=(10, 8))
    
    if labels is None:
        labels = [f'算法 {i+1}' for i in range(len(histories))]
    
    # 绘制等高线（如果问题是2维的）
    if problem is not None and hasattr(problem, 'dim') and problem.dim == 2:
        x_range = []
        y_range = []
        
        # 收集所有点的范围
        for history in histories:
            if 'x_vals' in history:
                x_vals = np.array(history['x_vals'])
                if x_vals.shape[1] == 2:
                    x_range.extend(x_vals[:, 0])
                    y_range.extend(x_vals[:, 1])
        
        if x_range and y_range:
            x_min, x_max = min(x_range), max(x_range)
            y_min, y_max = min(y_range), max(y_range)
            
            # 扩展范围
            x_margin = (x_max - x_min) * 0.2
            y_margin = (y_max - y_min) * 0.2
            
            x = np.linspace(x_min - x_margin, x_max + x_margin, 50)
            y = np.linspace(y_min - y_margin, y_max + y_margin, 50)
            X, Y = np.meshgrid(x, y)
            
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    try:
                        Z[i, j] = problem.objective(np.array([X[i, j], Y[i, j]]))
                    except:
                        Z[i, j] = np.inf
            
            # 处理无穷值
            Z = np.where(np.isfinite(Z), Z, np.max(Z[np.isfinite(Z)]))
            
            plt.contour(X, Y, Z, levels=20, alpha=0.3, colors='gray')
    
    # 绘制轨迹
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    
    for i, (history, color) in enumerate(zip(histories, colors)):
        if 'x_vals' in history:
            x_vals = np.array(history['x_vals'])
            if x_vals.shape[1] >= 2:
                plt.plot(x_vals[:, 0], x_vals[:, 1], 'o-', 
                        color=color, label=labels[i], 
                        markersize=4, linewidth=2, alpha=0.7)
                
                # 标记起点和终点
                plt.plot(x_vals[0, 0], x_vals[0, 1], 's', 
                        color=color, markersize=8, markeredgecolor='black')
                plt.plot(x_vals[-1, 0], x_vals[-1, 1], '*', 
                        color=color, markersize=12, markeredgecolor='black')
    
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def plot_performance_comparison(results, metrics=['目标函数值', '迭代次数', '运行时间']):
    """
    绘制性能比较图
    
    参数:
        results: 结果字典
        metrics: 要比较的指标
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    algorithms = list(results.keys())
    
    for i, metric in enumerate(metrics):
        values = []
        
        for alg in algorithms:
            if metric == '目标函数值' and 'f_opt' in results[alg]:
                values.append(results[alg]['f_opt'])
            elif metric == '迭代次数' and 'n_iter' in results[alg]:
                values.append(results[alg]['n_iter'])
            elif metric == '运行时间' and 'time' in results[alg]:
                values.append(results[alg]['time'])
            else:
                values.append(0)
        
        bars = axes[i].bar(algorithms, values, alpha=0.7)
        axes[i].set_title(f'{metric}比较')
        axes[i].set_ylabel(metric)
        
        # 旋转x轴标签
        axes[i].tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            if metric == '运行时间':
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(values),
                           f'{value:.3f}s', ha='center', va='bottom')
            elif metric == '目标函数值':
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(values),
                           f'{value:.2e}', ha='center', va='bottom')
            else:
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(values),
                           f'{int(value)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def animate_convergence(history, problem=None, title="优化过程动画"):
    """
    创建收敛过程的动画
    
    参数:
        history: 单个算法的历史记录
        problem: 优化问题
        title: 动画标题
    """
    if 'x_vals' not in history or 'f_vals' not in history:
        print("历史记录不完整，无法创建动画")
        return
    
    x_vals = np.array(history['x_vals'])
    f_vals = history['f_vals']
    
    if x_vals.shape[1] != 2:
        print("只支持2维问题的动画")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 设置轨迹图
    x_range = [x_vals[:, 0].min(), x_vals[:, 0].max()]
    y_range = [x_vals[:, 1].min(), x_vals[:, 1].max()]
    
    if problem is not None:
        # 绘制等高线
        x_margin = (x_range[1] - x_range[0]) * 0.2
        y_margin = (y_range[1] - y_range[0]) * 0.2
        
        x = np.linspace(x_range[0] - x_margin, x_range[1] + x_margin, 50)
        y = np.linspace(y_range[0] - y_margin, y_range[1] + y_margin, 50)
        X, Y = np.meshgrid(x, y)
        
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    Z[i, j] = problem.objective(np.array([X[i, j], Y[i, j]]))
                except:
                    Z[i, j] = np.inf
        
        Z = np.where(np.isfinite(Z), Z, np.max(Z[np.isfinite(Z)]))
        ax1.contour(X, Y, Z, levels=20, alpha=0.3, colors='gray')
    
    ax1.set_xlim(x_range[0] - 0.1*(x_range[1] - x_range[0]), 
                 x_range[1] + 0.1*(x_range[1] - x_range[0]))
    ax1.set_ylim(y_range[0] - 0.1*(y_range[1] - y_range[0]), 
                 y_range[1] + 0.1*(y_range[1] - y_range[0]))
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.set_title('优化轨迹')
    
    # 设置收敛图
    ax2.set_xlim(0, len(f_vals))
    ax2.set_ylim(min(f_vals) * 0.9, max(f_vals) * 1.1)
    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('目标函数值')
    ax2.set_title('收敛曲线')
    ax2.set_yscale('log')
    
    # 初始化图形元素
    trajectory_line, = ax1.plot([], [], 'b-', alpha=0.7)
    current_point, = ax1.plot([], [], 'ro', markersize=8)
    convergence_line, = ax2.plot([], [], 'b-', linewidth=2)
    current_f, = ax2.plot([], [], 'ro', markersize=8)
    
    def animate(frame):
        # 更新轨迹
        trajectory_line.set_data(x_vals[:frame+1, 0], x_vals[:frame+1, 1])
        current_point.set_data([x_vals[frame, 0]], [x_vals[frame, 1]])
        
        # 更新收敛曲线
        convergence_line.set_data(range(frame+1), f_vals[:frame+1])
        current_f.set_data([frame], [f_vals[frame]])
        
        ax1.set_title(f'优化轨迹 (迭代 {frame+1})')
        ax2.set_title(f'收敛曲线 (f = {f_vals[frame]:.2e})')
        
        return trajectory_line, current_point, convergence_line, current_f
    
    anim = FuncAnimation(fig, animate, frames=len(x_vals), 
                        interval=200, blit=True, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return anim

def plot_sparsity_pattern(x, title="稀疏模式"):
    """
    绘制解的稀疏模式
    
    参数:
        x: 解向量
        title: 图表标题
    """
    plt.figure(figsize=(10, 2))
    
    # 创建稀疏模式矩阵
    sparse_matrix = np.abs(x).reshape(1, -1)
    
    plt.imshow(sparse_matrix, cmap='Blues', aspect='auto')
    plt.colorbar(label='|x_i|')
    plt.title(title)
    plt.xlabel('变量索引')
    plt.yticks([])
    
    # 标记非零元素
    nonzero_indices = np.where(np.abs(x) > 1e-6)[0]
    plt.scatter(nonzero_indices, np.zeros_like(nonzero_indices), 
               color='red', marker='|', s=100, alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    print(f"非零元素个数: {len(nonzero_indices)} / {len(x)}")
    print(f"稀疏度: {(1 - len(nonzero_indices) / len(x)) * 100:.1f}%")

def plot_step_sizes(history: Dict, 
                   figsize: Tuple[int, int] = (10, 6),
                   save_path: Optional[str] = None):
    """
    绘制步长变化曲线
    
    Args:
        history: 算法历史记录
        figsize: 图像大小
        save_path: 保存路径
    """
    if 'step_sizes' not in history:
        print("历史记录中没有step_sizes信息")
        return
    
    plt.figure(figsize=figsize)
    
    step_sizes = history['step_sizes']
    iterations = np.arange(len(step_sizes))
    
    plt.plot(iterations, step_sizes, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('步长', fontsize=12)
    plt.title('步长变化曲线', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_subgradient_norms(history: Dict, 
                          figsize: Tuple[int, int] = (10, 6),
                          save_path: Optional[str] = None):
    """
    绘制次梯度范数变化曲线
    
    Args:
        history: 算法历史记录
        figsize: 图像大小
        save_path: 保存路径
    """
    norm_key = None
    for key in ['subgradient_norms', 'gradient_norms']:
        if key in history:
            norm_key = key
            break
    
    if norm_key is None:
        print("历史记录中没有梯度/次梯度范数信息")
        return
    
    plt.figure(figsize=figsize)
    
    norms = history[norm_key]
    iterations = np.arange(len(norms))
    
    plt.semilogy(iterations, norms, 'r-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('次梯度范数 (对数尺度)', fontsize=12)
    plt.title('次梯度范数变化曲线', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_algorithm_characteristics(algorithms_info: Dict[str, Dict],
                                  figsize: Tuple[int, int] = (10, 8),
                                  save_path: Optional[str] = None):
    """
    绘制算法特性雷达图
    
    Args:
        algorithms_info: 算法信息字典
        figsize: 图像大小
        save_path: 保存路径
    """
    # 特性维度
    characteristics = ['收敛速度', '内存需求', '计算复杂度', '适用规模', '实现难度']
    
    # 创建雷达图
    angles = np.linspace(0, 2 * np.pi, len(characteristics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # 闭合
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms_info)))
    
    for i, (alg_name, info) in enumerate(algorithms_info.items()):
        values = [info.get(char, 3) for char in characteristics]  # 默认值3
        values += [values[0]]  # 闭合
        
        ax.plot(angles, values, 'o-', linewidth=2, 
               label=alg_name, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(characteristics)
    ax.set_ylim(0, 5)
    ax.set_title('算法特性比较', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show() 