"""
工具函数包

包含:
- 投影算子
- 近端算子
- 可视化工具
"""

from .projections import project_box, project_simplex, project_l2_ball
from .proximal_ops import prox_l1, prox_l2, prox_elastic_net
from .visualization import plot_convergence, plot_comparison

__all__ = [
    'project_box',
    'project_simplex', 
    'project_l2_ball',
    'prox_l1',
    'prox_l2',
    'prox_elastic_net',
    'plot_convergence',
    'plot_comparison'
] 