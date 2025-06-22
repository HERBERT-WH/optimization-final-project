"""
优化问题定义包

包含各种类型的非光滑凸优化问题:
- 基础问题类
- 正则化问题
- 约束问题
"""

from .base import OptimizationProblem
from .regularized import L1RegularizedProblem, L2RegularizedProblem, ElasticNetProblem
from .constrained import BoxConstrainedProblem, SimplexConstrainedProblem

__all__ = [
    'OptimizationProblem',
    'L1RegularizedProblem',
    'L2RegularizedProblem', 
    'ElasticNetProblem',
    'BoxConstrainedProblem',
    'SimplexConstrainedProblem'
] 