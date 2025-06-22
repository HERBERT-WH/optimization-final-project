"""
非光滑凸优化算法包

包含以下算法的实现:
- 次梯度法 (Subgradient Method)
- 近端梯度法 (Proximal Gradient Method)
- 束方法 (Bundle Methods)
- 切割平面法 (Cutting Plane Methods)
- 分裂方法 (Splitting Methods)
"""

from .subgradient import SubgradientMethod
from .proximal_gradient import ProximalGradientMethod, FISTA
from .bundle_method import BundleMethod
from .cutting_plane import KelleyMethod
from .splitting import DouglasRachford, ADMM

__all__ = [
    'SubgradientMethod',
    'ProximalGradientMethod', 
    'FISTA',
    'BundleMethod',
    'KelleyMethod',
    'DouglasRachford',
    'ADMM'
] 