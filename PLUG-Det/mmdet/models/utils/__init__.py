# Copyright (c) OpenMMLab. All rights reserved.
from .brick_wrappers import AdaptiveAvgPool2d, adaptive_avg_pool2d
from .builder import build_linear_layer, build_transformer
from .res_layer import ResLayer, SimplifiedBasicBlock

__all__ = [
    'adaptive_avg_pool2d', 'AdaptiveAvgPool2d', 
    'build_transformer', 'build_linear_layer', 
    'ResLayer', 'SimplifiedBasicBlock',
]
