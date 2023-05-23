'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-04-28 19:28:13
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-06-28 09:57:58
FilePath: /mmdetection-2.22.0/mmdet/models/dense_heads/__init__.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# Copyright (c) OpenMMLab. All rights reserved.
from .anchor_free_head import AnchorFreeHead
from .anchor_head import AnchorHead
from .rpn_head import RPNHead
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin
from .plug_head import PLUGHead
__all__ = [
    'AnchorFreeHead', 'AnchorHead',
    'RPNHead','PLUGHead','BaseDenseHead','BBoxTestMixin'
]
