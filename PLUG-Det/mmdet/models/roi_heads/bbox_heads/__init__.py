# Copyright (c) OpenMMLab. All rights reserved.
from .bbox_head import BBoxHead
from .MIL_bbox_head import Shared2FCInstanceMILHead
from .wsddn_head import WSDDNHead
from .oicr_head import OICRHead
from .convfc_bbox_head import (ConvFCBBoxHead, Shared2FCBBoxHead,
                               Shared4Conv1FCBBoxHead)
__all__ = [
    'BBoxHead', 'Shared2FCInstanceMILHead','WSDDNHead','OICRHead','ConvFCBBoxHead','Shared2FCBBoxHead','Shared4Conv1FCBBoxHead'
]
