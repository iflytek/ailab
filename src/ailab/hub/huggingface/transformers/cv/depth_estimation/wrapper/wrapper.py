#!/usr/bin/env python
# coding:utf-8
"""
@license: Apache License2
@Author: xiaohan4
@time: 2023/03/17
@project: ailab
"""
import json
import os.path

from aiges.core.types import *

try:
    from aiges_embed import ResponseData, Response, DataListNode, DataListCls  # c++
except:
    from aiges.dto import Response, ResponseData, DataListNode, DataListCls

from aiges.sdk import WrapperBase, \
    JsonBodyField, StringBodyField, ImageBodyField, \
    StringParamField
from aiges.utils.log import log, getFileLogger

import io

# from ifly_atp_sdk.huggingface.pipelines import pipeline
from transformers import pipeline
from PIL import Image

# 使用的模型
model = "Intel/dpt-hybrid-midas"
task = "depth-estimation"


# 定义模型的超参数和输入参数
class UserRequest(object):
    input1 = ImageBodyField(key="image", path='./cat.jpg')


# 定义模型的输出参数
class UserResponse(object):
    accept1 = StringBodyField(key="result")
    accept2 = ImageBodyField(key="image")


# 定义服务推理逻辑
class Wrapper(WrapperBase):
    serviceId = task
    version = "v1"
    requestCls = UserRequest()
    responseCls = UserResponse()
    model = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filelogger = None

    def wrapperInit(self, config: {}) -> int:
        log.info("Initializing ...")
        self.pipe = pipeline(task=task, model=model)
        self.filelogger = getFileLogger()
        return 0

    def wrapperOnceExec(self, params: {}, reqData: DataListCls) -> Response:
        self.filelogger.info("got reqdata , %s" % reqData.list)
        input = reqData.get("image").data
        img = Image.open(io.BytesIO(input))
        result = self.pipe(img)
        self.filelogger.info("result: %s" % result)

        predicted_depth = result['predicted_depth'].tolist()
        depth = result['depth']
        # 使用Response封装result
        res = Response()
        resd = ResponseData()
        resd.key = "result"
        resd.setDataType(DataText)
        resd.status = Once
        resd.setData(json.dumps(predicted_depth).encode("utf-8"))

        resd1 = ResponseData()
        resd1.key = "image"
        resd1.setDataType(DataImage)
        resd1.status = Once
        img_bytes = io.BytesIO()
        depth.convert("RGB")
        depth.save(img_bytes, format="jpeg")
        depth.save("./result_depth.jpeg", format="JPEG")
        resd1.setData(img_bytes.getvalue())
        res.list = [resd, resd1]
        return res

    def wrapperFini(cls) -> int:
        return 0

    def wrapperError(cls, ret: int) -> str:
        if ret == 100:
            return "user error defined here"
        return ""

    '''
        此函数保留测试用，不可删除
    '''

    def wrapperTestFunc(cls, data: [], respData: []):
        pass


if __name__ == '__main__':
    m = Wrapper()
    m.run()
