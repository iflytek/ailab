#!/usr/bin/env python
# coding:utf-8
"""
@license: Apache License2
@author: xiaohan4
@time: 2023/2/28 10:43
@project: ailab
"""

import json
from aiges.core.types import *
# from ifly_atp_sdk.huggingface.pipelines import pipeline
from transformers import pipeline

try:
    from aiges_embed import ResponseData, Response, DataListNode, DataListCls  # c++
except:
    from aiges.dto import Response, ResponseData, DataListNode, DataListCls

from aiges.sdk import WrapperBase, \
    AudioBodyField, \
    StringBodyField, StringParamField
from aiges.utils.log import log, getFileLogger

task = "audio-classification"
model = "superb/wav2vec2-base-superb-ks"
input1_key = "audio"


# 定义模型的超参数和输入参数
class UserRequest(object):
    input1 = AudioBodyField(key=input1_key, path="./mlk.flac")


# 定义模型的输出参数
class UserResponse(object):
    accept1 = StringBodyField(key="result")


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
        input_audio = reqData.get("audio").data
        result = self.pipe(input_audio)

        self.filelogger.info(result)
        # 使用Response封装result
        res = Response()
        resd = ResponseData()
        resd.key = "result"
        resd.setDataType(DataText)
        resd.status = Once
        resd.setData(json.dumps(result).encode("utf-8"))
        res.list = [resd]
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
    m = Wrapper(legacy=False)
    m.run()
    # print(m.schema())
