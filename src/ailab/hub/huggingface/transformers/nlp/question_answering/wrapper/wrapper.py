#!/usr/bin/env python
# coding:utf-8
"""
@license: Apache License2
@Author: xiaohan4
@time: 2023/03/18
@project: ailab
"""

import json
# from ifly_atp_sdk.huggingface.pipelines import pipeline
from transformers import pipeline

from aiges.core.types import *

try:
    from aiges_embed import ResponseData, Response, DataListNode, DataListCls  # c++
except:
    from aiges.dto import Response, ResponseData, DataListNode, DataListCls

from aiges.sdk import WrapperBase, \
    JsonBodyField, StringBodyField, \
    StringParamField
from aiges.utils.log import log, getFileLogger

# 使用的模型
model = "distilbert-base-cased-distilled-squad"
task = "question-answering"
input1_key = "question"
input2_key = "context"


# 定义模型的超参数和输入参数
class UserRequest(object):
    input1 = StringBodyField(key=input1_key, value="What is the most expensive object in this world?".encode("utf-8"))
    input2 = StringBodyField(key=input2_key, value="Life is precious, love is more precious.".encode("utf-8"))


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
        question = reqData.get(input1_key).data.decode("utf-8")
        context = reqData.get(input2_key).data.decode("utf-8")
        result = self.pipe(question, context)
        self.filelogger.info("result: %s" % result)

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
    m = Wrapper()
    m.run()
