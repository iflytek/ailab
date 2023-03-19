#!/usr/bin/env python
# coding:utf-8
"""
@license: Apache License2
@Author: hanxiao
@time: 2023/03/19
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
model = "gpt2"
task = "text-generation"
input1_key = "text"


# 定义模型的超参数和输入参数
class UserRequest(object):
    input1 = StringBodyField(key=input1_key,
                             value='{"text":"What is the most precious object in the world?", "max_length":30, "num_return_sequences":3}'
                             .encode("utf-8"))


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
        input_text = reqData.get(input1_key).data.decode("utf-8")
        args_dict = json.loads(input_text)
        result = self.pipe(args_dict["text"], max_length=args_dict["max_length"],
                           num_return_sequences=args_dict["num_return_sequences"])
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
