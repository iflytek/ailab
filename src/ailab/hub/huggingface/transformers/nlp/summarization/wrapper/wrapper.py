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
model = "sshleifer/distilbart-cnn-12-6"
task = "summarization"
input1_key = "text"


# 定义模型的超参数和输入参数
class UserRequest(object):
    input1 = StringBodyField(key=input1_key, value="The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, "
                                                   "and the tallest structure in Paris. Its base is square, "
                                                   "measuring 125 metres (410 ft) on each side. "
                                                   "During its construction, "
                                                   "the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, "
                                                   "a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. "
                                                   "It was the first structure to reach a height of 300 metres. "
                                                   "Due to the addition of a broadcasting aerial at the top of the tower in 1957, "
                                                   "it is now taller than the Chrysler Building by 5.2 metres (17 ft). "
                                                   "Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.".encode("utf-8"))


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
        result = self.pipe(input_text)
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
