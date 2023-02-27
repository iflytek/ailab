#!/usr/bin/env python
# coding:utf-8
"""
@license: Apache License2
@file: wrapper.py
@time: 2022-08-19 02:05:07.467170
@project: mnist
@project: ./
"""
import json
import os.path

from aiges.core.types import *

try:
    from aiges_embed import ResponseData, Response, DataListNode, DataListCls  # c++
except:
    from aiges.dto import Response, ResponseData, DataListNode, DataListCls

from aiges.sdk import WrapperBase, \
    ImageBodyField, \
    StringBodyField, StringParamField
from aiges.utils.log import log, getFileLogger

# 导入inference.py中的依赖包
import io

from ifly_atp_sdk.huggingface.pipelines import pipeline
# from transformers import pipeline


# 定义模型的超参数和输入参数
class UserRequest(object):
    input1 = StringBodyField(key="text", value=b"I have a problem with my iphone that needs to be resolved asap!!")
    input2 = StringParamField(key="task", value="automatic-speech-recognition")


# 定义模型的输出参数
class UserResponse(object):
    accept1 = StringBodyField(key="result")


# 定义服务推理逻辑
class Wrapper(WrapperBase):
    serviceId = "text_pipeline"
    version = "v1"
    requestCls = UserRequest()
    responseCls = UserResponse()
    model = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filelogger = None

    def wrapperInit(self, config: {}) -> int:
        log.info("Initializing ...")
        self.pipe = pipeline( model="facebook/bart-large-mnli")
        self.filelogger = getFileLogger()
        return 0

    def wrapperOnceExec(self, params: {}, reqData: DataListCls) -> Response:
        # 读取测试图片并进行模型推理
        self.filelogger.info("got reqdata , %s" % reqData.list)
        input_text = reqData.get("text").data.decode('utf-8')
        result = self.pipe(input_text,
                           candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"], )

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
    m = Wrapper()
    m.run()
