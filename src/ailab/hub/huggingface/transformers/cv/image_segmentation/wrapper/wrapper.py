#!/usr/bin/env python
# coding:utf-8
"""
@license: Apache License2
@Author: xiaohan4
@time: 2023/03/18
@project: ailab
"""
import base64
import json
import io
# from ifly_atp_sdk.huggingface.pipelines import pipeline
from transformers import pipeline
from PIL import Image

from aiges.core.types import *

try:
    from aiges_embed import ResponseData, Response, DataListNode, DataListCls  # c++
except:
    from aiges.dto import Response, ResponseData, DataListNode, DataListCls
from aiges.sdk import WrapperBase, \
    JsonBodyField, StringBodyField, ImageBodyField, \
    StringParamField
from aiges.utils.log import log, getFileLogger

# 使用的模型
model = "facebook/detr-resnet-50-panoptic"
task = "image-segmentation"
input1_key = "image"


def image_to_str(image_byte):
    base64_image_byte = base64.b64encode(image_byte)
    image_str = base64_image_byte.decode('ascii')  # byte类型转换为str
    return image_str


def str_to_image(str):
    image_bytes = base64.b64decode(str)
    img = Image.open(io.BytesIO(image_bytes))
    img.convert("RGB")
    img.save("./test_str_to_img.jpg", format="JPEG")


# 定义模型的超参数和输入参数
class UserRequest(object):
    input1 = ImageBodyField(key=input1_key, path='./cat.jpg')


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
        input = reqData.get(input1_key).data
        img = Image.open(io.BytesIO(input))
        result = self.pipe(img)
        self.filelogger.info("result: %s" % result)

        i = 0
        for item in result:
            i = i + 1
            img_bytes = io.BytesIO()
            img = item['mask']
            img.convert("RGB")
            img.save(img_bytes, format="JPEG")
            img.save("./result_mask_" + str(i) + ".jpg", format="JPEG")
            item['mask'] = image_to_str(img_bytes.getvalue())

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
