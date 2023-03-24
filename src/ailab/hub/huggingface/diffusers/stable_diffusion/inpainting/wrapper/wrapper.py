#!/usr/bin/env python
# coding:utf-8
"""
@license: Apache License2
@Author: glchen
@time: 2023/02/27
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
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch

# 使用的模型
model = "runwayml/stable-diffusion-inpainting"
task = "stable-diffusion-inpainting"
prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
device = "cuda"

# 定义模型的超参数和输入参数
class UserRequest(object):
    input1 = ImageBodyField(key="init_image", path='./overture-creations-5sI6fQgYIuo.png')
    input2 = ImageBodyField(key="mask_image", path='./overture-creations-5sI6fQgYIuo_mask.png')
    input3 = StringParamField(key="task", value=task)


# 定义模型的输出参数
class UserResponse(object):
    accept1 = JsonBodyField(key="result")


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
        # pipeline(model=model)
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(model, torch_dtype=torch.float16).to("cuda")
        self.filelogger = getFileLogger()
        return 0

    def wrapperOnceExec(self, params: {}, reqData: DataListCls) -> Response:
        # 读取测试图片并进行模型推理
        self.filelogger.info("got reqdata , %s" % reqData.list)
        input = reqData.get("init_image").data
        init_img = Image.open(io.BytesIO(input))
        init_img = init_img.convert("RGB")
        init_img = init_img.resize((512, 512))

        input1 = reqData.get("mask_img").data
        mask_img = Image.open(io.BytesIO(input1))
        mask_img = mask_img.convert("RGB")
        mask_img = mask_img.resize((512, 512))

        result = self.pipe(prompt=prompt, image=init_img, mask_image=mask_img).images[0]
        self.filelogger.info("result: %s" % result)


        # 使用Response封装result
        res = Response()
        resd = ResponseData()
        resd.key = "init_image"
        resd.setDataType(DataImage)
        resd.status = Once
        img_bytes = io.BytesIO()
        result.save(img_bytes, format="png")
        result.save("./cat.png", format="PNG")
        resd.setData(img_bytes.getvalue())
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
