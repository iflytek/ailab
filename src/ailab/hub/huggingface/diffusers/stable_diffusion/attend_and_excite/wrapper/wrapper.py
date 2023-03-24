#!/usr/bin/env python
# coding:utf-8
"""
@license: Apache License2
@Author: glchen
@time: 2023/03/23
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
from diffusers import StableDiffusionAttendAndExcitePipeline
from PIL import Image
import torch

# 使用的模型
model = "CompVis/stable-diffusion-v1-4"
task = "stable-diffusion-attend-and-excite"
prompt = "a cat and a frog"
token_indices = [2, 5]
seed = 6141



# 定义模型的超参数和输入参数
class UserRequest(object):
    input1 = StringBodyField(key="text", value=prompt.encode("utf-8"))
    input2 = StringParamField(key="task", value=task)


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
        self.pipe = StableDiffusionAttendAndExcitePipeline.from_pretrained(model, torch_dtype=torch.float16).to("cuda")
        self.pipe.get_indices(prompt)
        self.filelogger = getFileLogger()
        return 0

    def wrapperOnceExec(self, params: {}, reqData: DataListCls) -> Response:
        # 读取文字并进行模型推理
        self.filelogger.info("got reqdata , %s" % reqData.list)
        input_text = reqData.get("text").data.decode("utf-8")
        generator = torch.Generator("cuda").manual_seed(seed)
        result = self.pipe(prompt=prompt,
                           token_indices=token_indices,
                           guidance_scale=7.5,
                           generator=generator,
                           num_inference_steps=50,
                           max_iter_to_alter=25).images[0]
        self.filelogger.info("result: %s" % result)

        # predicted_depth = result['predicted_depth'].tolist()
        # depth = result['depth']

        # 使用Response封装result
        res = Response()
        resd = ResponseData()
        resd.key = "image"
        resd.setDataType(DataImage)
        resd.status = Once
        img_bytes = io.BytesIO()
        result.convert("RGB")
        result.save(img_bytes, format="png")
        result.save(f"./{prompt}_{seed}.png", format="PNG")
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
