#!/usr/bin/env python
# coding:utf-8
"""
@license: Apache License2
@file: wrapper.py
@time: 2023-02-08 17:50:55.464284
@project: chat-gpt-lang-chain
@project: ./
"""

import sys
import hashlib

try:
    from aiges_embed import ResponseData, Response, DataListNode, DataListCls, SessionCreateResponse, callback  # c++
except:
    from aiges.dto import Response, ResponseData, DataListNode, DataListCls, SessionCreateResponse, callback

from aiges.stream import StreamHandleThread
from aiges.sdk import WrapperBase, \
    StringParamField, \
    ImageBodyField, \
    StringBodyField
from aiges.utils.log import log
from aiges.core.types import *

########
# 请在此区域导入您的依赖库

# Todo
# for example: import numpy
from chat import ChatWrapper, set_openai_api_key

'''
定义请求类:
 params:  params 开头的属性代表最终HTTP协议中的功能参数parameters部分，
          params Field支持 StringParamField，
          NumberParamField，BooleanParamField,IntegerParamField，每个字段均支持枚举
          params 属性多用于协议中的控制字段，请求body字段不属于params范畴

 input:    input字段多用与请求数据段，即body部分，当前支持 ImageBodyField, StringBodyField, 和AudioBodyField
'''


class UserRequest(object):
    # StringParamField多用于控制参数
    # 指明 enums, maxLength, required有助于自动根据要求配置协议schema
    params1 = StringParamField(key="p1", enums=["3", "eee"], value='3')
    params2 = StringParamField(key="p2", maxLength=44, required=True)
    params3 = StringParamField(key="p3", maxLength=44, required=False)

    # imagebodyfield 指明path，有助于本地调试wrapper.py
    input1 = ImageBodyField(key="data", path="test_data/test.png")
    input3 = ImageBodyField(key="data2", path="test_data/test.png")
    # stringbodyfiled 指明 value，用于本地调试时的测试值
    input2 = StringBodyField(key="switch", value="ctrl")


'''
定义响应类:
 accepts:  accepts代表响应中包含哪些字段, 以及数据类型

 input:    input字段多用与请求数据段，即body部分，当前支持 ImageBodyField, StringBodyField, 和AudioBodyField
'''


class UserResponse(object):
    # 此类定义响应返回数据段，请务必指明对应key
    # 支持 ImageBodyField， AudioBodyField,  StringBodyField
    # 如果响应是json， 请使用StringBodyField
    accept1 = StringBodyField(key="boxes")
    accept2 = StringBodyField(key="boxes2")


'''
用户实现， 名称必须为Wrapper, 必须继承SDK中的 WrapperBase类
'''


class Wrapper(WrapperBase):
    serviceId = "mmocr"
    version = "backup.0"
    requestCls = UserRequest()
    responseCls = UserResponse()

    '''
    服务初始化
    @param config:
        插件初始化需要的一些配置，字典类型
        key: 配置名
        value: 配置的值
    @return
        ret: 错误码。无错误时返回0
    '''

    def wrapperInit(self, config: {}) -> int:
        log.info(config)
        log.info("Initializing ...")
        Wrapper.session_total = config.get("common.lic", 1)
        self.session.init_wrapper_config(config)
        self.session.init_handle_pool("thread", 1, WorkerThread)
        return 0

    '''
    非会话模式计算接口,对应oneShot请求,可能存在并发调用
    @param params 功能参数
    @param  reqData     请求数据实体字段 DataListCls,可通过 aiges.dto.DataListCls查看
    @return 
        响应必须返回 Response类，非Response类将会引起未知错误
    '''

    def wrapperOnceExec(cls, params: {}, reqData: DataListCls) -> Response:
        log.info("got reqdata , %s" % reqData.list)
        #        print(type(reqData.list[0].data))
        #        print(type(reqData.list[0].data))
        #        print(reqData.list[0].len)
        for req in reqData.list:
            log.info("reqData key: %s , size is %d" % (req.key, len(req.data)))
        log.warning("reqData bytes md5sum is %s" % hashlib.md5(reqData.list[0].data).hexdigest())
        log.info("I am infer logic...please inplement")
        log.info("Testing reqData get: ")
        rg = reqData.get("data")
        log.info("get key: %s" % rg.key)
        log.info("get key: %d" % len(rg.data))

        # test not reqdata
        k = "dd"
        n = reqData.get(k)
        if not n:
            log.error("reqData not has this key %s" % k)

        log.warning("reqData bytes md5sum is %s" % hashlib.md5(reqData.list[0].data).hexdigest())
        log.info("I am infer logic...please inplement")
        r = Response()
        # 错误处理
        # return r.response_err(100)
        l = ResponseData()
        l.key = "ccc"
        l.status = 1
        d = open("test_data/test.png", "rb").read()
        l.len = len(d)
        l.data = d
        l.type = 0
        r.list = [l, l, l]
        return r

    '''
    服务逆初始化

    @return
        ret:错误码。无错误码时返回0
    '''

    def wrapperFini(cls) -> int:
        return 0

    '''
    非会话模式计算接口,对应oneShot请求,可能存在并发调用
    @param ret wrapperOnceExec返回的response中的error_code 将会被自动传入本函数并通过http响应返回给最终用户
    @return 
        str 错误提示会返回在接口响应中
    '''

    def wrapperError(cls, ret: int) -> str:
        if ret == 100:
            return "user error defined here"
        return ""

    '''
        此函数保留测试用，不可删除
    '''

    def wrapperTestFunc(cls, data: [], respData: []):
        r = Response()
        l = ResponseData()
        l.key = "ccc"
        l.status = 1
        d = open("pybind11/docs/pybind11-logo.png", "rb").read()
        l.len = len(d)
        l.data = d
        r.list = [l, l, l]
        return r

    def wrapperCreate(self, params: {}, sid: str) -> SessionCreateResponse:
        """
        非会话模式计算接口,对应oneShot请求,可能存在并发调用
        @param ret wrapperOnceExec返回的response中的error_code 将会被自动传入本函数并通过http响应返回给最终用户
        @return
            SessionCreateResponse类, 如果返回不是该类会报错
        """
        sp = SessionCreateResponse()
        # 这里是取 handle
        handle = self.session.get_idle_handle()
        if not handle:
            sp.error_code = -1
            sp.handle = ""
            return sp

        _session = self.session.get_session(handle=handle)
        if _session == None:
            log.info("can't create this handle:" % handle)
            sp.error_code = -1
            sp.handle = ""
            return sp
        _session.setup_sid(sid)
        _session.setup_params(params)
        _session.setup_callback_fn(callback)

        # print(sid)
        s = SessionCreateResponse()
        s.handle = handle
        s.error_code = 0
        return s

    def wrapperWrite(self, handle: str, req: DataListCls, sid: str) -> int:
        """
        会话模式下: 上行数据写入接口
        :param handle: 会话handle 字符串
        :param req:  请求数据结构
        :param sid:  请求会话ID
        :return:
        """
        _session = self.session.get_session(handle=handle)
        if _session == None:
            log.info("can't get this handle:" % handle)
            return -1
        _session.in_q.put(req)
        return 0

    def wrapperRead(self, handle: str, sid: str) -> Response:
        """
        会话模式: 当前此接口在会话模式且异步取结果时下不会被调用！！！！！返回数据由callback返回
        同步取结果模式时，下行数据返回接口
                  如果为异步返回结果时，需要设置加载器为asyncMode=true [当前默认此模式],
        :param handle: 请求数据结构
        :param sid: 请求会话ID
        :return: Response类
        """
        _session = self.session.get_session(handle=handle)
        r = Response()
        l = ResponseData()
        if _session.out_q.empty():
            l.status = 1
            l.len = 0
            l.data = b''
            l.key = "boxes"
            r.list = [l]
            return r
        rs = _session.out_q.get()
        if rs.list[0].status == DataEnd:
            _session.reset()
        if not isinstance(rs, Response):
            raise Exception("check response")

        return rs


class WorkerThread(StreamHandleThread):
    """
    流式示例 thread，
    """

    def __init__(self, session_thread, in_q, out_q):
        super().__init__(session_thread, in_q, out_q)
        self.api_key = "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"

    def init_chat(self, *args, **kwargs):
        self.history_state = []
        self.chain_state, self.express_chain_state, self.llm_state, self.embeddings_state, \
        self.qa_chain_state, self.memory_state = set_openai_api_key(
            self.api_key)

        self.chat = ChatWrapper()

    def run(self):
        self.init_chat(self.session_thread.handle)
        while True:
            req = self.in_q.get()
            # print("#######get####")
            # print(self.session_thread.params)
            self.infer(req)

    def infer(self, req: DataListCls):
        trace_chain_state = False
        speak_text_state = False
        talking_head_state = True
        monologue_state = False
        express_chain_state = None
        num_words_state = 0
        formality_state = "N/A"
        anticipation_level_state = "N/A"
        joy_level_state = "N/A"
        trust_level_state = "N/A"
        fear_level_state = "N/A"
        surprise_level_state = "N/A"
        sadness_level_state = "N/A"
        disgust_level_state = "N/A"
        anger_level_state = "N/A"
        lang_level_state = "N/A"
        translate_to_state = "N/A"
        literary_style_state = "N/A"
        docsearch_state = None
        use_embeddings_state = False
        msg = req.get('message').data.decode('utf-8')
        self.chatbot, self.history_state, video_html, my_file, audio_html, tmp_aud_file, message = self.chat(
            self.api_key, msg,
            self.history_state,
            self.chain_state,
            trace_chain_state,
            speak_text_state,
            talking_head_state,
            monologue_state,
            express_chain_state,
            num_words_state,
            formality_state,
            anticipation_level_state,
            joy_level_state,
            trust_level_state,
            fear_level_state,
            surprise_level_state,
            sadness_level_state,
            disgust_level_state,
            anger_level_state,
            lang_level_state,
            translate_to_state,
            literary_style_state,
            self.qa_chain_state,
            docsearch_state,
            use_embeddings_state)
        pass


if __name__ == '__main__':
    m = Wrapper()
    # m.schema()
    m.run()
