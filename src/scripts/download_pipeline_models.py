#!/usr/bin/env python
# coding:utf-8
""" 
@author: xiaohan4
@license: Apache Licence 
@file: download_pipeline_models.py
@time: 2023/3/19
@contact: xiaohan4@iflytek.com
@site:  
@software: PyCharm 

# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛ 
"""
import transformers
#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

from transformers.pipelines import pipeline, SUPPORTED_TASKS
from concurrent.futures import ThreadPoolExecutor, wait

# 通过pipeline下载所有task默认模型
def get_default_task_models(workders=8):
    pool = ThreadPoolExecutor(max_workers=workders)
    futures = []

    for task, spec in SUPPORTED_TASKS.items():
        if task.startswith("translation"):
            for lang in spec['default'].keys():
                task = 'translation_' + lang[0] + '_to_' + lang[-1]
                futures.append(pool.submit(pipeline, task))
        else:
            futures.append(pool.submit(pipeline, task))

    done, not_done = wait(futures)
    print("done:", done)
    print("not_done:", not_done)


if __name__ == '__main__':
    get_default_task_models(16)
