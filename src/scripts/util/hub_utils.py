#!/usr/bin/env python
# coding:utf-8
""" 
@author: nivic ybyang7
@license: Apache Licence 
@file: models_list
@time: 2023/03/23
@contact: ybyang7@iflytek.com
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

#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.


from huggingface_hub import list_datasets, DatasetFilter, list_models, HfApi, DatasetSearchArguments, \
    ModelSearchArguments, ModelFilter


# args = DatasetSearchArguments()

# args = ModelSearchArguments()

def list_datasets_by(task, top=10):
    f = DatasetFilter(task_categories=task)
    dss = list_datasets(filter=f)
    siz = len(dss)
    print("this task: %s, has %d datasets" % (task, siz))
    dss.sort(key=lambda element: element.downloads, reverse=True)
    return dss[:top]


def list_models_by_task(task, top=10, lib="transformers"):
    filter = ModelFilter(task=task, library=lib)
    models = list_models(filter=filter)
    siz = len(models)
    print("this task: %s, library:%s , has %d models" % (task, lib, siz))
    models.sort(key=lambda element: element.downloads, reverse=True)
    return models[:top]


if __name__ == '__main__':
    list_models_by_task('image-classification')
