#!/usr/bin/env python
# coding:utf-8
""" 
@author: nivic ybyang7
@license: Apache Licence 
@file: download_top_models
@time: 2023/02/24
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
import os.path
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import datasets
import transformers
import subprocess

supported_task = transformers.pipelines.SUPPORTED_TASKS

defautl_download_dir = "/mnt/atpdata/models_hub/huggingface"

huggingface_repo = 'https://huggingface.co/'


def get_task_models():
    models_hub_dict = {}
    for task, spec in supported_task.items():
        default_model_sets = set()
        default_model_spec = spec.get("default")
        if 'model' in default_model_spec:
            for mtype, item in default_model_spec.get('model').items():
                repo_name, revision = item
                default_model_sets.add(item)
        else:
            ms = default_model_spec.values()
            for m in ms:
                if 'model' in m:
                    for mtype, item in m.get('model').items():
                        repo_name, revision = item
                        default_model_sets.add(item)
        models_hub_dict[task] = default_model_sets
    pprint(models_hub_dict)
    return models_hub_dict


def git_clone(task, model_name, revision, local_dir=defautl_download_dir):
    """

    :param model_name:
    :param revision:
    :return:
    """
    repo_url = huggingface_repo + model_name
    local_path = os.path.join(local_dir, task, model_name)
    cmd1 = f"git clone {repo_url}  {local_path}"
    cmd2 = f"git checkout {revision}"

    if os.path.exists(local_path):
        os.chdir(local_path)
        subprocess.call(cmd2, shell=True)
    else:
        subprocess.call(cmd1, shell=True)
    os.chdir(local_path)
    subprocess.call(cmd2, shell=True)
    pass

def download_models():
    tm = get_task_models()
    # 创建一个包含2条线程的线程池
    works = 6
    pool = ThreadPoolExecutor(max_workers=6)

    futures = []
    for task_name, t in tm.items():
        for repo, revision in t:
            print(repo, revision)
            futures.append(pool.submit(git_clone, task_name, repo, revision))

    wait(futures, return_when=ALL_COMPLETED)

def get_datasets():
    ds = datasets.list_datasets(with_details=True)
    print(ds)

    pass

if __name__ == '__main__':
    #download_models()
    get_datasets()