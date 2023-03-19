#!/usr/bin/env python
# coding:utf-8
""" 
@author: nivic ybyang7
@license: Apache Licence 
@file: models_import
@time: 2023/03/18
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

import giteapy
import os
from giteapy.rest import ApiException
from pprint import pprint
import re

init_tial_pwd = "abcd.1234"


# 捕获异常装饰器
def catch_err(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ApiException as e:
            print(f"Exception when calling AdminApi->{func.__name__}: %s\n" % e)
            return None

    return wrapper


class Repo():
    def __init__(self, task, owner, name, origin_name, source):
        """
        :param task:
        :param owner:
        :param name:
        :param origin_name:  bert-base-cased openai/clip-vit-large-patch14 this two form...
        :param source:
        """

        self.origin_name = origin_name
        self.task = task
        self.owner = owner
        self.name = name
        self.source = source

    def check_name(self, name):
        name_regex = "(?P<user>[a-zA-Z0-9_]+)[/])*(?P<repo>[a-zA-Z0-9_]+)"


class AILABGit():
    def __init__(self, access_token=None, host="http://localhost:3000/api/v1", username=""):
        if not access_token or not username:
            raise Exception("you must specify the access token and username ")
        # Configure API key authorization: AccessToken
        configuration = giteapy.Configuration()
        configuration.api_key['access_token'] = access_token
        configuration.host = host
        self.username = username
        # create an instance of the API class
        cfg = giteapy.ApiClient(configuration)
        self.admin_api = giteapy.AdminApi(cfg)
        self.repo_api = giteapy.RepositoryApi(cfg)
        self.user_api = giteapy.UserApi(cfg)

    @catch_err
    def create_user(self, user_name):
        us = giteapy.CreateUserOption(username=user_name, must_change_password=True, password=init_tial_pwd, )

    @catch_err
    def create_repo(self, repo_name, description, private=True):
        repo = giteapy.CreateRepoOption(name=repo_name, description=description, private=private)
        # Create an organization
        api_response = self.admin_api.admin_create_repo(self.username, repo)
        pprint(api_response)

    @catch_err
    def create_org(self, org_name, description=""):
        repo = giteapy.CreateOrgOption(full_name=org_name, username=self.username, description=description)

        user = self.user_api.user_get_current()
        print(user)

        # Create an organization
        api_response = self.admin_api.admin_create_org(self.username, repo)
        pprint(api_response)

    @catch_err
    def add_repo_user(self, collaborator, repo):
        self.repo_api.repo_add_collaborator()
        self.repo_api.repo

    @catch_err
    def push_repo(self):
        self.repo_api.repo_add_collaborator()

    def check_get_user(self, user) -> (giteapy.User, bool):
        try:
            user = self.user_api.user_get(user)
        except ApiException as e:
            if e.status == 404:
                return None, False

        return user, True


class ModelsDirectoryIter:
    """
    模型仓库根目录
    /mnt/atpdata/models_hub/huggingface
    """

    def __init__(self, path="/mnt/atpdata/models_hub/huggingface", source="huggingface"):
        self.root = path
        self.repos = []
        self.source = source

        self.init_repos()

    def init_repos(self):
        """

        :return:
        """
        tasks = os.listdir(self.root)
        for task in tasks:
            print(f"getting {task}")
            level1_dir = f"{self.root}/{task}"
            fi_or_dirs = os.listdir(level1_dir)
            repo = ""
            user = ""
            for u in fi_or_dirs:
                level2_dir = f"{self.root}/{task}/{u}"
                fi_or_dirs2 = os.listdir(level2_dir)
                if "config.json" in fi_or_dirs2:
                    repo = u
                    # 如果没有用户级，默认huggingface
                    self.new_repo(task, self.source, repo, self.source, repo)
                else:
                    for rp in fi_or_dirs2:
                        level3_dir = f"{self.root}/{task}/{u}/{rp}"
                        fi_or_dirs3 = os.listdir(level3_dir)
                        if "config.json" in fi_or_dirs3:
                            user = u
                            repo = rp
                            # 有用户级
                            self.new_repo(task, user, repo, self.source, f"{user}/{repo}")

    def new_repo(self, task, owner, name, source="huggingface", origin_name=""):
        r = Repo(task, owner, name, origin_name, source)
        self.repos.append(r)


if __name__ == '__main__':
    token = os.environ.get("GITEA_ADMIN_TOKEN")
    host = os.environ.get("GITEA_API_URL")
    if not token:
        print("please set GITEA_ADMIN_TOKEN")
        exit()

    c = AILABGit(access_token=token, host=host,
                 username="administrator")

    # c.create_repo("test5", description="niubi")
    # c.create_org("huggingface")
    print(c.check_get_user("administrator"))
    b = ModelsDirectoryIter()
    print(len(b.repos))
