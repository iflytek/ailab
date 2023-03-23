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
import json
from enum import Enum, unique

import giteapy
import os
from giteapy.rest import ApiException
from pprint import pprint
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, as_completed
import markdown2

user_repo_pattern = re.compile("[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+")
ADMIN = "administrator"
DEFAULT_WORKERS = 8


# 捕获异常装饰器
def catch_err(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ApiException as e:
            print(f"Exception when calling AdminApi->{func.__name__}: %s\n" % e)
            return None

    return wrapper


#
@unique
class RepoType(Enum):
    Model = 0
    DataSet = 1
    Demo = 2
    Other = 3


class Repo():
    def __init__(self, task, owner, name, origin_name, source, local_path=""):
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
        self.local_path = local_path
        if not os.path.isdir(local_path):
            raise NotADirectoryError("please check this repo %" % self.origin_name)

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
    def create_user(self, user_name, email):
        us = giteapy.CreateUserOption(username=user_name, email=email, must_change_password=True,
                                      password=initial_pwd, )
        api_response = self.admin_api.admin_create_user(body=us)
        pprint(api_response)

    @catch_err
    def create_repo(self, owner, repo_name, description, private=True):
        repo = giteapy.CreateRepoOption(name=repo_name, description=description, private=private)
        # Create an organization
        api_response = self.admin_api.admin_create_repo(owner, repo)
        pprint(api_response)
        return api_response

    @catch_err
    def create_org(self, org_name, description=""):
        repo = giteapy.CreateOrgOption(full_name=org_name, username=self.username, description=description)

        user = self.user_api.user_get_current()
        print(user)

        # Create an organization
        api_response = self.admin_api.admin_create_org(self.username, repo)
        pprint(api_response)

    @catch_err
    def update_repo_meta(self, owner, repo, description=""):
        repo_option = giteapy.EditRepoOption(description=description, name=repo)
        api_response = self.repo_api.repo_edit(owner=owner, repo=repo, body=repo_option)
        pprint(api_response)

    @catch_err
    def add_repo_user(self, owner, collaborator, repo):
        body = giteapy.AddCollaboratorOption()
        api_response = self.repo_api.repo_add_collaborator(owner, repo, collaborator)
        pprint(api_response)

    @catch_err
    def push_repo(self):
        self.repo_api.repo_add_collaborator()

    @catch_err
    def update_topic(self, owner, repo, topics=None):
        option = giteapy.RepoTopicOptions(topics=topics)
        api_response = self.repo_api.repo_update_topics(owner=owner, repo=repo, body=option)
        pprint(api_response)
        return api_response

    @catch_err
    def get_topic(self, owner, repo):
        api_response = self.repo_api.repo_list_topics(owner=owner, repo=repo)
        pprint(api_response)
        return api_response

    def check_get_user(self, user) -> (giteapy.User, bool):
        try:
            user = self.user_api.user_get(user)
        except ApiException as e:
            if e.status == 404:
                return None, False

        return user, True

    def check_get_repo(self, owner, repo) -> (giteapy.Repository, bool):
        try:
            repo = self.repo_api.repo_get(owner, repo)
        except ApiException as e:
            if e.status == 404:
                return None, False
            elif e.status == 409:
                return repo, True

        return repo, True


class ModelsDirectoryIter:
    """
    模型仓库根目录
    /mnt/atpdata/models_hub/huggingface
    """

    def __init__(self, git: AILABGit = None, path="/mnt/atpdata/models_hub/huggingface", source="huggingface"):
        self.root = path
        self.repos = []
        self.source = source
        self.git = git
        self.init_repos()
        self.executor = ThreadPoolExecutor(max_workers=DEFAULT_WORKERS)
        self.jobs = []

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
                    self.add2repos(task, self.source, repo, self.source, repo, level2_dir)
                else:
                    for rp in fi_or_dirs2:
                        level3_dir = f"{self.root}/{task}/{u}/{rp}"
                        fi_or_dirs3 = os.listdir(level3_dir)
                        if "config.json" in fi_or_dirs3:
                            user = u
                            repo = rp
                            # 有用户级
                            self.add2repos(task, user, repo, self.source, f"{user}/{repo}", level3_dir)

    def add2repos(self, task, owner, name, source="huggingface", origin_name="", local_path=""):
        r = Repo(task, owner, name, origin_name, source, local_path)
        self.repos.append(r)

    def check_readme_meta(self, path):
        rfile = f"{r.local_path}/README.md"
        if not os.path.exists(rfile):
            return {}
        meta = {}
        with open(rfile, 'rb') as rd:
            html = markdown2.markdown(rd.read().decode("utf-8"), extras=["metadata"])
            print("metadata: ", html.metadata)

        return html.metadata

    def new_repo(self, r: Repo, rt: RepoType):
        # 0. check meta,  license adn so on
        ### todo
        metadata = self.check_readme_meta(r.local_path)

        # 1. get user, repo
        user, repo = "", ""
        if re.match(user_repo_pattern, r.origin_name):
            user, repo = r.origin_name.split('/')
        else:
            user = r.source
            repo = r.origin_name

        if not user or not repo:
            raise Exception("Exception check repo name... %s from %s" % (r.name, r.source))
        # 2. create user if not exists
        _user, exists = self.git.check_get_user(user)
        if not exists:
            self.git.create_user(user, email=f'{user}@iflytek.com')

        # 3. create repo if not exists
        _repo, exists = self.git.check_get_repo(user, repo)

        # topic = {"type": rt.name.lower()}
        description = f"{user}/{repo} is a forked repo from {r.source}."
        if not exists:
            _repo = self.git.create_repo(user, repo, description=description, private=False)
        ssh_url = _repo.ssh_url
        print(ssh_url)

        # 3.1 topics
        topics_res = self.git.get_topic(user, repo)
        topics = topics_res.topics
        if not topics:
            topics = []
        if not rt.name.lower() in topics:
            topics.append(rt.name.lower())
        lic = metadata.get("license", None)
        if lic:
            print(type(lic))
            if isinstance(lic, list):
                lic = lic[0]

            lic = lic.replace(".", "-")
            lic = lic.replace(" ", "-")

            lic_label = f"{lic}"
            if lic_label not in topics:
                topics.append(lic_label)

        if r.task not in topics:
            topics.append(r.task)

        print("updating topics: %s" % (str(topics)))
        self.git.update_topic(user, repo, topics)
        # 4. update repo meta
        description += f" License: {lic}"

        self.git.update_repo_meta(user, repo, description=description)
        # 5. add collabrator to repo
        self.git.add_repo_user(user, ADMIN, repo)

        self.jobs.append(self.executor.submit(initial_push_repo, ssh_url, r))


def initial_push_repo(ssh_url, r: Repo):
    cwd = os.getcwd()
    os.chdir(r.local_path)
    cmds = ["git remote  add ailab %s" % ssh_url,
            "git push -u ailab main"
            ]
    # cmds = ["ls -l"]
    print("executing....")
    for cmd in cmds:
        subprocess.call(cmd, shell=True)
    return "ok"


if __name__ == '__main__':
    initial_pwd = os.environ.get("AILAB_REPO_INIT_PWD")
    token = os.environ.get("GITEA_ADMIN_TOKEN")
    host = os.environ.get("GITEA_API_URL")
    if not token:
        print("please set GITEA_ADMIN_TOKEN")
        exit()

    c = AILABGit(access_token=token, host=host,
                 username=ADMIN)

    # c.create_repo("test5", description="niubi")
    # c.create_org("huggingface")
    print(c.check_get_user(ADMIN))
    b = ModelsDirectoryIter(c)
    for r in b.repos:
        b.new_repo(r, RepoType.Model)

    for future in as_completed(b.jobs):
        data = future.result()
        print("{}".format(data))

    wait(b.jobs, return_when=ALL_COMPLETED)
