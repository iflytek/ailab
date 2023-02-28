# please visit https://github.com/xfyun/aiges/releases to get stable and suitable iamges.

FROM public.ecr.aws/iflytek-open/aiges-gpu:11.6-1.17-3.9.13-ubuntu1804-v2.0.0-rc6

#  do this if you are on the chinese server.

RUN pip3 config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple/

ADD requirements.txt /home/aiges

RUN pip install -r /home/aiges/requirements.txt

ADD wrapper /home/aiges

# companion.xfyun.iflytek:6868 is the default config center. It depends on how your athena serving framework built
CMD ["sh", "-c", "./AIservice -m=0 -c=aiges.toml -s=svcName -u=http://companion.xfyun.iflytek:6868 -p=AIaaS -g=dx"]