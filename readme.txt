项目：/root/project

模块替换目录：/root/project/mysite/grp_model，依次替换对应的模型

替换画图程序：导出huatu.py 修改里面的内容，测试通过后，进行文件覆盖

# 修改代码需要重启服务
/root/project/python39/bin/uwsgi --reload uwsgi.ini


# 启动服务
ps -aux|grep uwsgi
发现进程不存在：/root/project/python39/bin/uwsgi --ini uwsgi.ini
