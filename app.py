import os
import json
import time
import csv

import numpy
import pandas as pd
from resnet import run_model_cls
from huatu import huatu_cls
from flask_cors import CORS
from flask import Flask, render_template, request, send_file, Response

app = Flask(__name__)
base_dir = "/root/project/mysite/"

import logging, traceback

app.logger.setLevel(logging.INFO)

handler = logging.FileHandler(f'{base_dir}data/log/flask.log')
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

app.logger.addHandler(handler)

# 上传文件
@app.route("/resnet/file_upload", methods=['POST'])
def FileUploadCls():
    try:
        res_data = {
            "status": 1
        }
        file = request.files.get('file')
        if file:
            file.save(f"{base_dir}data/files/uploads/" + file.filename)
            res_data["message"] = "success"
            res_data["result"] = {
                "csvName": file.filename
            }
            return res_data
        else:
            res_data["message"] = "fail"
            return res_data
    except:
        app.logger.info(traceback.format_exc())
        return {
            "status": 0,
            "message": "请求错误！"
        }

# 获取数据模型运行结果
@app.route("/resnet/handle_csv", methods=['GET', 'POST'])
def HandleCsvCls():
    try:
        params = request.values
        newFile = params.get("newFile")  # 是否为新文件数据预跑
        frequency = params.get("frequency")
        timeWindow = params.get("timeWindow")
        gap = params.get("gap")
        excelent = params.get("excelent")
        qualified = params.get("qualified")
        insufficient = params.get("insufficient")
        csvName = params.get("csvName")
        jsonName = csvName.replace('.csv', '.json')
        jpgName = csvName.replace('.csv', '.jpg')
        GPR_Csv = f"{base_dir}data/files/uploads/{csvName}"
        res_json = f"{base_dir}data/files/json/{jsonName}"
        jpg_name = f"{base_dir}data/files/image/{csvName.replace('.csv', '.jpg')}"
        if int(newFile) == 1:
            ml_data = run_model_cls(frequency, GPR_Csv, res_json)
            huatu_cls(frequency, int(timeWindow), float(gap), int(excelent), int(qualified), int(insufficient), GPR_Csv, ml_data, jpg_name)
        else:
            f = open(res_json, "r", encoding="utf8")
            ml_data = (numpy.array(json.loads(f.read()))).flatten()
            f.close()
            huatu_cls(frequency, int(timeWindow), float(gap), int(excelent), int(qualified), int(insufficient), GPR_Csv,
                      ml_data, jpg_name)
        t = str(time.time())
        res = {
            "jpg": f"http://120.55.95.126/resnet/download_file?fileName={jpgName}&t={t}",
            "grpData": f"http://120.55.95.126/resnet/download_json?fileName={jsonName}&t={t}"
        }
        return {"status": 1, "message": "success", "result": res}
    except:
        app.logger.info(traceback.format_exc())
        return {
            "status": 0,
            "message": "请求错误！"
        }

# 下载文件
@app.route("/resnet/download_file", methods=['GET', 'POST'])
def DownloadFileCls():
    params = request.values
    fileName = params.get("fileName")
    fileName = f"{base_dir}data/files/image/{fileName}"
    return send_file(path_or_file=fileName)
    #file = open(fileName, "rb").read()
    #return Response(file, mimetype='image/jpeg')

# 下载json
@app.route("/resnet/download_json", methods=['GET', 'POST'])
def DownloadJsonCls():
    params = request.values
    fileName = params.get("fileName")
    fileName = f"{base_dir}data/files/json/{fileName}"
    return send_file(path_or_file=fileName)
    #file = open(fileName, "rb").read()

CORS(app, resources=r'/*')
if __name__=='__main__':
    app.run()
