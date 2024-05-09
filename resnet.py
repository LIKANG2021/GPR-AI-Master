# -*- coding: utf-8 -*-
"""
@author: Kang Li, Tongji University
"""
import os, json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from scipy.interpolate import interp2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##填充数据具有相同的列数
def pad_sample_data(input_array, target_shape=(1024, 506)):
    if input_array.shape == target_shape:
        # 输入数组已经是目标尺寸，无需插值
        return input_array
    else:
        # 创建插值函数
        interpolator = interp2d(np.arange(input_array.shape[1]), np.arange(input_array.shape[0]), input_array,
                                kind='linear')

        # 生成插值结果
        x_new = np.linspace(0, input_array.shape[1] - 1, target_shape[1])
        y_new = np.linspace(0, input_array.shape[0] - 1, target_shape[0])
        interpolated_data = interpolator(x_new, y_new)

        return interpolated_data

# 模型运行
def run_model_cls(GPR_Frequency, GPR_Csv, res_json):
    ##导入模型
    #GPR_Frequency = '300'  ##网站上传入的参数
    model_name_dict = {
        '300': 'grp_model/300.pth',
        '400': 'grp_model/400.pth',
        '600': 'grp_model/600.pth',
        '700': 'grp_model/700.pth'
    }
    model_name = model_name_dict[GPR_Frequency]  # 模型名
    resnet_model = torch.load(model_name, map_location='cpu')

    ##读入测试集数据
    test_data_path = GPR_Csv  # 上传的csv文件
    data_test = pd.read_csv(test_data_path, header=None)

    ##数据维度填充
    data_test = pad_sample_data(data_test)

    X_test = torch.tensor(data_test, dtype=torch.float32)
    X_expand_test = X_test.unsqueeze(0)  # 在第二个维度上添加一个维度，将X变为(129, 1, 1024, 506)
    X_expand_test = X_expand_test.unsqueeze(1)  # 在第二个维度上添加一个维度，将X变为(129, 1, 1024, 506)
    X_expand = X_expand_test.expand(-1, 3, -1, -1)

    X_expand = X_expand.to(device)
    resnet_model = resnet_model.to(device)

    outputs = resnet_model(X_expand)
    predictions = outputs.detach().cpu()
    f = open(res_json, "w", encoding="utf8")
    f.write(json.dumps(predictions.tolist()))
    f.close()
    return (predictions.numpy()).flatten()
