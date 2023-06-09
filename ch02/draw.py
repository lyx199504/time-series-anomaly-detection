#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/3/10 10:29
# @Author : LYX-夜光
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import rcParams

from dataPreprocessing import standard
from optUtils import read_json

import matplotlib.pyplot as plt


def read_NAB_data(dataset_name):
    data_list = []
    label_list = []
    # windows = read_json("../datasets/Numenta Anomaly Benchmark/labels/combined_windows.json")
    labels = read_json("../datasets/Numenta Anomaly Benchmark/labels/combined_labels.json")
    path_list = list(Path("../datasets/Numenta Anomaly Benchmark/data/%s" % dataset_name).glob("*.csv"))
    for path in path_list:
        raw_data = pd.read_csv(path, index_col='timestamp')
        raw_value = raw_data.value
        raw_value = standard(raw_value)
        data_list.append(raw_value)
        # for window in windows[dataset_name + "/%s" % str(path).split('\\')[-1]]:
        #     start, end = window[0].split('.')[0], window[1].split('.')[0]
        #     raw_data.loc[start: end, 'label'] = 1
        for timestamp in labels[dataset_name + "/%s" % str(path).split('\\')[-1]]:
            raw_data.loc[timestamp, 'label'] = 1
        raw_label = raw_data.label
        label_list.append(raw_label)
    return data_list, label_list

def read_Yahoo_data(dataset_name):
    data_list = []
    label_list = []
    path_list = list(Path("../datasets/Yahoo! Webscope S5/%s" % dataset_name).glob("*.csv"))

    for path in path_list:
        raw_data = pd.read_csv(path)
        try:
            raw_value = raw_data.value.values
            raw_value = standard(raw_value)
            data_list.append(raw_value)
        except:
            continue

        # 构造时间序列标签
        try:
            raw_label = raw_data.is_anomaly
        except:
            raw_label = raw_data.anomaly
        label_list.append(raw_label)

    return data_list, label_list

if __name__ == "__main__":
    data_list, label_list = read_Yahoo_data("A1Benchmark")

    # dataset_name = "artificialWithAnomaly"
    # data_list, label_list = read_NAB_data(dataset_name)

    config = {
        "font.family": 'serif',
        "font.size": 18,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)

    # plt.rc('font', family='Times New Roman')
    fig = plt.figure(1, figsize=(20, 5))

    ax = fig.add_subplot(1, 3, 1)
    no = 0
    series = data_list[no]
    st, en = 500, 700
    ix = np.arange(0, len(series))[st: en]
    datas = series[ix]
    ax.plot(ix, datas, c='dodgerblue')
    ax.set_title("平稳性", fontsize=20)
    ax.set_ylim([-3, 12])
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(1, 3, 2)
    no = 40
    series = data_list[no]
    st, en = 200, 350
    ix = np.arange(0, len(series))[st: en]
    datas = series[ix]
    ax.plot(ix, datas, c='dodgerblue')
    ax.set_title("周期性", fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(1, 3, 3)
    no = 2
    series = data_list[no]
    st, en = 400, 1200
    ix = np.arange(0, len(series))[st: en]
    datas = series[ix]
    ax.plot(ix, datas, c='dodgerblue')
    ax.set_title("非平稳非周期性", fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.axis('off')
    plt.savefig('type.png', transparent=True)
    plt.show()


    # # plt.rc('font', family='Times New Roman')
    # fig = plt.figure(1, figsize=(20, 5))
    #
    # ax = fig.add_subplot(1, 3, 1)
    # no = 0
    # series = data_list[no]
    # label = label_list[no]
    # st, en = 1100, 1300
    # ix = np.arange(0, len(series))[st: en]
    # datas = series[ix]
    # labels = label[ix]
    # ax.plot(ix, datas, c='dodgerblue')
    # ax.set_title("点异常", fontsize=20)
    # ax.scatter(ix[labels > 0], datas[labels > 0], c='red', s=100)
    # ax.set_xticks([])
    # ax.set_yticks([])
    #
    # ax = fig.add_subplot(1, 3, 2)
    # no = 20
    # series = data_list[no]
    # label = label_list[no]
    # st, en = 750, 1050
    # ix = np.arange(0, len(series))[st: en]
    # datas = series[ix]
    # labels = label[ix]
    # ax.plot(ix, datas, c='dodgerblue')
    # ax.set_title("集合异常", fontsize=20)
    # ax.scatter(ix[labels > 0], datas[labels > 0], c='red', s=100)
    # ax.set_xticks([])
    # ax.set_yticks([])
    #
    # ax = fig.add_subplot(1, 3, 3)
    # no = 40
    # series = data_list[no]
    # label = label_list[no]
    # st, en = 1200, 1700
    # ix = np.arange(0, len(series))[st: en]
    # datas = series[ix]
    # labels = label[ix]
    # ax.plot(ix, datas, c='dodgerblue')
    # ax.set_title("上下文异常", fontsize=20)
    # ax.scatter(ix[labels > 0], datas[labels > 0], c='red', s=100)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # # ax.axis('off')
    # # plt.savefig('anomaly.png', transparent=True)
    # plt.show()