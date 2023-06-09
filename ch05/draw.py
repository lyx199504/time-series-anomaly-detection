#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/2/16 22:55
# @Author : LYX-夜光
import os
import sys
# 导入上一级目录，并回到上一级
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path

from dataPreprocessing import standard

from matplotlib import rcParams

from optUtils import yaml_config, read_json
from optUtils.logUtil import get_lines_from_log

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 200)


def read_NAB_data(dataset_name):
    data_list = []
    label_list = []
    # windows = read_json("./datasets/Numenta Anomaly Benchmark/labels/combined_windows.json")
    labels = read_json("./datasets/Numenta Anomaly Benchmark/labels/combined_labels.json")
    path_list = list(Path("./datasets/Numenta Anomaly Benchmark/data/%s" % dataset_name).glob("*.csv"))
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
    path_list = list(Path("./datasets/Yahoo! Webscope S5/%s" % dataset_name).glob("*.csv"))

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

def add_datas(pd_name):
    pd_name['mean'] = pd_name.mean(axis=1)
    pd_name = pd_name.round(4)
    # for col in pd_name:
    #     pd_name.loc['best_model', col] = pd_name.index[np.argmax(pd_name[col])]
    return pd_name


if __name__ == "__main__":
    config = {
        "font.family": 'serif',
        "font.size": 18,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)

    # # plt.rc('font', family='Times New Roman')
    # fig = plt.figure(1, figsize=(10, 7))
    #
    # # data_list, label_list = read_NAB_data("realAdExchange")
    # data_list, label_list = read_Yahoo_data("A1Benchmark")
    #
    # ax = fig.add_subplot(2, 1, 1)
    # no = 16
    # series = data_list[no]
    # label = label_list[no]
    #
    # st, en = 424, 600
    # ix = np.arange(0, len(series))[st: en]
    # datas = series[ix]
    # labels = label[ix]
    # ax.plot(ix, datas, c='dodgerblue')
    # ax.scatter(ix[labels > 0], datas[labels > 0], c='red', s=100)
    # ax.set_xticks([])
    # ax.set_yticks([])
    #
    # ax = fig.add_subplot(2, 1, 2)
    # st, en = 1200, 1347
    # ix = np.arange(0, len(series))[st: en]
    # datas = series[ix]
    # labels = label[ix]
    # ax.plot(ix, datas, c='dodgerblue')
    # ax.scatter(ix[labels > 0], datas[labels > 0], c='red', s=100)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # # ax.axis('off')
    # plt.savefig('./ch05/location.png', transparent=True)
    # plt.show()


    model_list = [
        'c_lstm', 'c_lstm_ae', 'tcn', 'fft_1d_cnn',
        'cmc_lstm', 'f_se_lstm',
        'tf_ad_60_10', 'tf_ad_60_20', 'tf_ad_60_30', 'tf_ad_60_40', 'tf_ad_60_50',
        't_ad', 'f_ad',
    ]
    dataset_list = [
        'A1Benchmark', 'realAdExchange', 'realAWSCloudwatch', 'realKnownCause', 'realTraffic', 'realTweets',
        'KPI',
    ]
    f1_loc = pd.DataFrame(index=model_list, columns=dataset_list, dtype=float)
    r_loc = pd.DataFrame(index=model_list, columns=dataset_list, dtype=float)
    f1 = pd.DataFrame(index=model_list, columns=dataset_list, dtype=float)
    r = pd.DataFrame(index=model_list, columns=dataset_list, dtype=float)
    p = pd.DataFrame(index=model_list, columns=dataset_list, dtype=float)
    auc = pd.DataFrame(index=model_list, columns=dataset_list, dtype=float)
    for model_name in model_list:
        for dataset_name in dataset_list:
            try:
                log_name = '_'.join([model_name, dataset_name])
                metrics_mean = get_lines_from_log(log_name, -1)['test_score_dict']
                f1_loc.loc[model_name, dataset_name] = metrics_mean['f1_location_score']
                r_loc.loc[model_name, dataset_name] = metrics_mean['recall_location_score']
                f1.loc[model_name, dataset_name] = metrics_mean['f1_score']
                r.loc[model_name, dataset_name] = metrics_mean['recall_score']
                p.loc[model_name, dataset_name] = metrics_mean['precision_score']
                auc.loc[model_name, dataset_name] = metrics_mean['roc_auc_score']
            except:
                f1_loc.loc[model_name, dataset_name] = np.nan
                r_loc.loc[model_name, dataset_name] = np.nan
                f1.loc[model_name, dataset_name] = np.nan
                r.loc[model_name, dataset_name] = np.nan
                p.loc[model_name, dataset_name] = np.nan
                auc.loc[model_name, dataset_name] = np.nan
    f1_loc = add_datas(f1_loc)
    r_loc = add_datas(r_loc)
    f1 = add_datas(f1)
    r = add_datas(r)
    p = add_datas(p)
    auc = add_datas(auc)

    print("F1_loc: \n", f1_loc, "\n")
    print("F1: \n", f1, "\n")
    print("R_loc: \n", r_loc, "\n")
    print("R: \n", r, "\n")
    print("P: \n", p, "\n")
    print("AUC: \n", auc, "\n")


    # yaml_config['dir']['log_dir'] = './log'

    # # 滑动窗口大小的合理性验证
    # plt.rc('font', family='Times New Roman')
    # # plt.figure(1, figsize=(20, 5))
    # model_list = ['tf_ad_60_10', 'tf_ad_60_20', 'tf_ad_60_30', 'tf_ad_60_40', 'tf_ad_60_50']
    # x_labels = ['$\mathrm{F1_{\it{loc}}}$', '$\mathrm{R_{\it{loc}}}$', 'F1', 'R', 'P', 'AUC']
    # data = np.array([
    #     f1_loc.loc[model_list, 'KPI'].values,
    #     r_loc.loc[model_list, 'KPI'].values,
    #     f1.loc[model_list, 'KPI'].values,
    #     r.loc[model_list, 'KPI'].values,
    #     p.loc[model_list, 'KPI'].values,
    #     auc.loc[model_list, 'KPI'].values,
    # ])
    # plt.plot(x_labels, data[:, 0], '#FFDF92', linewidth=2, marker='v', markersize=10, label='10')
    # plt.plot(x_labels, data[:, 1], '#4B74B2', linewidth=2, marker='s', markersize=10, label='20')
    # plt.plot(x_labels, data[:, 2], '#FC8C5A', linewidth=2, marker='o', markersize=10, label='30')
    # plt.plot(x_labels, data[:, 3], '#90BEE0', linewidth=2, marker='*', markersize=10, label='40')
    # plt.plot(x_labels, data[:, 4], '#DB3124', linewidth=2, marker='^', markersize=10, label='50')
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.legend(loc='lower right', fontsize=15, frameon=False)
    # # plt.savefig('ex1.png', transparent=True)
    # plt.show()

    # # 时频特征的有效性验证
    # plt.figure(1, figsize=(8, 6))
    # model_list = ['t_ad', 'f_ad', 'tf_ad_60_30']
    # model_name_list = ['T-AD', 'F-AD', 'TF-AD']
    # color_list = ['#90BEE0', '#4B74B2', '#FC8C5A']
    # x = np.arange(0, 500, 10)
    # for ix, model_name in enumerate(model_list):
    #     log_name = '_'.join([model_name, "KPI"])
    #     loss_list = np.array(list(map(lambda x: x['val_loss'], get_lines_from_log(log_name, [0, 500]))))
    #     plt.plot(x, loss_list[x], color_list[ix], linewidth=2, label=model_name_list[ix])
    #
    # plt.xlabel("训练轮数（$\mathrm{epoch}$）", fontsize=15)
    # plt.ylabel("验证损失值", fontsize=15)
    # plt.xticks(fontsize=15, fontproperties='Times New Roman')
    # plt.yticks(fontsize=15, fontproperties='Times New Roman')
    # plt.legend(fontsize=15, prop={"family": "Times New Roman", "size": 15}, frameon=False)
    # # plt.savefig('ex2-1.png', transparent=True)
    # plt.show()

    # plt.rc('font', family='Times New Roman')
    # # plt.figure(1, figsize=(20, 5))
    # model_list = ['t_ad', 'f_ad', 'tf_ad_60_30']
    # x_labels = ['$\mathrm{F1_{\it{loc}}}$', '$\mathrm{R_{\it{loc}}}$', 'F1', 'R', 'P', 'AUC']
    # data = np.array([
    #     f1_loc.loc[model_list, 'KPI'].values,
    #     r_loc.loc[model_list, 'KPI'].values,
    #     f1.loc[model_list, 'KPI'].values,
    #     r.loc[model_list, 'KPI'].values,
    #     p.loc[model_list, 'KPI'].values,
    #     auc.loc[model_list, 'KPI'].values,
    # ])
    # plt.plot(x_labels, data[:, 0], '#90BEE0', linewidth=2, marker='v', markersize=10, label='T-AD')
    # plt.plot(x_labels, data[:, 1], '#4B74B2', linewidth=2, marker='s', markersize=10, label='F-AD')
    # plt.plot(x_labels, data[:, 2], '#FC8C5A', linewidth=2, marker='o', markersize=10, label='TF-AD')
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.legend(loc='lower right', fontsize=15, frameon=False)
    # # plt.savefig('ex1.png', transparent=True)
    # plt.show()

    # # 实验4
    # plt.rc('font', family='Times New Roman')
    # model_list = ['lstm_dnn', 'cnn_lstm_dnn', 'f_se_lstm']
    # x_labels = ['A1', 'Ad', 'AWS', 'Known', 'Traffic', 'Tweets']
    # f1_ = f1.loc[model_list, dataset_list].values
    # plt.plot(x_labels, f1_[0, :], '#FFDF92', linewidth=2, marker='^', markersize=10, label='LSTM+DNN')
    # plt.plot(x_labels, f1_[1, :], '#4B74B2', linewidth=2, marker='o', markersize=10, label='CNN+LSTM+DNN')
    # plt.plot(x_labels, f1_[2, :], '#DB3124', linewidth=2, marker='s', markersize=10, label='SENet+LSTM+DNN')
    # plt.xticks(x_labels, fontsize=15)
    # # plt.ylim(0, 1)
    # # plt.ylabel('F1 score', fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.legend(loc='lower left', fontsize=15, frameon=False)
    # plt.savefig('ex4.png', transparent=True)
    # plt.show()

    # # 实验3
    # dataset_name = ['A1', 'Ad', 'AWS', 'Known', 'Traffic', 'Tweets']
    # # metrics_list = [f1, r, p]
    # # metrics_name = ['F1', 'Recall', 'Precision']
    # xlowlist = [0.95, 0.95, 0.88, 0.80, 0.92, 0.95]
    # xtoplist = [1, 1, 1, 1, 1, 1]
    # fig = plt.figure(1, figsize=(20, 10))
    # for ix, data_name in enumerate(dataset_list):
    #     f1_ = f1.loc[model_list, data_name].values
    #     r_ = r.loc[model_list, data_name].values
    #     p_ = p.loc[model_list, data_name].values
    #
    #     data = np.array([f1_, r_, p_]).transpose()
    #
    #     X = np.arange(len(data[0]))
    #     width = 0.15
    #     ax = fig.add_subplot(2, 3, ix + 1)
    #     ax.bar(X - width/2*3, data[0], color='#90BEE0', width=width, label="C-LSTM")
    #     ax.bar(X - width/2, data[1], color='#FFDF92', width=width, label="C-LSTM-AE")
    #     ax.bar(X + width/2, data[2], color='#DB3124', width=width, label="CMC-LSTM")
    #     ax.bar(X + width/2*3, data[3], color='#4B74B2', width=width, label="F-SE-LSTM")
    #     plt.title(dataset_name[ix], fontsize=20)
    #     plt.xticks(X, ['F1', 'R', 'P'], fontsize=15)
    #     plt.ylim(xlowlist[ix], xtoplist[ix])
    #     plt.yticks(fontsize=15)
    #     # plt.legend(fontsize=12, loc=(0.02, 1.02), ncol=3)
    #     plt.legend(loc='upper left', fontsize=12, frameon=False)
    # # plt.savefig('ex5.png', transparent=True)
    # plt.show()

    # # 实验3.1
    # plt.figure(1, figsize=(8, 6))
    # model_list = ['tcn', 'c_lstm', 'c_lstm_ae', 'fft_1d_cnn', 'cmc_lstm', 'f_se_lstm', 'tf_ad_60_30']
    # model_name_list = ['TCN', 'C-LSTM', 'C-LSTM-AE', 'FFT-1D-CNN', 'CMC-LSTM', 'F-SE-LSTM', 'TF-AD']
    # color_list = ['#90BEE0', '#FFDF92', '#BEB8DC', '#8ECFC9', '#DB3124', '#4B74B2', '#FC8C5A']
    #
    # for ix, model_name in enumerate(model_list):
    #     log_name = '_'.join([model_name, "KPI"])
    #     loss_list = np.array(list(map(lambda x: x['val_loss'], get_lines_from_log(log_name, [0, 500]))))
    #     x = np.arange(0, len(loss_list), 9)
    #     plt.plot(x, loss_list[x], color_list[ix], linewidth=2, label=model_name_list[ix])
    #
    # plt.xlabel("训练轮数（$\mathrm{epoch}$）", fontsize=15)
    # plt.ylabel("验证损失值", fontsize=15)
    # plt.xticks(fontsize=15, fontproperties='Times New Roman')
    # plt.yticks(fontsize=15, fontproperties='Times New Roman')
    # plt.legend(fontsize=15, prop={"family": "Times New Roman", "size": 15}, frameon=False)
    # # plt.savefig('ex3-1.png', transparent=True)
    # plt.show()
