#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/2/16 22:55
# @Author : LYX-夜光
import os
import sys
# 导入上一级目录，并回到上一级
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from matplotlib import rcParams

from optUtils import yaml_config
from optUtils.logUtil import get_lines_from_log

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 200)

def add_datas(pd_name):
    pd_name['mean'] = pd_name.mean(axis=1)
    pd_name = pd_name.round(4)
    # for col in pd_name:
    #     pd_name.loc['best_model', col] = pd_name.index[np.argmax(pd_name[col])]
    return pd_name

if __name__ == "__main__":
    model_list = [
        'dnn_t_1', 'dnn_f_1', 'dnn_t_2', 'dnn_f_2',
        'cnn_t_1', 'cnn_f_1', 'cnn_t_2', 'cnn_f_2',
        'lstm_t_1', 'lstm_f_1', 'lstm_t_2', 'lstm_f_2',
        'dnn_10', 'dnn_20', 'dnn_f_2', 'dnn_40', 'dnn_50',
        'cnn_10', 'cnn_20', 'cnn_f_2', 'cnn_40', 'cnn_50',
        'lstm_10', 'lstm_20', 'lstm_f_2', 'lstm_40', 'lstm_50',
        'lstm_dnn', 'cnn_lstm_dnn', 'f_se_lstm',
        'c_lstm', 'c_lstm_ae', 'tcn', 'fft_1d_cnn',
        'cmc_lstm_3_32',
    ]
    dataset_list = [
        'A1Benchmark', 'realAdExchange', 'realAWSCloudwatch', 'realKnownCause', 'realTraffic', 'realTweets',
    ]
    f1 = pd.DataFrame(index=model_list, columns=dataset_list, dtype=float)
    r = pd.DataFrame(index=model_list, columns=dataset_list, dtype=float)
    p = pd.DataFrame(index=model_list, columns=dataset_list, dtype=float)
    a = pd.DataFrame(index=model_list, columns=dataset_list, dtype=float)
    for model_name in model_list:
        for dataset_name in dataset_list:
            try:
                log_name = '_'.join([model_name, dataset_name])
                metrics_mean = get_lines_from_log(log_name, 501)['test_score_dict']
                f1.loc[model_name, dataset_name] = metrics_mean['f1_score']
                r.loc[model_name, dataset_name] = metrics_mean['recall_score']
                p.loc[model_name, dataset_name] = metrics_mean['precision_score']
                a.loc[model_name, dataset_name] = metrics_mean['accuracy_score']
            except:
                f1.loc[model_name, dataset_name] = np.nan
                r.loc[model_name, dataset_name] = np.nan
                p.loc[model_name, dataset_name] = np.nan
                a.loc[model_name, dataset_name] = np.nan
    f1 = add_datas(f1)
    r = add_datas(r)
    p = add_datas(p)
    a = add_datas(a)

    print(f1)

    config = {
        "font.family": 'serif',
        "font.size": 18,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)

    # plt.rcParams['font.sans-serif'] = ['Times New Roman']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams['figure.figsize'] = (10, 4.8)


    yaml_config['dir']['log_dir'] = './log'


    # # 实验2
    # plt.rc('font', family='Times New Roman')
    # plt.figure(1, figsize=(20, 15))
    # model_list = [
    #     ['dnn_t_1', 'dnn_f_1', 'dnn_t_2', 'dnn_f_2'],
    #     ['cnn_t_1', 'cnn_f_1', 'cnn_t_2', 'cnn_f_2'],
    #     ['lstm_t_1', 'lstm_f_1', 'lstm_t_2', 'lstm_f_2'],
    # ]
    # x_labels = ['A1', 'Ad', 'AWS', 'Known', 'Traffic', 'Tweets']
    # left_list = [[0.1, 0.1, 0.6], [0.0, 0.0, 0.5], [0.4, 0.6, 0.7]]
    # label_list = ['F1', 'R', 'P']
    # for k, metric in enumerate([f1, r, p]):
    #     left = left_list[k]
    #     for i, models in enumerate(model_list):
    #         m = metric.loc[models, dataset_list].values
    #         plt.subplot(3, 3, k*3+i+1)
    #         plt.plot(x_labels, m[0, :], '#90BEE0', linewidth=3, marker='^', markersize=12, label='TV')
    #         plt.plot(x_labels, m[1, :], '#FFDF92', linewidth=3, marker='o', markersize=12, label='FV')
    #         plt.plot(x_labels, m[2, :], '#4B74B2', linewidth=3, marker='s', markersize=12, label='TM')
    #         plt.plot(x_labels, m[3, :], '#DB3124', linewidth=3, marker='v', markersize=12, label='FM')
    #         plt.title(models[0].split('_')[0].upper(), fontsize=20)
    #         plt.xticks(x_labels, fontsize=15)
    #         plt.ylim(left[i], 1)
    #         plt.ylabel(label_list[k], fontsize=15)
    #         plt.yticks(fontsize=15)
    #         plt.legend(loc='lower center', fontsize=12, frameon=False, ncol=4)
    # # plt.savefig('ex2.png', transparent=True)
    # plt.show()


    # # 实验4
    # plt.rc('font', family='Times New Roman')
    # model_list = ['lstm_dnn', 'cnn_lstm_dnn', 'f_se_lstm']
    # x_labels = ['A1', 'Ad', 'AWS', 'Known', 'Traffic', 'Tweets']
    # for k, metric in enumerate([f1, r, p]):
    #     f1_ = f1.loc[model_list, dataset_list].values
    #     plt.plot(x_labels, f1_[0, :], '#FFDF92', linewidth=2, marker='^', markersize=10, label='LSTM+DNN')
    #     plt.plot(x_labels, f1_[1, :], '#4B74B2', linewidth=2, marker='o', markersize=10, label='CNN+LSTM+DNN')
    #     plt.plot(x_labels, f1_[2, :], '#DB3124', linewidth=2, marker='s', markersize=10, label='SENet+LSTM+DNN')
    #     plt.xticks(x_labels, fontsize=15)
    #     # plt.ylim(0, 1)
    #     # plt.ylabel('F1 score', fontsize=15)
    #     plt.yticks(fontsize=15)
    #     plt.legend(loc='lower left', fontsize=15, frameon=False)
    # plt.savefig('ex4.png', transparent=True)
    # plt.show()

    # # 实验5
    # plt.rc('font', family='Times New Roman')
    # model_list = ['tcn', 'c_lstm', 'c_lstm_ae', 'fft_1d_cnn', 'cmc_lstm_3_32', 'f_se_lstm']
    # dataset_name = ['A1', 'Ad', 'AWS', 'Known', 'Traffic', 'Tweets']
    # # metrics_list = [f1, r, p]
    # # metrics_name = ['F1', 'Recall', 'Precision']
    # xlowlist = [0.75, 0.9, 0.6, 0.65, 0.8, 0.8]
    # xtoplist = [1, 1, 1, 1, 1, 1]
    # fig = plt.figure(1, figsize=(21, 11))
    # for ix, data_name in enumerate(dataset_list):
    #     f1_ = f1.loc[model_list, data_name].values
    #     r_ = r.loc[model_list, data_name].values
    #     p_ = p.loc[model_list, data_name].values
    #
    #     data = np.array([f1_, r_, p_]).transpose()
    #
    #     X = np.arange(len(data[0]))
    #     width = 0.12
    #     ax = fig.add_subplot(2, 3, ix + 1)
    #     ax.bar(X - width/2*5, data[0], color='#90BEE0', width=width, label="TCN")
    #     ax.bar(X - width/2*3, data[1], color='#FFDF92', width=width, label="C-LSTM")
    #     ax.bar(X - width/2, data[2], color='#BEB8DC', width=width, label="C-LSTM-AE")
    #     ax.bar(X + width/2, data[3], color='#8ECFC9', width=width, label="FFT-1D-CNN")
    #     ax.bar(X + width/2*3, data[4], color='#DB3124', width=width, label="CMC-LSTM")
    #     ax.bar(X + width/2*5, data[5], color='#4B74B2', width=width, label="F-SE-LSTM")
    #     plt.title(dataset_name[ix], fontsize=20)
    #     plt.xticks(X, ['F1', 'R', 'P'], fontsize=15)
    #     plt.ylim(xlowlist[ix], xtoplist[ix])
    #     plt.yticks(fontsize=15)
    #     # plt.legend(fontsize=12, loc=(0.02, 1.02), ncol=3)
    #     # plt.legend(loc='upper left', fontsize=12, frameon=False)
    # lines, labels = fig.axes[-1].get_legend_handles_labels()
    # fig.legend(lines, labels, loc=(0.25, 0.93), fontsize=15, frameon=False, ncol=6)
    # # plt.savefig('ex5.png', transparent=True)
    # plt.show()
