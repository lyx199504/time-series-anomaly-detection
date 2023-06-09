#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/2/16 22:55
# @Author : LYX-夜光
import os
import sys
# 导入上一级目录，并回到上一级
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from optUtils import yaml_config
from optUtils.logUtil import get_lines_from_log

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import rcParams

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
        'cnn', 'ms_cnn',
        'c_lstm', 'c_lstm_ae', 'tcn',
        'imc_lstm_3_48', 'cmc_lstm_3_32', 'smc_lstm_3_32',
    ]
    dataset_list = [
        'A1Benchmark', 'realAdExchange', 'realAWSCloudwatch', 'realKnownCause', 'realTraffic', 'realTweets'
    ]
    f1 = pd.DataFrame(index=model_list, columns=dataset_list, dtype=float)
    r = pd.DataFrame(index=model_list, columns=dataset_list, dtype=float)
    p = pd.DataFrame(index=model_list, columns=dataset_list, dtype=float)
    a = pd.DataFrame(index=model_list, columns=dataset_list, dtype=float)
    for model_name in model_list:
        for dataset_name in dataset_list:
            try:
                log_name = '_'.join([model_name, dataset_name])
                metrics_mean = get_lines_from_log(log_name, -1)['test_score_dict']
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
    print(r)
    print(p)
    print(a)


    config = {
        "font.family": 'serif',
        "font.size": 18,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)

    # plt.rcParams['font.sans-serif'] = ['Times New Roman']
    # plt.rcParams['font.family'] = 'STSong'
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams['figure.figsize'] = (10, 4.8)

    # # 实验1
    # model_list = ['cnn', 'ms_cnn']
    # dataset_list = ['A1Benchmark', 'realAdExchange', 'realAWSCloudwatch', 'realKnownCause', 'realTraffic', 'realTweets']
    # data = f1.loc[model_list].values
    # X = np.arange(len(data[0]))
    # # fig = plt.figure(figsize=(12, 8))
    # fig = plt.figure()
    # # ax = fig.add_axes([0, 0, 1, 1])
    # # ax.bar(X + 0.00, data[0], color='r', width=0.25)
    # # ax.bar(X + 0.25, data[1], color='dodgerblue', width=0.25)
    # plt.bar(X - 0.125, data[0], color='#FF9E02', width=0.25, label="CNN")
    # plt.bar(X + 0.125, data[1], color='#87BBA4', width=0.25, label="MS-CNN")
    # X_ = ['$\mathrm{%s}$' % x for x in ['A1', 'Ad', 'AWS', 'Known', 'Traffic', 'Tweets']] + ['均值']
    # plt.xticks(X, X_, fontsize=15)
    # plt.ylim(0.5, 1)
    # plt.yticks(fontsize=15, fontproperties='Times New Roman')
    # plt.legend(fontsize=15, prop={"family": "Times New Roman", "size": 15})
    # # plt.savefig('mscnn.png', transparent=True)
    # plt.show()

    # # 实验2
    # fig = plt.figure(1, figsize=(21, 5.1))
    # size_list = [2, 3, 4]
    # hidden_list = [16, 32, 48, 64]
    # model_list = ['imc_lstm', 'cmc_lstm', 'smc_lstm']
    # model_name_list = ['IMC-LSTM', 'CMC-LSTM', 'SMC-LSTM']
    # dataset_list = ['A1Benchmark', 'realAdExchange', 'realAWSCloudwatch', 'realKnownCause', 'realTraffic', 'realTweets']
    # for ix, model_name in enumerate(model_list):
    #     xf1 = []
    #     for s in size_list:
    #         xf1_ = []
    #         for h in hidden_list:
    #             f1 = pd.DataFrame(index=model_list, columns=dataset_list, dtype=float)
    #             for dataset_name in dataset_list:
    #                 # yaml_config['dir']['log_dir'] = './log-%s-%s' % (s, h)
    #                 log_name = '_'.join([model_name, str(s), str(h), dataset_name])
    #                 # metrics_mean = get_lines_from_log(log_name, -1)['mean_score_dict']
    #                 metrics_mean = get_lines_from_log(log_name, 501)['test_score_dict']
    #                 f1.loc[model_name, dataset_name] = metrics_mean['f1_score']
    #             f1 = add_datas(f1)
    #             xf1_.append(f1.loc[model_name, 'mean'])
    #             # xf1_pos.append(f1_pos.loc[model_list, 'mean'].values)
    #         xf1.append(xf1_)
    #     xf1 = np.array(xf1)
    #     # xf1_pos = np.array(xf1_pos)
    #     print(xf1)
    #     # print(xf1_pos)
    #
    #     ax = fig.add_subplot(1, 3, ix + 1)
    #
    #     # heatmap = ax.pcolor(xf1, cmap='viridis', edgecolors='w', linewidths=2, vmax=97, vmin=92)
    #     # heatmap = ax.pcolor(xf1, cmap='YlGnBu_r', edgecolors='w', linewidths=2, vmax=97, vmin=92)
    #     heatmap = ax.pcolor(xf1, cmap='Reds_r', edgecolors='w', linewidths=2, vmax=0.965, vmin=0.925)
    #     cbar = fig.colorbar(heatmap)
    #     for label in cbar.ax.get_yticklabels():
    #         label.set_fontproperties('Times New Roman')
    #     # plt.imshow(xf1, cmap='bone', vmax=97, vmin=92)
    #     # plt.imshow(xf1, cmap='viridis', vmax=97, vmin=92)
    #     ax.set_title(model_name_list[ix], fontsize=20, fontproperties='Times New Roman')
    #
    #     ax.set_xticks(np.arange(len(hidden_list)) + 0.5, hidden_list, fontsize=15, fontproperties='Times New Roman')
    #     ax.set_xlabel('每类卷积核的数量（$\mathrm{N}$）', fontsize=15)
    #     ax.set_yticks(np.arange(len(size_list)) + 0.5, size_list, fontsize=15, fontproperties='Times New Roman')
    #     ax.set_ylabel('卷积核类型数量（$\mathrm{K}$）', fontsize=15)
    #     # plt.grid(which="major", color="w", linestyle='-', linewidth=3)
    # # plt.savefig('mclstm.png', transparent=True)
    # plt.show()

    # # 实验3
    # model_list = ['imc_lstm_3_48', 'cmc_lstm_3_32', 'smc_lstm_3_32']
    # dataset_list = ['A1Benchmark', 'realAdExchange', 'realAWSCloudwatch', 'realKnownCause', 'realTraffic', 'realTweets']
    # f1_ = f1.loc[model_list].values
    # X = ['$\mathrm{%s}$' % x for x in ['A1', 'Ad', 'AWS', 'Known', 'Traffic', 'Tweets']] + ['均值']
    #
    # plt.plot(X, f1_[0, :], '#4B74B2', linewidth=2, marker='^', markersize=10, label='IMC-LSTM')
    # plt.plot(X, f1_[1, :], '#DB3124', linewidth=2, marker='o', markersize=10, label='CMC-LSTM')
    # plt.plot(X, f1_[2, :], '#90BEE0', linewidth=2, marker='s', markersize=10, label='SMC-LSTM')
    # # plt.plot(X, f1_[3, :], '#FFDF92', linewidth=3, marker='v', markersize=12, label='FM')
    # plt.legend(fontsize=15, prop={"family": "Times New Roman", "size": 15})
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15, fontproperties='Times New Roman')
    # # plt.savefig('mclstm_2.png', transparent=True)
    # plt.show()

    # 实验4
    plt.rc('font', family='Times New Roman')
    model_list = ['tcn', 'c_lstm', 'c_lstm_ae', 'cmc_lstm_3_32']
    fig = plt.figure(1, figsize=(21, 11))
    xlowlist = [0.8, 0.9, 0.6, 0.65, 0.8, 0.8]
    xtoplist = [1, 1, 1, 1, 1, 1]
    dataset_name = ['A1', 'Ad', 'AWS', 'Known', 'Traffic', 'Tweets']
    for ix, data_name in enumerate(dataset_list):
        f1_ = f1.loc[model_list, data_name].values
        r_ = r.loc[model_list, data_name].values
        p_ = p.loc[model_list, data_name].values
        a_ = a.loc[model_list, data_name].values

        data = np.array([f1_, r_, p_, a_]).transpose()

        X = np.arange(len(data[0]))
        width = 0.15
        ax = fig.add_subplot(2, 3, ix + 1)
        # plt.bar(X - width*2, data[0], color='r', width=width, label="C-LSTM")
        # plt.bar(X - width, data[1], color='lightsalmon', width=width, label="C-LSTM-AE")
        # plt.bar(X, data[2], color='dodgerblue', width=width, label="IMC-LSTM")
        # plt.bar(X + width, data[3], color='gold', width=width, label="CMC-LSTM")
        # plt.bar(X + width*2, data[4], color='limegreen', width=width, label="SMC-LSTM")
        ax.bar(X - width/2*3, data[0], color='#90BEE0', width=width, label="TCN")
        ax.bar(X - width/2, data[1], color='#FFDF92', width=width, label="C-LSTM")
        ax.bar(X + width/2, data[2], color='#BEB8DC', width=width, label="C-LSTM-AE")
        ax.bar(X + width/2*3, data[3], color='#DB3124', width=width, label="CMC-LSTM")
        # ax.bar(X + width/2*3, data[3], color='#4B74B2', width=width, label="SMC-LSTM")
        plt.title(dataset_name[ix], fontsize=20)
        plt.xticks(X, ['F1', 'R', 'P', 'A'], fontsize=15)
        plt.ylim(xlowlist[ix], xtoplist[ix])
        plt.yticks(fontsize=15)
        # plt.legend(fontsize=12, loc=(0.02, 1.02), ncol=3)
        # plt.legend(fontsize=12)
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc=(0.33, 0.93), fontsize=15, frameon=False, ncol=4)
    plt.savefig('e4.png', transparent=True)
    plt.show()
