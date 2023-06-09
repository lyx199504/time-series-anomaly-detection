#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/10/25 11:08
# @Author : LYX-夜光
import os
import sys
# 导入上一级目录，并回到上一级
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

from dataPreprocessing import getDataset, standard
from dl_models.cnn import CNN_F_2
from dl_models.dnn import DNN_F_2
from dl_models.lstm import LSTM_F_2

from optUtils import yaml_config
from optUtils.dataUtil import stratified_shuffle_index


if __name__ == "__main__":
    seq_len = 60
    sub_seq_len_list = [10, 20, 30, 50]
    dataset_list = ['realAdExchange', 'realTraffic', 'realKnownCause', 'realAWSCloudwatch', 'A1Benchmark', 'realTweets']
    model_list = [DNN_F_2, CNN_F_2, LSTM_F_2]

    for model_clf in model_list:
        for sub_seq_len in sub_seq_len_list:
            for dataset_name in dataset_list:
                X, y, _ = getDataset(dataset_name, seq_len=seq_len, pre_list=[standard])

                seed, fold = yaml_config['cus_param']['seed'], yaml_config['cv_param']['fold']
                # 按照标签分层抽样
                shuffle_index = stratified_shuffle_index(y, n_splits=fold, random_state=seed)
                X, y = X[shuffle_index], y[shuffle_index]

                P, total = sum(y > 0), len(y)
                print("+: %d (%.2f%%)" % (P, P / total * 100), "-: %d (%.2f%%)" % (total - P, (1 - P / total) * 100))

                train_point, val_point = int(len(X) * 0.6), int(len(X) * 0.8)

                model = model_clf(learning_rate=0.001, batch_size=512, epochs=500, random_state=1, seq_len=seq_len, sub_seq_len=sub_seq_len)
                model.model_name = model.model_name + "_%s" % dataset_name
                model.param_search = False
                model.save_model = True
                model.device = 'cuda:1'
                model.metrics = f1_score
                model.metrics_list = [recall_score, precision_score, accuracy_score]
                model.fit(X[:train_point], y[:train_point], X[train_point:val_point], y[train_point:val_point])
                model.test_score(X[val_point:], y[val_point:])
