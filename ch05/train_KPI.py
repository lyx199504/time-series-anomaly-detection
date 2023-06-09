#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/3/5 15:16
# @Author : LYX-夜光
import os
import sys
# 导入上一级目录，并回到上一级
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, accuracy_score
from ch05.metrics import f1_location_score, recall_location_score

from dataPreprocessing import getDataset, standard

from dl_models.c_lstm import C_LSTM
from dl_models.c_lstm_ae import C_LSTM_AE
from dl_models.tcn import TCN
from dl_models.fft_1d_cnn import FFT_1D_CNN
from dl_models.mc_lstm import CMC_LSTM
from dl_models.f_se_lstm import F_SE_LSTM
from dl_models.tf_ad import TF_AD, T_AD, F_AD

from optUtils import yaml_config
from optUtils.dataUtil import stratified_shuffle_index

if __name__ == "__main__":
    seq_len = 60
    model_list = [C_LSTM, C_LSTM_AE, TCN, FFT_1D_CNN, CMC_LSTM, F_SE_LSTM, TF_AD, T_AD, F_AD]

    for model_clf in model_list:
        X_pre, y_pre, loc_pre = getDataset("KPI_preliminary", seq_len=seq_len, pre_list=[standard])
        X_fin, y_fin, loc_fin = getDataset("KPI_final", seq_len=seq_len, pre_list=[standard])

        X = np.vstack([X_pre, X_fin])
        y = np.hstack([y_pre, y_fin])
        loc = np.hstack([loc_pre, loc_fin])

        seed, fold = yaml_config['cus_param']['seed'], yaml_config['cv_param']['fold']
        # 按照标签分层抽样
        shuffle_index = stratified_shuffle_index(y, n_splits=fold, random_state=seed)
        X, y = X[shuffle_index], y[shuffle_index]
        loc = loc[shuffle_index]

        P, total = sum(y > 0), len(y)
        print("+: %d (%.2f%%)" % (P, P / total * 100), "-: %d (%.2f%%)" % (total - P, (1 - P / total) * 100))

        train_point, val_point = int(len(X) * 0.6), int(len(X) * 0.8)

        model = model_clf(learning_rate=0.001, batch_size=512, epochs=500, random_state=1, seq_len=seq_len)

        # model.create_model()
        # print(sum([param.nelement() for param in model.parameters()]))
        # exit()

        model.model_name = model.model_name + "_KPI"
        model.param_search = False
        model.save_model = True
        model.device = 'cuda:0'
        model.metrics = f1_score
        model.fit(X[:train_point], y[:train_point], X[train_point:val_point], y[train_point:val_point])
        model.metrics_list = [recall_score, f1_location_score, recall_location_score, precision_score, roc_auc_score, accuracy_score]
        model.test_score(X[val_point:], y[val_point:], loc=loc[val_point:])
