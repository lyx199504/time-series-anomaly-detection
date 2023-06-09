#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/2/26 15:56
# @Author : LYX-夜光
import time

import joblib
import numpy as np

from dataPreprocessing import getDataset, standard
from optUtils.logUtil import get_lines_from_log

if __name__ == "__main__":
    seq_len = 60
    model_name_list = ['c_lstm', 'c_lstm_ae', 'c_lstm', 'tcn', 'fft_1d_cnn', 'f_se_lstm', 'cmc_lstm', 'tf_ad_60_30']

    # dataset_name = "A1Benchmark"
    # X, y, _ = getDataset(dataset_name, seq_len=seq_len, pre_list=[standard])

    dataset_name = 'KPI'
    X_pre, y_pre, _ = getDataset("KPI_preliminary", seq_len=seq_len, pre_list=[standard])
    X_fin, y_fin, _ = getDataset("KPI_final", seq_len=seq_len, pre_list=[standard])
    X = np.vstack([X_pre, X_fin])
    y = np.hstack([y_pre, y_fin])

    P, total = sum(y > 0), len(y)
    print("+: %d (%.2f%%)" % (P, P / total * 100), "-: %d (%.2f%%)" % (total - P, (1 - P / total) * 100))

    for model_name in model_name_list:

        model_param = get_lines_from_log('%s_%s' % (model_name, dataset_name), 0)
        mdl = joblib.load(model_param['model_path'])
        mdl.device = 'cuda:0'

        start_time = time.time()
        y_pred = mdl.predict(X, batch_size=512)
        print("%s: " % model_name, time.time() - start_time)