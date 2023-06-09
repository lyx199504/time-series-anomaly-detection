#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/2/28 13:29
# @Author : LYX-夜光
import numpy as np
from sklearn.metrics import precision_score


# 基于异常点的F1值
def f1_location_score(loc, y_true, y_pred):
    recall_loc = recall_location_score(loc, y_pred)
    precision = precision_score(y_true, y_pred)
    denominator = recall_loc + precision
    return 0 if denominator == 0 else 2*precision*recall_loc / denominator

# 基于异常点的召回率
def recall_location_score(loc, y_pred):
    max_loc = max(loc)
    anomaly_num = np.zeros(max_loc)
    pred_num = np.zeros(max_loc)
    for ix, l in enumerate(loc):
        if l == 0:
            continue
        anomaly_num[l-1] += 1
        if y_pred[ix] == 1:
            pred_num[l-1] += 1

    location = np.arange(max_loc) + 1
    pred_weighted = np.sum(location * pred_num)
    anomaly_weighted = np.sum(location * anomaly_num)
    recall_loc = pred_weighted / anomaly_weighted
    return recall_loc
