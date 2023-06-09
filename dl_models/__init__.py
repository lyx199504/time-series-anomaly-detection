#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/4/23 22:35
# @Author : LYX-夜光

import joblib

from optUtils import yaml_config
from optUtils.logUtil import logging_config, get_lines_from_log
from optUtils.pytorchModel import DeepLearningClassifier


class AD(DeepLearningClassifier):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60, sub_seq_len=40):
        super().__init__(learning_rate, epochs, batch_size, random_state, device)
        self.label_num = 2  # 二分类
        self.seq_len = seq_len
        self.sub_seq_len = sub_seq_len

    # 测试数据
    def test_score(self, X, y, loc=None):
        model_param_list = get_lines_from_log(self.model_name, (0, self.epochs))
        val_score_list = list(map(lambda x: x['val_loss'], model_param_list))
        length = 10
        val_score_sum = sum(val_score_list[:length])
        best_val_score_sum = val_score_sum
        epoch_right = length
        for i in range(length, len(val_score_list)):
            val_score_sum = val_score_sum - val_score_list[i - length] + val_score_list[i]
            if best_val_score_sum > val_score_sum:
                best_val_score_sum = val_score_sum
                epoch_right = i + 1
        epoch = epoch_right - length
        for i in range(epoch + 1, epoch_right):
            if val_score_list[epoch] > val_score_list[i]:
                epoch = i
        model_param = model_param_list[epoch]
        log_dir = yaml_config['dir']['log_dir']
        logger = logging_config(self.model_name, log_dir + '/%s.log' % self.model_name)
        logger.info({
            "===================== Test score of the selected model ====================="
        })
        mdl = joblib.load(model_param['model_path'])
        mdl.device = self.device
        test_score = mdl.score(X, y, batch_size=512)
        mdl.metrics_list = self.metrics_list
        test_score_list = mdl.score_list(X, y, loc=loc, batch_size=512)
        test_score_dict = {self.metrics.__name__: test_score}
        for i, metrics in enumerate(self.metrics_list):
            test_score_dict.update({metrics.__name__: test_score_list[i]})

        logger.info({
            "select_epoch": model_param['epoch'],
            "test_score_dict": test_score_dict,
        })

    # 评价指标列表
    def score_list(self, X, y, y_prob=None, loc=None, batch_size=10000):
        score_list = []
        if y_prob is None:
            y_prob = self.predict_proba(X, batch_size)
        y_pred = self.predict(X, y_prob, batch_size)
        for metrics in self.metrics_list:
            if 'auc' in metrics.__name__:
                score = metrics(y, y_prob[:, 1]) if len(y_prob.shape) > 1 else metrics(y, y_prob)
            elif 'f1_location_score' == metrics.__name__:
                score = 0.0 if loc is None else metrics(loc, y, y_pred)
            elif 'recall_location_score' == metrics.__name__:
                score = 0.0 if loc is None else metrics(loc, y_pred)
            else:
                score = metrics(y, y_pred)
            score_list.append(score)
        return score_list
