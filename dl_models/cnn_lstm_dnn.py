#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/4/23 22:36
# @Author : LYX-夜光
import torch
from torch import nn

from dl_models import AD


class CNN_LSTM_DNN(AD):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60, sub_seq_len=40):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len, sub_seq_len)
        self.model_name = "cnn_lstm_dnn"

    def create_model(self):
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 4), stride=(1, 4)),
            nn.Tanh(),
        )
        self.lstm = nn.LSTM(input_size=80, hidden_size=64)
        self.dnn = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=64, out_features=32),
            nn.Tanh(),
            nn.Linear(in_features=32, out_features=self.label_num),
        )

    def forward(self, X):
        # 滑动窗口构建矩阵
        X = torch.as_strided(
            X,
            size=(X.shape[0], self.seq_len - self.sub_seq_len + 1, self.sub_seq_len),
            stride=(X.stride(0), X.stride(1), X.stride(1))
        )
        # 构建频率矩阵
        X = torch.abs(torch.fft.rfft(X, dim=-1))
        X = X.unsqueeze(1)
        X = self.cnn(X)
        H = X.permute(2, 0, 1, 3).flatten(2)
        _, (h, _) = self.lstm(H)
        y = self.dnn(h.squeeze(0))
        return y
