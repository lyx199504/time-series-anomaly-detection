#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/11/25 11:49
# @Author : LYX-夜光
import torch
from torch import nn

from dl_models import AD


class C_LSTM_AE(AD):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60, sub_seq_len=10):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len, sub_seq_len)
        self.model_name = "c_lstm_ae"

    def create_model(self):
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )
        self.lstm_encoder = nn.LSTM(input_size=64, hidden_size=64)
        self.lstm_decoder = nn.LSTM(input_size=64, hidden_size=128)
        self.dnn = nn.Sequential(
            nn.Linear(in_features=128, out_features=510),
            nn.ReLU(),
            nn.Linear(in_features=510, out_features=self.label_num),
        )

    def forward(self, X):
        # 滑动窗口构建矩阵
        X = torch.as_strided(
            X,
            size=(X.shape[0], self.seq_len - self.sub_seq_len + 1, self.sub_seq_len),
            stride=(X.stride(0), X.stride(1), X.stride(1))
        )
        X = X.unsqueeze(1)
        X = self.conv1(X)
        X = self.conv2(X)
        X = X.permute(2, 0, 1, 3).flatten(2)
        H, _ = self.lstm_encoder(X)
        _, (h, _) = self.lstm_decoder(H)
        y = self.dnn(h.squeeze(0))
        return y

