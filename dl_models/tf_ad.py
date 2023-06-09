#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/3/1 21:03
# @Author : LYX-夜光
import torch
from torch import nn

from dl_models import AD


class TF_AD(AD):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60, sub_seq_len=30):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len, sub_seq_len)
        self.model_name = "tf_ad_%s_%s" % (seq_len, sub_seq_len)

        self.input_size = int((int(sub_seq_len/2)+1)/4)

    def create_model(self):
        self.cnn_t = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 4), stride=(1, 4)),
            nn.ReLU(),
        )
        self.cnn_f = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=64*self.input_size, hidden_size=64, bidirectional=True)
        self.dnn = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=128, out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=self.label_num),
        )

    def forward(self, X):
        # 滑动窗口构建矩阵
        X_t = torch.as_strided(
            X,
            size=(X.shape[0], self.seq_len - self.sub_seq_len + 1, self.sub_seq_len),
            stride=(X.stride(0), X.stride(1), X.stride(1))
        )
        # 构建频率矩阵
        X_f = torch.abs(torch.fft.rfft(X_t, dim=-1))
        Z_t = self.cnn_t(X_t.unsqueeze(1))
        Z_f = self.cnn_f(X_f.unsqueeze(1))
        Z_t = Z_t.permute(2, 0, 1, 3).flatten(2)
        Z_f = Z_f.permute(2, 0, 1, 3).flatten(2)
        Z = torch.cat([Z_t, Z_f], dim=2)
        _, (h, _) = self.lstm(Z)
        y = torch.cat(list(h), dim=-1)
        y = self.dnn(y)
        return y


class T_AD(AD):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60, sub_seq_len=30):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len, sub_seq_len)
        self.model_name = "t_ad"

        self.input_size = int((int(sub_seq_len / 2) + 1) / 4)

    def create_model(self):
        self.cnn_t = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 4), stride=(1, 4)),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=32*self.input_size, hidden_size=64, bidirectional=True)
        self.dnn = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=128, out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=self.label_num),
        )

    def forward(self, X):
        # 滑动窗口构建矩阵
        X = torch.as_strided(
            X,
            size=(X.shape[0], self.seq_len - self.sub_seq_len + 1, self.sub_seq_len),
            stride=(X.stride(0), X.stride(1), X.stride(1))
        )
        # 构建频率矩阵
        Z = self.cnn_t(X.unsqueeze(1))
        Z = Z.permute(2, 0, 1, 3).flatten(2)
        _, (h, _) = self.lstm(Z)
        y = torch.cat(list(h), dim=-1)
        y = self.dnn(y)
        return y

class F_AD(AD):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60, sub_seq_len=30):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len, sub_seq_len)
        self.model_name = "f_ad"

        self.input_size = int((int(sub_seq_len/2)+1)/4)

    def create_model(self):
        self.cnn_f = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=32*self.input_size, hidden_size=64, bidirectional=True)
        self.dnn = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=128, out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=self.label_num),
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
        Z = self.cnn_f(X.unsqueeze(1))
        Z = Z.permute(2, 0, 1, 3).flatten(2)
        _, (h, _) = self.lstm(Z)
        y = torch.cat(list(h), dim=-1)
        y = self.dnn(y)
        return y