#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/10/24 15:17
# @Author : LYX-夜光
import torch
from torch import nn

from dl_models import AD


class LSTM_T_1(AD):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len)
        self.model_name = "lstm_t_1"

        self.input_size = 1

    def create_model(self):
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=32)
        self.linear = nn.Linear(in_features=32, out_features=self.label_num)

    def forward(self, X):
        X = X.unsqueeze(-1)
        X = X.permute(1, 0, 2)
        _, (h, _) = self.lstm(X)
        y = self.linear(h.squeeze(0))
        return y

class LSTM_F_1(LSTM_T_1):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len)
        self.model_name = "lstm_f_1"

    def forward(self, X):
        # 频率
        X = torch.abs(torch.fft.rfft(X, dim=-1))
        X = X.unsqueeze(-1)
        X = X.permute(1, 0, 2)
        _, (h, _) = self.lstm(X)
        y = self.linear(h.squeeze(0))
        return y

class LSTM_T_2(AD):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60, sub_seq_len=40):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len, sub_seq_len)
        self.model_name = "lstm_t_2"

        self.input_size = self.sub_seq_len

    def create_model(self):
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=32)
        self.linear = nn.Linear(in_features=32, out_features=self.label_num)

    def forward(self, X):
        # 滑动窗口构建矩阵
        X = torch.as_strided(
            X,
            size=(X.shape[0], self.seq_len - self.sub_seq_len + 1, self.sub_seq_len),
            stride=(X.stride(0), X.stride(1), X.stride(1))
        )
        X = X.permute(1, 0, 2)
        _, (h, _) = self.lstm(X)
        y = self.linear(h.squeeze(0))
        return y

class LSTM_F_2(LSTM_T_2):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60, sub_seq_len=40):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len, sub_seq_len)
        self.model_name = "lstm_f_2" if self.sub_seq_len == 40 else "lstm_%d" % self.sub_seq_len

        self.input_size = int(self.sub_seq_len/2)+1

    def forward(self, X):
        # 滑动窗口构建矩阵
        X = torch.as_strided(
            X,
            size=(X.shape[0], self.seq_len - self.sub_seq_len + 1, self.sub_seq_len),
            stride=(X.stride(0), X.stride(1), X.stride(1))
        )
        # 构建频率矩阵
        X = torch.abs(torch.fft.rfft(X, dim=-1))
        X = X.permute(1, 0, 2)
        _, (h, _) = self.lstm(X)
        y = self.linear(h.squeeze(0))
        return y