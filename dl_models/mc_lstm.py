#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/2/26 21:02
# @Author : LYX-夜光
import torch
from torch import nn

from dl_models import AD


class IMC_LSTM(AD):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60, conv_type=3, conv_num=48):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len)
        self.model_name = "imc_lstm"

        self.conv_type = conv_type
        self.conv_num = conv_num

    def create_model(self):
        self.cnn_list, self.lstm_list = nn.ModuleList([]), nn.ModuleList([])
        for i in range(self.conv_type):
            kernel_size = i*2 + 1
            padding = i
            pooling_size = 3
            self.cnn_list.append(
                nn.Sequential(
                    nn.Conv1d(1, self.conv_num, kernel_size=kernel_size, stride=1, padding=padding),
                    nn.Tanh(),
                    nn.MaxPool1d(kernel_size=pooling_size, stride=pooling_size),
                )
            )
            self.lstm_list.append(
                nn.LSTM(input_size=self.conv_num, hidden_size=self.conv_num),
            )
        self.dnn = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.conv_type*self.conv_num, out_features=self.conv_num),
            nn.Tanh(),
            nn.Linear(in_features=self.conv_num, out_features=self.label_num),
        )

    def forward(self, X):
        X = X.unsqueeze(1)
        H = []
        for cnn, lstm in zip(self.cnn_list, self.lstm_list):
            X_ = cnn(X).permute(2, 0, 1)
            _, (h, _) = lstm(X_)
            H.append(h.squeeze(0))
        H = torch.cat(H, dim=1)
        y = self.dnn(H)
        return y


class CMC_LSTM(AD):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60, conv_type=3, conv_num=32):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len)
        self.model_name = "cmc_lstm"

        self.conv_type = conv_type
        self.conv_num = conv_num

    def create_model(self):
        self.cnn_list = nn.ModuleList([])
        for i in range(self.conv_type):
            kernel_size = i*2 + 1
            padding = i
            pooling_size = 3
            self.cnn_list.append(
                nn.Sequential(
                    nn.Conv1d(1, self.conv_num, kernel_size=kernel_size, stride=1, padding=padding),
                    nn.Tanh(),
                    nn.MaxPool1d(kernel_size=pooling_size, stride=pooling_size),
                )
            )
        self.lstm = nn.LSTM(input_size=self.conv_type*self.conv_num, hidden_size=self.conv_type*self.conv_num)
        self.dnn = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.conv_type*self.conv_num, out_features=self.conv_num),
            nn.Tanh(),
            nn.Linear(in_features=self.conv_num, out_features=self.label_num),
        )

    def forward(self, X):
        X = X.unsqueeze(1)
        Z = []
        for cnn in self.cnn_list:
            Z.append(cnn(X).permute(2, 0, 1))
        Z = torch.cat(Z, dim=-1)
        _, (h, _) = self.lstm(Z)
        y = self.dnn(h.squeeze(0))
        return y


class SMC_LSTM(AD):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu', seq_len=60, conv_type=3, conv_num=32):
        super().__init__(learning_rate, epochs, batch_size, random_state, device, seq_len)
        self.model_name = "smc_lstm"

        self.conv_type = conv_type
        self.conv_num = conv_num

    def create_model(self):
        self.cnn_list = nn.ModuleList([])
        for i in range(self.conv_type):
            kernel_size = i*2 + 1
            padding = i
            pooling_size = 3
            self.cnn_list.append(
                nn.Sequential(
                    nn.Conv1d(1, self.conv_num, kernel_size=kernel_size, stride=1, padding=padding),
                    nn.Tanh(),
                    nn.MaxPool1d(kernel_size=pooling_size, stride=pooling_size),
                )
            )
        self.lstm = nn.LSTM(input_size=self.conv_num, hidden_size=self.conv_num)
        self.dnn = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.conv_type*self.conv_num, out_features=self.conv_num),
            nn.Tanh(),
            nn.Linear(in_features=self.conv_num, out_features=self.label_num),
        )

    def forward(self, X):
        X = X.unsqueeze(1)
        H = []
        for cnn in self.cnn_list:
            X_ = cnn(X).permute(2, 0, 1)
            _, (h, _) = self.lstm(X_)
            H.append(h.squeeze(0))
        H = torch.cat(H, dim=1)
        y = self.dnn(H)
        return y
