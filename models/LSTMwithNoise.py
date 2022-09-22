import os
import sys
import time
import numpy
import torch
import wandb
import config
import logging
import numpy as np
import torch.nn as nn
from losses import l2_loss
import torch.optim as optim
import torch.nn.functional as F
from data.loader import data_loader


"""
输入数据格式：
rnn = nn.LSTM(input_size,hidden_size,num_layers)
input(seq_len, batch, input_size)
h0(num_layers * num_directions, batch, hidden_size)
c0(num_layers * num_directions, batch, hidden_size)

输出数据格式：
output(seq_len, batch, hidden_size * num_directions)
hn(num_layers * num_directions, batch, hidden_size)
cn(num_layers * num_directions, batch, hidden_size)

"""
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True


def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype


class LSTMwithNoise(nn.Module):
    def __init__(self):
        super(LSTMwithNoise, self).__init__()
        self.obs_len = 8
        self.pred_len = 12

        self.traj_lstm_model = nn.LSTMCell(2, 8)
        self.pred_lstm_model = nn.LSTMCell(2, 8)
        self.pred_position = nn.Linear(8, 2)

    def forward(self, obs_traj, pred_traj_gt):
        batch = obs_traj.shape[1]
        pred_traj = []
        traj_lstm_h_t = torch.randn(batch, 8).cuda()
        traj_lstm_c_t = torch.randn(batch, 8).cuda()

        for i, input_t in enumerate(obs_traj):
            """input_t:(人数, 2), input_embedded:(人数，16)"""
            input_embedded = F.relu(input_t.squeeze(0))
            lstm_state = self.traj_lstm_model(input_embedded, (traj_lstm_h_t, traj_lstm_c_t))
            traj_lstm_h_t, traj_lstm_c_t = lstm_state

        """(batch，hidden_size)"""
        pred_lstm_h_t = traj_lstm_h_t
        pred_lstm_c_t = torch.zeros_like(pred_lstm_h_t).cuda()
        output = torch.zeros_like(obs_traj[self.obs_len - 1]).cuda()
        lstm_state_context = torch.zeros_like(traj_lstm_h_t).cuda()
        noise = torch.randn(output.shape).cuda()
        print("output: ", output.shape)
        print("noise: ", noise.shape)

        for i in range(self.pred_len):
            """(batch, 2)"""
            input = torch.cat([output.detach(), noise], 1)
            lstm_state = self.pred_lstm_model(output + noise, (pred_lstm_h_t, pred_lstm_c_t))
            pred_lstm_h_t = lstm_state[0]
            pred_lstm_c_t = lstm_state[1]
            output = torch.tanh(self.pred_position(pred_lstm_h_t)) * 4.4
            pred_traj += [output] #.detach()

        return torch.stack(pred_traj)