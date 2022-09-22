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


class MYLSTM(nn.Module):
    def __init__(self, config):
        super(MYLSTM, self).__init__()
        self.config = config
        self.obs_len = 8
        self.pred_len = 12

        self.traj_lstm_model = nn.LSTMCell(2, 8)
        self.pred_lstm_model = nn.LSTMCell(2, 8)
        self.pred_position = nn.Linear(8, 2)

    def forward(self, obs_traj):
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

        for i in range(self.pred_len):
            """(batch, 2)"""
            #input_embedded = output
            #input_cat = torch.cat([lstm_state_context.detach(),output.detach()], dim=-1)
            #inputtest = self.inputLayer_decoder(input_cat)
            #input_embedded = F.relu(input_cat)
            #output = torch.tanh(self.pred_position(pred_lstm_h_t)) * 4.4
            #output = f(x)
            lstm_state = self.pred_lstm_model(output.detach(), (pred_lstm_h_t, pred_lstm_c_t))
            pred_lstm_h_t = lstm_state[0]
            pred_lstm_c_t = lstm_state[1]
            output = torch.tanh(self.pred_position(pred_lstm_h_t)) * 4.4
            pred_traj += [output] #.detach()

        return torch.stack(pred_traj)



class MPIS:
    def __init__(self):
        self.config = config.Config()
        wandb.init(project="mpis", name="lstm", reinit=True)
        torch.manual_seed(42)
        np.random.seed(42)
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"

        _, self.train_loader = data_loader(self.config, "datasets/eth/train/", augment=self.config.augment)
        _, self.val_loader = data_loader(self.config, "datasets/eth/val/")

        print('There are {} iterations per epoch'.format(len(self.train_loader)))

        self.model = MYLSTM(self.config)
        self.model.type(torch.cuda.FloatTensor).train()
        print(self.model)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, betas=(0.5, 0.9999))
        self.loss_function = nn.SmoothL1Loss()

    def test(self):
        self.model = torch.load('model.pkl')
        for batch in self.val_loader:
            batch = [tensor.cuda() for tensor in batch]
            obs_traj, pred_traj_gt, _, _, _, _, _, _, _ = batch
            pred_traj = self.model(obs_traj)
            self.loss = self.loss_function(pred_traj, pred_traj_gt)
            self.loss.backward()
            self.optimizer.step()
            wandb.log({"Loss": self.loss})

    def train(self):
        for i in range(100): #self.config.num_epochs):
            for batch in self.train_loader:
                batch = [tensor.cuda() for tensor in batch]
                obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, _, _, _, _, _ = batch
                for param in self.model.parameters(): param.grad = None
                pred_traj = self.model(obs_traj_rel)
                print("pred traj: ", pred_traj[0])
                print("gt traj: ", pred_traj_gt_rel[0])
                #print("pred grad: ", pred_traj.grad)
                self.loss = self.loss_function(pred_traj, pred_traj_gt_rel)
                print("loss: ", self.loss)
                self.loss.backward()
                self.optimizer.step()
                wandb.log({"Loss": self.loss})
            print(self.loss)
            if i % 20 == 0:
                torch.save(self.model, 'model.pkl')


if __name__ == '__main__':
    mpis = MPIS()
    mpis.train()
