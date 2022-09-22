import os
import time
import torch
import wandb
import config
import numpy as np
import torch.utils.data
from torch import nn, optim
from data.loader import data_loader
from torch.nn import functional as F
import matplotlib.pyplot as plt
from mpis_evaluate import evaluate
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True


def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


class Pooling_net(nn.Module):
    def __init__(self, embedding_dim=32, h_dim=32,activation='relu', batch_norm=False, dropout=0.0):
        super(Pooling_net, self).__init__()
        self.h_dim = h_dim
        self.bottleneck_dim = h_dim
        self.embedding_dim = embedding_dim

        self.mlp_pre_dim = embedding_dim + h_dim * 2
        self.mlp_pre_pool_dims = [self.mlp_pre_dim, 64, self.bottleneck_dim]
        self.attn = nn.Linear(self.bottleneck_dim, 1)
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.mlp_pre_pool = make_mlp(
            self.mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)

    def forward(self, corr_index, nei_index, nei_num, lstm_state, curr_pos_abs, plot_att=False):
        self.N = corr_index.shape[0]
        hj_t = lstm_state.unsqueeze(0).expand(self.N, self.N, self.h_dim)
        hi_t = lstm_state.unsqueeze(1).expand(self.N, self.N, self.h_dim)
        nei_index_t = nei_index.view((-1))
        corr_t = corr_index.reshape((self.N * self.N, -1))
        r_t = self.spatial_embedding(corr_t[nei_index_t > 0])
        mlp_h_input = torch.cat((r_t, hj_t[nei_index > 0], hi_t[nei_index > 0]), 1)
        curr_pool_h = self.mlp_pre_pool(mlp_h_input)
        # Message Passing
        H = torch.full((self.N * self.N, self.bottleneck_dim), -np.Inf, device=torch.device("cuda"),dtype=curr_pool_h.dtype)
        H[nei_index_t > 0] = curr_pool_h
        pool_h = H.view(self.N, self.N, -1).max(1)[0]
        pool_h[pool_h == -np.Inf] = 0.
        return pool_h, (0, 0, 0), 0


class MYLSTM(nn.Module):
    def __init__(self):
        super(MYLSTM, self).__init__()
        self.obs_len = 8
        self.pred_len = 12

        self.traj_lstm_model = nn.LSTMCell(16, 8)
        self.pred_lstm_model = nn.LSTMCell(16, 8)
        self.pred_position = nn.Linear(8, 2)

        self.latent_to_position_mean = nn.Linear(12, 2)
        self.latent_to_position_var = nn.Linear(12, 2)
        self.pl_net = Pooling_net(h_dim=8)
        self.inputLayer_decoder = nn.Linear(10, 16)
        self.inputLayer_encoder = nn.Linear(2, 16)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, obs_traj, obs_traj_obs, nei_index, nei_num_index):
        batch = obs_traj.shape[1]
        curr_pos_abs = obs_traj_obs[-1]
        pred_traj = []
        traj_lstm_h_t = torch.randn(batch, 8).cuda()
        traj_lstm_c_t = torch.randn(batch, 8).cuda()

        for i, input_t in enumerate(obs_traj):
            """input_t:(人数, 2), input_embedded:(人数，16)"""
            input_embedded = F.relu(self.inputLayer_encoder(input_t.squeeze(0)))
            lstm_state = self.traj_lstm_model(input_embedded, (traj_lstm_h_t, traj_lstm_c_t))
            traj_lstm_h_t, traj_lstm_c_t = lstm_state

        """(batch，hidden_size)"""
        pred_lstm_h_t = traj_lstm_h_t
        pred_lstm_c_t = torch.zeros_like(pred_lstm_h_t).cuda()
        output = torch.zeros_like(obs_traj[self.obs_len - 1]).cuda()
        lstm_state_context = torch.zeros_like(traj_lstm_h_t).cuda()
        init = False
        for i in range(self.pred_len):
            """(batch, 2)"""
            input_cat = torch.cat([lstm_state_context.detach(), output.detach()], dim=-1)
            """(batch, 16)"""
            input_embedded = F.relu(self.inputLayer_decoder(input_cat))  # detach from history as input
            lstm_state = self.pred_lstm_model(input_embedded, (pred_lstm_h_t, pred_lstm_c_t))
            pred_lstm_h_t, pred_lstm_c_t = lstm_state

            """(batch,batch,2)"""
            corr = curr_pos_abs.repeat(batch, 1, 1)
            corr_index = (corr.transpose(0, 1) - corr)
            lstm_state_hidden = pred_lstm_h_t
            lstm_state_context, _, _ = self.pl_net(corr_index, nei_index[i], nei_num_index, lstm_state_hidden, curr_pos_abs)


            pre_mean = pred_lstm_h_t[:, :4]
            pre_logvar = pred_lstm_h_t[:, 4:]

            pre_mean = torch.cat([pre_mean, lstm_state_context], dim=-1) # (batch, 12)
            pre_logvar = torch.cat([pre_logvar, lstm_state_context], dim=-1) # (batch, 12)

            mu = self.latent_to_position_mean(pre_mean)  # (batch, 2)
            logvar = self.latent_to_position_var(pre_logvar)  # (batch, 2)


            position = self.reparameterize(mu, logvar)
            output = position
            pred_traj += [output]  # .detach()

            if not init:
                mean = mu.unsqueeze(0)
                var = logvar.unsqueeze(0)
                init = True
            else:
                mean = torch.cat((mean, mu.unsqueeze(0)), 0)
                var = torch.cat((var, logvar.unsqueeze(0)), 0)

        return torch.stack(pred_traj), mean, var

