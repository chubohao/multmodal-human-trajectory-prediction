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
from gaussian_distribution import Gaussian_Distribution
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


class RecognizeNet(nn.Module):
    def __init__(self):
        super(RecognizeNet, self).__init__()
        self.separate = nn.Linear(16, 4)

    def forward(self, c, x):
        input = torch.cat((x, c), 1)  # (batch, 16)
        h = self.separate(input)
        mu = h[:,:2]
        logvar = h[:,2:]

        return mu, logvar  # (batch, 2)


class PriorNet(nn.Module):
    def __init__(self):
        super(PriorNet, self).__init__()
        self.separate = nn.Linear(8, 4)

    def forward(self, c):
        h = self.separate(c)
        mu = h[:, :2]
        logvar = h[:, 2:]
        return mu, logvar  # (batch, 2)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.inputLayer_encoder = nn.Linear(2, 16)
        self.obs_lstm = nn.LSTMCell(16, 8)
        self.pre_lstm = nn.LSTMCell(16, 8)

    def forward(self, obs_traj_rel, pre_traj_rel):
        batch = obs_traj_rel.shape[1]
        hidden_h = torch.randn(batch, 8).cuda()
        hidden_c = torch.randn(batch, 8).cuda()

        for i, input_t in enumerate(obs_traj_rel):
            input_embedded = self.inputLayer_encoder(input_t.squeeze(0))
            lstm_state = self.obs_lstm(input_embedded, (hidden_h, hidden_c))
            hidden_h, hidden_c = lstm_state

        c = hidden_h.permute(1, 0).reshape(batch, -1)  # (batch, 8)

        hidden_c = torch.randn(batch, 8).cuda()
        for i, input_t in enumerate(pre_traj_rel):
            input_embedded = self.inputLayer_encoder(input_t.squeeze(0))
            lstm_state = self.pre_lstm(input_embedded, (hidden_h, hidden_c))
            hidden_h, hidden_c = lstm_state

        x = hidden_h.permute(1, 0).reshape(batch, -1) # (batch, 8)

        return c, x



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.input_decoder = nn.Linear(20, 16)
        self.var_lstm = nn.LSTMCell(16, 8)
        self.mean_lstm = nn.LSTMCell(16, 8)
        self.latent_to_position_mean = nn.Linear(12, 2)
        self.latent_to_position_var = nn.Linear(12, 2)
        self.pl_net = Pooling_net(h_dim=8)
        self.z_to_h = nn.Linear(2, 8)
        self.p2p = nn.Linear(2,2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, p, c, z, obs_traj_pos, nei_index, nei_num_index):
        batch = obs_traj_pos.shape[1]
        pre_position_rel = []
        curr_pos_abs = obs_traj_pos[-1]
        pre_distribution_h = self.z_to_h(z)
        pre_distribution_c = torch.randn(batch, 8).cuda()
        lstm_state_context = torch.zeros_like(pre_distribution_h).cuda()

        """取第7帧坐标(batch,2)"""
        previous_postion = p
        init = False
        for i in range(12):
            # 8282
            embedded_input = torch.cat([lstm_state_context.detach(), previous_postion.detach(), c.detach(), z.detach()], dim=-1)
            embedded_input = F.relu(self.input_decoder(embedded_input))
            pre_distribution_state = self.mean_lstm(embedded_input, (pre_distribution_h, pre_distribution_c))
            pre_distribution_h = pre_distribution_state[0]  # (batch, 8)
            pre_distribution_c = pre_distribution_state[1]  # (batch, 8)

            """(batch,batch,2)"""
            corr = curr_pos_abs.repeat(batch, 1, 1)
            corr_index = (corr.transpose(0, 1) - corr)
            lstm_state_hidden = pre_distribution_state[0]
            lstm_state_context, _, _ = self.pl_net(corr_index, nei_index[i], nei_num_index, lstm_state_hidden, curr_pos_abs)

            #concat_output = lstm_state_context + lstm_state_hidden

            pre_mean = pre_distribution_h[:, :4]
            pre_logvar = pre_distribution_h[:, 4:]

            pre_mean = torch.cat([pre_mean, lstm_state_context], dim=-1)
            pre_logvar = torch.cat([pre_logvar, lstm_state_context], dim=-1)

            mu = self.latent_to_position_mean(pre_mean)           # (batch, 2)
            logvar = self.latent_to_position_var(pre_logvar)      # (batch, 2)

            position = self.reparameterize(mu, logvar)
            output = position
            #output = position
            previous_postion = output
            curr_pos_abs = (curr_pos_abs + output).detach()  # detach from history as input

            if not init:
                mean = mu.unsqueeze(0)
                var = logvar.unsqueeze(0)
                init = True
            else:
                mean = torch.cat((mean, mu.unsqueeze(0)), 0)
                var = torch.cat((var, logvar.unsqueeze(0)), 0)

            # Predicted postion
            pre_position_rel += [output]
        return torch.stack(pre_position_rel), mean, var


class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.encoder = Encoder()
        self.prior = PriorNet()
        self.recognize = RecognizeNet()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, obs_traj_rel, pre_traj_rel, obs_traj_pos, nei_index, nei_num_index, predict=False):

        c, x = self.encoder(obs_traj_rel, pre_traj_rel)

        if not predict:
            # run it when predict is true
            recognise_mu, recognise_logvar = self.recognize(c, x)   # (batch, 2)
            prior_mu, prior_logvar = self.prior(c)                  # (batch, 2)
            z = self.reparameterize(recognise_mu, recognise_logvar) # (bathc, 2)
            # here is a additional parameter groundtruth in the self.decoder
            position, pred_mean, pred_var = self.decoder(obs_traj_rel[-1], c, z, obs_traj_pos, nei_index, nei_num_index)
            return position, prior_mu, prior_logvar, recognise_mu, recognise_logvar, pred_mean, pred_var
        else:
            # run it when predict is flase
            prior_mu, prior_logvar = self.prior(c)
            z = self.reparameterize(prior_mu, prior_logvar)
            # here is no additional parameter groundtruth in the self.decoder
            position, pred_mean, pred_var = self.decoder(obs_traj_rel[-1], c, z, obs_traj_pos, nei_index, nei_num_index)
            return position, prior_mu, prior_logvar, None, None, pred_mean, pred_var
