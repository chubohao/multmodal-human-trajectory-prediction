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


class Encoder(nn.Module):
    def __init__(self, feature_size, lstm_input_size, lstm_hidden_size):
        super(Encoder, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.inputLayer_encoder = nn.Linear(feature_size, lstm_input_size)
        self.obs_lstm = nn.LSTMCell(lstm_input_size, lstm_hidden_size)
        self.pre_lstm = nn.LSTMCell(lstm_input_size, lstm_hidden_size)

    def forward(self, obs_traj_rel, pre_traj_rel):
        batch = obs_traj_rel.shape[1]
        hidden_h = torch.randn(batch, self.lstm_hidden_size).cuda()
        hidden_c = torch.randn(batch, self.lstm_hidden_size).cuda()

        for i, input_t in enumerate(obs_traj_rel):
            input_t = input_t.squeeze(0)
            input_embedded = self.inputLayer_encoder(input_t)
            lstm_state = self.obs_lstm(input_embedded, (hidden_h, hidden_c))
            hidden_h, hidden_c = lstm_state

        c = hidden_h.permute(1, 0).reshape(batch, -1) # (batch, lstm_hidden_size)

        hidden_c = torch.randn(batch, self.lstm_hidden_size).cuda()
        for i, input_t in enumerate(pre_traj_rel):
            input_embedded = self.inputLayer_encoder(input_t)
            lstm_state = self.pre_lstm(input_embedded, (hidden_h, hidden_c))
            hidden_h, hidden_c = lstm_state

        x = hidden_h.permute(1, 0).reshape(batch, -1) # (batch, lstm_hidden_size)

        return c, x, batch


class PriorNet(nn.Module):
    def __init__(self, lstm_output_size, latent_size):
        super(PriorNet, self).__init__()
        self.latent_size = latent_size
        self.separate = nn.Linear(lstm_output_size, 2 * self.latent_size)

    def forward(self, c):
        h = self.separate(c)
        prior_mu = h[:, :self.latent_size]
        prior_logvar = h[:, self.latent_size:]
        return prior_mu, prior_logvar


class RecognizeNet(nn.Module):
    def __init__(self, lstm_hidden_size, latent_size):
        super(RecognizeNet, self).__init__()
        self.latent_size = latent_size
        self.separate = nn.Linear(2*lstm_hidden_size, 2*latent_size)

    def forward(self, c, x):
        h = self.separate(torch.cat((x, c), 1))
        recog_mu = h[:, :self.latent_size]
        recog_logvar = h[:, self.latent_size:]

        return recog_mu, recog_logvar



class Decoder(nn.Module):
    def __init__(self, latent_size, feature_size, condition_size, decoder_lstm_input_size, lstm_hidden_size = 8, context_state_size = 8):
        super(Decoder, self).__init__()
        #lstm_state, prev_pos, c, z,
        self.latent_size = latent_size
        self.context_state_size = context_state_size
        self.feature_size = feature_size
        self.lstm_input_size = decoder_lstm_input_size
        self.condition_size = condition_size
        self.lstm_hidden_size = lstm_hidden_size
        self.half_lstm_hidden_size = int(self.lstm_hidden_size / 2)
        # c, z, p, polling
        self.input_decoder = nn.Linear(self.condition_size + self.latent_size + self.context_state_size + self.feature_size, self.lstm_input_size)
        self.position_lstm = nn.LSTMCell(self.lstm_input_size, self.lstm_hidden_size)

        self.latent_to_position = nn.Linear(self.half_lstm_hidden_size + context_state_size, feature_size)
        #lstm_state_size
        self.pl_net = Pooling_net(h_dim=self.context_state_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def plot_variance(self, mean, logvar, past_postion):
        for i in range(20):
            p = past_postion[:, i, :]
            x = p[:, 0].cpu()
            y = p[:, 1].cpu()
            plt.plot(x, y, "o-", c='grey', markersize=4)
        curr_pos = past_postion[-1]
        for i in range(3):
            for i in range(20):
                index = i
                x = []
                y = []
                curr_pos_abs = curr_pos
                x += [curr_pos_abs[index, 0].item()]
                y += [curr_pos_abs[index, 1].item()]
                for i in range(12):
                    mean_i = mean[i]
                    sigma_i = logvar[i]
                    position = self.reparameterizes(mean_i, sigma_i, i)
                    output = position
                    curr_pos_abs = (curr_pos_abs + output)
                    x += [curr_pos_abs[index, 0].item()]
                    y += [curr_pos_abs[index, 1].item()]
                plt.plot(x, y, '-', markersize=2)

        """
        p = past_postion[:, index, :]
        x = p[:, 0].cpu()
        y = p[:, 1].cpu()
        plt.plot(x, y, "o-", c='grey', markersize=4)
        minx_o = min(x)
        miny_o = min(y)
        maxx_o = max(x)
        maxy_o = max(y)
        curr_pos = past_postion[-1]
        for i in range(50):
            x = []
            y = []
            curr_pos_abs = curr_pos
            x += [curr_pos_abs[index, 0].item()]
            y += [curr_pos_abs[index, 1].item()]
            for i in range(12):
                mean_i = mean[i]
                sigma_i = logvar[i]
                position = self.reparameterize(mean_i, sigma_i)
                output = position
                curr_pos_abs = (curr_pos_abs + output)
                x += [curr_pos_abs[index, 0].item()]
                y += [curr_pos_abs[index, 1].item()]
            plt.plot(x, y, '.', markersize=2)
            minx_p = min(x)
            miny_p = min(y)
            maxx_p = max(x)
            maxy_p = max(y)

        index = 5
        p = past_postion[:, index, :]
        x = p[:, 0].cpu()
        y = p[:, 1].cpu()
        plt.plot(x, y, "o-", c='grey', markersize=4)
        minx_o = min(x)
        miny_o = min(y)
        maxx_o = max(x)
        maxy_o = max(y)
        curr_pos = past_postion[-1]
        for i in range(50):
            x = []
            y = []
            curr_pos_abs = curr_pos
            x += [curr_pos_abs[index, 0].item()]
            y += [curr_pos_abs[index, 1].item()]
            for i in range(12):
                mean_i = mean[i]
                sigma_i = logvar[i]
                position = self.reparameterize(mean_i, sigma_i)
                output = position
                curr_pos_abs = (curr_pos_abs + output)
                x += [curr_pos_abs[index, 0].item()]
                y += [curr_pos_abs[index, 1].item()]
            plt.plot(x, y, '.', markersize=2)
            minx_p = min(x)
            miny_p = min(y)
            maxx_p = max(x)
            maxy_p = max(y)

        plt.xlim(min(minx_o, minx_p) - 1, max(maxx_o, maxx_p) + 1)
        plt.ylim(min(miny_o, miny_p) - 1, max(maxy_o, maxy_p) + 1)
        """
        plt.show()

    def forward(self, last_position, c, z, obs_traj_pos, nei_index, nei_num_index):
        pre_position_rel = []
        batch = obs_traj_pos.shape[1]
        curr_pos_abs = obs_traj_pos[-1]
        pre_distribution_h = torch.randn(batch, self.lstm_hidden_size).cuda()
        pre_distribution_c = torch.randn(batch, self.lstm_hidden_size).cuda()

        lstm_state_context = torch.zeros((batch, self.context_state_size)).cuda()

        """取第7帧坐标(batch,2)"""
        previous_postion = last_position
        init = False
        for i in range(12):
            embedded_input = torch.cat([previous_postion.detach(), c.detach(), z.detach(), lstm_state_context.detach()], dim=-1)
            embedded_input = F.relu(self.input_decoder(embedded_input))
            pre_distribution_state = self.position_lstm(embedded_input, (pre_distribution_h, pre_distribution_c))
            pre_distribution_h = pre_distribution_state[0]  # (batch, 8)
            pre_distribution_c = pre_distribution_state[1]  # (batch, 8)

            """(batch,batch,2)"""
            corr = curr_pos_abs.repeat(batch, 1, 1)
            corr_index = (corr.transpose(0, 1) - corr)
            lstm_state_hidden = pre_distribution_state[0]
            lstm_state_context, _, _ = self.pl_net(corr_index, nei_index[i], nei_num_index, lstm_state_hidden, curr_pos_abs)

            #concat_output = lstm_state_context + lstm_state_hidden

            pre_mean = pre_distribution_h[:, :self.half_lstm_hidden_size,]
            pre_logvar = pre_distribution_h[:, self.half_lstm_hidden_size:]

            pre_mean_context = torch.cat([pre_mean, lstm_state_context], dim=-1)
            pre_logvar_context = torch.cat([pre_logvar, lstm_state_context], dim=-1)

            mu = self.latent_to_position(pre_mean_context)           # (batch, 2)
            logvar = self.latent_to_position(pre_logvar_context)      # (batch, 2)


            position = self.reparameterize(mu, logvar)
            output = position
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

        self.withPrior = False
        self.feature_size = 2
        self.latent_size = 40
        self.condition_size = 16
        self.encoder_lstm_input_size = 16
        self.decoder_lstm_input_size = 16
        self.encoder = Encoder(self.feature_size, self.encoder_lstm_input_size, self.condition_size)
        self.prior = PriorNet(self.condition_size, self.latent_size)
        self.recognize = RecognizeNet(self.condition_size, self.latent_size)
        self.decoder = Decoder(self.latent_size, self.feature_size, self.condition_size, self.decoder_lstm_input_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, obs_traj_rel, pre_traj_rel, obs_traj_pos, nei_index, nei_num_index, predict=False):

        c, x, batch= self.encoder(obs_traj_rel, pre_traj_rel)

        if not predict:
            recognise_mu, recognise_logvar = self.recognize(c, x)
            z = self.reparameterize(recognise_mu, recognise_logvar)
            if self.withPrior:
                prior_mu, prior_logvar = self.prior(c)
                z_prior = self.reparameterize(prior_mu, prior_logvar)
            else:
                prior_mu = torch.zeros_like(z).cuda()
                prior_logvar = torch.zeros_like(z).cuda()
                z_prior = torch.randn(batch, self.latent_size).cuda()

            position, pred_mean, pred_var = self.decoder(obs_traj_rel[-1], c, z, obs_traj_pos, nei_index, nei_num_index)
            return position, prior_mu, prior_logvar, recognise_mu, recognise_logvar, pred_mean, pred_var, z, z_prior
        else:
            if self.withPrior:
                prior_mu, prior_logvar = self.prior(c)
                z_prior = self.reparameterize(prior_mu, prior_logvar)
            else:
                z_prior = torch.randn(batch, self.latent_size).cuda()

            position, pred_mean, pred_var = self.decoder(obs_traj_rel[-1], c, z_prior, obs_traj_pos, nei_index, nei_num_index)
            return position, None, None, None, None, None, None, None, None
