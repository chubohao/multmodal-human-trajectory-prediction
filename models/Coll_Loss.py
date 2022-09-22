import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import *
config = Config()


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
    def __init__(
            self, embedding_dim=32, h_dim=32,
            activation='relu', batch_norm=False, dropout=0.0
    ):
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

        #amount of persons in scene
        self.N = corr_index.shape[0]
        #print("N: ", self.N)
        #hidden states used for computations
        #unsqueeze adds dimenions at index
        #expands the matrix to given size filled with copied prev elements
        hj_t = lstm_state.unsqueeze(0).expand(self.N, self.N, self.h_dim)
        hi_t = lstm_state.unsqueeze(1).expand(self.N, self.N, self.h_dim)
        # nei index holds neighbors of person
        #neighbors[0,1] is person1 neighbor of person0: 1 -> neighbor, 0-> not neighbors
        #reduce by 1 dimesnion to list
        #print("nei index: ", nei_index)
        nei_index_t = nei_index.view((-1))
        #print("nei index: ", nei_index_t)
        #print("corr_index: ", corr_index)
        corr_t = corr_index.reshape((self.N * self.N, -1))
        #print("corr_t: ", corr_t)
        #what is spatial embedding needed for? 2 to 16dim?!
        #for all neighbors
        r_t = self.spatial_embedding(corr_t[nei_index_t > 0])
        #concatenate neighbor correlation, and both hidden states
        mlp_h_input = torch.cat((r_t, hj_t[nei_index > 0], hi_t[nei_index > 0]), 1)
        #put into mlp
        curr_pool_h = self.mlp_pre_pool(mlp_h_input)
        # Message Passing
        #fill matrix with -inf (same size as hiddenstates before, bottneck = hdim)
        H = torch.full((self.N * self.N, self.bottleneck_dim), -np.Inf, device=torch.device("cuda"),dtype=curr_pool_h.dtype)
        #set value to computed in mlp for neighbors
        H[nei_index_t > 0] = curr_pool_h
        #reduces H dimension and returns maximum along dimension 1 and then first entry?
        pool_h = H.view(self.N, self.N, -1).max(1)[0]
        #print("pool_h ", pool_h)
        #also nur noch nullen?
        pool_h[pool_h == -np.Inf] = 0.
        return pool_h, (0, 0, 0), 0

def get_noise(shape, noise_type):
    if noise_type == 'gaussian':
        return torch.randn(*shape).to('cuda')
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).to('cuda')
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


class CoLoss(nn.Module):
    def __init__(self,config):
        #initialize coloss model
        super(CoLoss, self).__init__()
        self.config = config
        #observe 8
        obs_len= config.obs_len
        #predict 12
        pred_len= config.pred_len
        #2 probably for x and y coords
        traj_lstm_input_size= 2
        #what is the hidden size where in the paper is it define? was 16 before but "fixed" by me to 8
        #is the hidden size the social context? limited to 8 or 16 because of max amount of people in scene?
        traj_lstm_hidden_size=8 #16
        #relative displacements embedded?
        rela_embed_size = 8


        #linear transformation on incoming data with Linear(input size, output size)
        #encoder input size 2: x and y
        self.inputLayer_encoder = nn.Linear(traj_lstm_input_size, rela_embed_size)
        #self.inputLayer_decoder = nn.Linear(traj_lstm_input_size + 16, rela_embed_size)
        #decoder input size 10: 2 + hidden?
        self.inputLayer_decoder = nn.Linear(traj_lstm_input_size + 8, rela_embed_size)
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.pl_net = Pooling_net(h_dim=traj_lstm_hidden_size)

        self.traj_lstm_hidden_size = traj_lstm_hidden_size
        self.traj_lstm_input_size = traj_lstm_input_size
        self.pred_lstm_hidden_size = self.traj_lstm_hidden_size
        #self.traj_lstm_model = nn.LSTMCell(rela_embed_size, 16)
        self.traj_lstm_model = nn.LSTMCell(rela_embed_size, 8)
        self.pred_lstm_model = nn.LSTMCell(rela_embed_size, 8)
        self.pred_hidden2pos = nn.Linear(traj_lstm_hidden_size, 2)

        self.init_parameters()


    def init_parameters(self):
        nn.init.constant_(self.inputLayer_encoder.bias, 0.0)
        nn.init.normal_(self.inputLayer_encoder.weight, std=self.config.std_in)

        nn.init.constant_(self.inputLayer_decoder.bias, 0.0)
        nn.init.normal_(self.inputLayer_decoder.weight, std=self.config.std_in)

        nn.init.xavier_uniform_(self.traj_lstm_model.weight_ih)
        nn.init.orthogonal_(self.traj_lstm_model.weight_hh, gain=0.001)

        nn.init.constant_(self.traj_lstm_model.bias_ih, 0.0)
        nn.init.constant_(self.traj_lstm_model.bias_hh, 0.0)
        n = self.traj_lstm_model.bias_ih.size(0)
        nn.init.constant_(self.traj_lstm_model.bias_ih[n // 4:n // 2], 1.0)

        nn.init.xavier_uniform_(self.pred_lstm_model.weight_ih)
        nn.init.orthogonal_(self.pred_lstm_model.weight_hh, gain=0.001)

        nn.init.constant_(self.pred_lstm_model.bias_ih, 0.0)
        nn.init.constant_(self.pred_lstm_model.bias_hh, 0.0)
        n = self.pred_lstm_model.bias_ih.size(0)
        nn.init.constant_(self.pred_lstm_model.bias_ih[n // 4:n // 2], 1.0)

        nn.init.constant_(self.pred_hidden2pos.bias, 0.0)
        nn.init.normal_(self.pred_hidden2pos.weight, std=self.config.std_out)

    def init_hidden_traj_lstm(self, batch):
        return (
            torch.randn(batch, 8).cuda(),
            torch.randn(batch, 8).cuda(),
        )


    def forward(self, traj_rel, obs_traj_pos, pred_traj_gt_pos, seq_start_end,
                nei_index, nei_num_index):

        #relative trajectory has shape 20, 330, 2  GUESS: ->20 tupels for every person,330 persons total, 2 values per person:x and y
        #batch = 330
        batch = traj_rel.shape[1]
        pred_traj_rel = []
        #lstm model has ht and ct output that is again input in the next lstm neuron
        #ct is the current state with remembered information
        #and ht is the output
        traj_lstm_h_t, traj_lstm_c_t = self.init_hidden_traj_lstm(batch)
        #print("ht size: ", traj_lstm_h_t.size())
        #print("ct size: ", traj_lstm_c_t.size())
        #relative trajectory entries from 0 to 7 (observations)
        #split the  into chunks of size 8
        #results in a kind of dictionary [1:tensor, 2:tensor]?
        #but does this not return a single tensor of size 8?
        chunksize = traj_rel[: self.obs_len].size(0)
        iteration_num = traj_rel[: self.obs_len].chunk(chunksize, dim=0)

        for i, input_t in enumerate(iteration_num):
            #print("input dim: ", input_t.size())
            #print("input: ", input_t)
            #relu activation function, f(x)=0 for x<0, f(x)=x for x > 0
            #squeeze removes all dimensions of size 1
            #input from 2 dim to 16dim -> embedding
            input_embedded = F.relu(self.inputLayer_encoder(input_t.squeeze(0)))
            #print("input embedded dim: ", input_embedded.size())
            #compute output and current state from lstm
            lstm_state = self.traj_lstm_model(input_embedded, (traj_lstm_h_t, traj_lstm_c_t))
            traj_lstm_h_t, traj_lstm_c_t = lstm_state

        #all knowledge is reset here?
        #output/prediction of lstm is h_t or should be ht
        pred_lstm_hidden = traj_lstm_h_t
        #remove system knowledge/state?
        pred_lstm_c_t = torch.zeros_like(pred_lstm_hidden).cuda()
        #init output as zeros scalar tensor?
        output = torch.zeros_like(traj_rel[self.obs_len - 1]).cuda()
        #init lstm state context
        lstm_state_context = torch.zeros_like(pred_lstm_hidden).cuda()
        #current pos is last observed pos
        curr_pos_abs = obs_traj_pos[-1]

        ##PREDICTIONS
        #for 12 predictions
        for i in range(self.pred_len):
            # detach from history as input
            #why do we use detach? it is important in autoregression for good prediction
            #needed for backpropagation
            input_cat = torch.cat([lstm_state_context.detach(),output.detach()], dim=-1)
            inputtest = self.inputLayer_decoder(input_cat)
            input_embedded = F.relu(inputtest)
            lstm_state = self.pred_lstm_model(
                input_embedded, (pred_lstm_hidden, pred_lstm_c_t)
            )
            pred_lstm_hidden = lstm_state[0]
            pred_lstm_c_t = lstm_state[1]
            #repeat last observed position (of every person?) batch times - so for every person
            corr = curr_pos_abs.repeat(batch, 1, 1)

            #print("currpos:", curr_pos_abs)
            #print("corr ", corr[0])
            #swap dimensions 0 and 1 and substract?
            corr_index = (corr.transpose(0,1)-corr)
            #print("corr index: ", corr_index[0])
            #print("neighbor index:", nei_index[i])
            #print("neighbor num index:", nei_num_index)
            lstm_state_hidden = lstm_state[0]
            lstm_state_context, _, _ = self.pl_net(corr_index, nei_index[i], nei_num_index, lstm_state_hidden, curr_pos_abs)
            concat_output = lstm_state_context + lstm_state_hidden
            output = torch.tanh(self.pred_hidden2pos(concat_output))* 4.4 # why 4.4 ?
            #print("output:", output)
            curr_pos_abs = (curr_pos_abs + output).detach() # detach from history as input
            #print("output:", output)
            pred_traj_rel += [output]

        return torch.stack(pred_traj_rel)
