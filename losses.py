import torch
import random
from config import *
config = Config()
Tensor = torch.cuda.FloatTensor
import numpy as np
import torch.autograd as autograd



def coll_smoothed_loss(pred_batch ,seq_start_end,  mask):

    coll_pro_szene = torch.zeros([seq_start_end.size()[0]]).to(pred_batch)
    z = torch.zeros([1]).to(pred_batch)
    y = torch.ones([1]).to(pred_batch)
    for i, (start, end) in enumerate(seq_start_end):
        start = start.item()
        end = end.item()
        #print("pred batch size", pred_batch.shape)
        #print("mask size", mask.shape)
        pred = pred_batch #* mask[:,start:end, start:end] # ToDo Change this if mask is needed!
        currSzene = pred[:, start:end].contiguous()
        dist_mat_pred = torch.cdist(currSzene, currSzene, p=2.0, compute_mode='donot_use_mm_for_euclid_dist')
        dist_mat_pred = dist_mat_pred * mask[:,start:end, start:end].detach() # detach mask from computational graph!
        dist_mat_pred = dist_mat_pred[dist_mat_pred !=0.]
        dist_mat_pred = torch.sigmoid((dist_mat_pred - config.collision_distance)*35.) * (z - y) + y   # cause binary tensor is not differentiable
        dist_mat_pred = dist_mat_pred.sum(dim=-1) # get number of coll for every pedestrian traj
        coll_pro_szene[i] = dist_mat_pred.sum().unsqueeze(dim=0)/(end - start)

    #print("coll pro szene: ", coll_pro_szene)
    #print("in szenes ", i)
    return coll_pro_szene.mean()


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.
    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    Input:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Output:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of
      input data.
    """
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def gan_g_loss(scores_fake):
    """
    Input:
    - scores_fake: Tensor of shape (N,) containing scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN generator loss
    """
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.2)
    return bce_loss(scores_fake, y_fake)


def gan_d_loss(scores_real, scores_fake):
    """
    Input:
    - scores_real: Tensor of shape (N,) giving scores for real samples
    - scores_fake: Tensor of shape (N,) giving scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN discriminator loss
    """
    y_real = torch.ones_like(scores_real) #* random.uniform(0.7, 1.2) #ToDO soft target! for hole tensor!!!!!
    y_fake = torch.zeros_like(scores_fake) #* random.uniform(0, 0.3)
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real + loss_fake

def smooth_l1_loss(input, target,loss_mask, beta=1. / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    #distance of input - target
    n = torch.abs(input - target)
    #print("distance n:", n.shape)
    n = n *loss_mask
    #print("distance n with mask:", n.shape)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss #.sum()


def l2_loss(pred_traj, pred_traj_gt, loss_mask, mode='average'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - loss_mask: Tensor of shape (batch, seq_len)
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """
    seq_len, batch, _ = pred_traj.size()
    #permute changes order of tensor. befpre (seq_len, batch, x/y) now (batch, seq_len, x/y)
    #loss mask: 1,1
    loss = smooth_l1_loss(pred_traj_gt.permute(1, 0, 2),
                          pred_traj.permute(1, 0, 2),loss_mask.permute(1, 0, 2),
                          beta = 0.05, size_average=False)

    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(loss_mask.data) # numel Calculate The Number Of Elements In A PyTorch Tensor
    elif mode == 'raw':  # FOR TRAINING!
        return loss.sum(dim=2).sum(dim=1)


def displacement_error(pred_traj, pred_traj_gt, val_mask, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    seq_len, _, _ = pred_traj.size()
    loss = (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2))*val_mask.permute(1, 0, 2)
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(
    pred_pos, pred_pos_gt, consider_ped=None, mode='sum'
):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    last pos
    - consider_ped: Tensor of shape (batch)
    Output:
    - loss: gives the eculidian displacement error
    """
    loss = pred_pos_gt - pred_pos
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt((loss*consider_ped).sum(dim=1))
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)


def compute_gradient_penalty1D(D,obs_traj_rel, pred_traj_gt_rel, pred_traj_fake_rel,
                               obs_traj, nei_num_index, nei_num, loss_mask):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    shape = pred_traj_gt_rel.size()
    batch_size = shape[1]
    real_samples = pred_traj_gt_rel.detach()
    fake_samples = pred_traj_fake_rel.detach()
    alpha = Tensor(np.random.random((batch_size, 1)))

    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))

    interpolates = autograd.Variable(interpolates, requires_grad=True)
    d_interpolates = D(obs_traj_rel, interpolates,
                       obs_traj,nei_num_index, nei_num, loss_mask)

    fake = torch.ones(d_interpolates.size()).cuda()
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradient_penalty = (gradients.norm(2, dim=1) - 1).pow(2).mean()
    return gradient_penalty
