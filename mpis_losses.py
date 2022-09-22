import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence

def smooth_l1_loss(input, target, loss_mask, beta=1. / 9, size_average=True):
    # very similar to the smooth_l1_loss from pytorch, but with the extra beta parameter
    n = torch.abs(input - target)*loss_mask
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss #.sum()


def l2_loss(pred_traj, pred_traj_gt, loss_mask, mode='average'):

    loss = smooth_l1_loss(pred_traj_gt.permute(1, 0, 2), pred_traj.permute(1, 0, 2),loss_mask.permute(1, 0, 2), beta = 0.05, size_average=False)

    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(loss_mask.data) # numel Calculate The Number Of Elements In A PyTorch Tensor
    elif mode == 'raw':  # FOR TRAINING!
        return loss.sum(dim=2).sum(dim=1)


def L2(pred_traj_rel_recon, pred_traj_gt_rel, loss_mask, pred_traj_gt, seq_start_end, mode="raw"):

    l2_loss_rel = []
    l2_loss_rel.append(l2_loss(pred_traj_rel_recon, pred_traj_gt_rel, loss_mask, mode=mode))
    l2_loss_rel = torch.stack(l2_loss_rel, dim=1)

    l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
    for start, end in seq_start_end.data:
        _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end - start)
        _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)  # [20]
        _l2_loss_rel = torch.min(_l2_loss_rel) / ((pred_traj_rel_recon.shape[0]) * (end - start))
        l2_loss_sum_rel += _l2_loss_rel
    return l2_loss_sum_rel


def loss_function(mu, logvar, pred_traj_rel_recon, pred_traj_gt_rel, loss_mask, pred_traj_gt, seq_start_end):
    # Loss 1
    KL_Divergence = -0.5 * torch.sum(1 + logvar - torch.exp(logvar) - mu ** 2)

    # Loss 2
    L2_loss = L2(pred_traj_rel_recon, pred_traj_gt_rel, loss_mask, pred_traj_gt, seq_start_end)

    return 2*KL_Divergence + L2_loss, KL_Divergence, L2_loss


def loss_function_new(gnll_loss, recog_mu, recog_logvar, prior_mu, prior_logvar, pred_traj_rel_recon, pred_traj_gt_rel, loss_mask, pred_traj_gt, seq_start_end):
    # Loss 1
    KL_Divergence = 0.5 * torch.sum(prior_logvar - recog_logvar - 1 + recog_logvar.exp() / prior_logvar.exp() + (prior_mu - recog_mu).pow(2) / prior_logvar.exp())
    # Loss 2
    L2_loss = L2(pred_traj_rel_recon, pred_traj_gt_rel, loss_mask, pred_traj_gt, seq_start_end)
    #Loss 3
    #GNLL_Loss = GNLL(position_mean, pred_traj_gt_rel, position_variance)**2
    return KL_Divergence + L2_loss + 0.1*gnll_loss, KL_Divergence, L2_loss, gnll_loss

#kldiv from https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
def kldiv_loss(z, prior_mu, prior_var, recog_mu, recog_var):
    # --------------------------
    # Monte carlo KL divergence
    # --------------------------

    #define the two probabilities, they are both normal
    #q is prior p(z|c) only given condition

    #print("prior_mu: " + str(prior_mu) + " prior_var var: " + str(prior_var))

    if(prior_mu == None or prior_var == None):
        q = torch.distributions.Normal(torch.zeros_like(recog_mu), torch.ones_like(recog_var))
    else:
        q = torch.distributions.Normal(prior_mu, torch.exp(prior_var))
    #p is posterior q(z|x,c) given also the pred input
    #print("recog_mu: " + str(recog_mu) + " recog var: " + str(recog_var))
    p = torch.distributions.Normal(recog_mu, torch.exp(recog_var))


    #log probabilities of distributions, given z
    log_qz = q.log_prob(z)
    #print("log_qz: ", torch.exp(log_qz))
    log_pzx = p.log_prob(z)
    #print("log_pzx: ", torch.exp(log_pzx))

    #print("log_qz: " + str(log_qz.shape))
    #print("log_pzx: " + str(log_pzx.shape))
    # kl
    kl = log_pzx - log_qz #(torch.mean(log_pzx) - torch.mean(log_qz))

    # sum over last dim to go from single dim distribution to multi-dim
    kl = kl.sum(-1)
    return kl

def get_distributions(prior_mu, prior_logvar, recog_mu, recog_logvar):
    x = torch.arange(0, 1, 0.05).cuda()
    p_normal = torch.exp(-torch.square(x-recog_mu) / 2 * torch.square(recog_logvar))
    p = p_normal / torch.sum(p_normal)
    q_normal = torch.exp(-torch.square(x - prior_mu) / 2 * torch.square(prior_logvar))
    q = q_normal / torch.sum(q_normal)

    return p, q

def reconstruction_loss(mu, sigma, pos):
    log_scale = nn.Parameter(torch.tensor([0.0]).cuda())
    scale = torch.exp(log_scale)
    dist = torch.distributions.Normal(mu, sigma)
    log_pxz = dist.log_prob(pos)
    log_pxz = log_pxz.sum(dim=(0,-1))
    #print("reconst loss shape: ", log_pxz.shape)
    recon_loss = log_pxz #torch.mean(log_pxz)

    return recon_loss

def compute_kernel(x, y):
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]

    tiled_x = x.view(x_size,1,dim).repeat(1, y_size,1)
    tiled_y = y.view(1,y_size,dim).repeat(x_size, 1,1)

    return torch.exp(-torch.mean((tiled_x - tiled_y)**2,dim=2)/dim*1.0)

def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2*torch.mean(xy_kernel)

def loss_function_gnll(z, z_prior, pred_mean, pred_var, recog_mu, recog_logvar, prior_mu, prior_logvar, pred_traj_rel_recon, pred_traj_gt_rel, loss_mask, pred_traj_gt, seq_start_end):
    #ELBO Loss (source: https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed)
    elbo = 0.5 * torch.sum(prior_logvar - recog_logvar - 1 + recog_logvar.exp() / prior_logvar.exp() + (prior_mu - recog_mu).pow(2) / prior_logvar.exp())

    #Elbo for N(0,1)
    #elbo = -0.5 * torch.sum(1 + recog_logvar - torch.exp(recog_logvar) - recog_mu ** 2)

    #kl_loss = kldiv_loss(z, prior_mu, prior_logvar, recog_mu, recog_logvar)
    #recon_loss = reconstruction_loss(pred_mean, pred_var, pred_traj_gt_rel)
    #elbo_loss = kl_loss - recon_loss
    #elbo_loss = torch.mean(elbo_loss)

    #computing distributions from mu and sigma and then kldiv
    #p, q = get_distributions(prior_mu, prior_logvar, recog_mu, recog_logvar)

    #kl_loss = torch.sum(torch.where(q == 0, torch.zeros_like(p), p * torch.log(p / q)))

    #torch KLDiv function
    #torchKlDiv = torch.nn.KLDivLoss(reduction = 'batchmean', log_target=True)
    #kl_loss = torchKlDiv(p,q)

    #L2 loss (MSE)
    L2_loss = L2(pred_traj_rel_recon, pred_traj_gt_rel, loss_mask, pred_traj_gt, seq_start_end)

    #Gaussian Negative Log Likelihood (better suited for our measure - includes variance)
    GNLL = nn.GaussianNLLLoss()
    GNLL_Loss = GNLL(pred_traj_rel_recon, pred_traj_gt_rel, pred_var)
    batchsize = pred_traj_rel_recon.shape[1]
    GNLL_mean = GNLL_Loss / batchsize

    #mmd = compute_mmd(z, z_prior)
    batch_size = z.shape[1]
    kernels = gaussian_kernel(z, z_prior)
    XX = gaussian_kernel(z, z)
    YY = gaussian_kernel(z_prior, z_prior)
    XY = gaussian_kernel(z, z_prior)
    YX = gaussian_kernel(z_prior, z)
    mmd = torch.mean(XX + YY - XY - YX)
    #print("mmd: ", mmd)

    return mmd, elbo, L2_loss, GNLL_Loss
