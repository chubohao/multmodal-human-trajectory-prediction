import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from config import *
import numpy as np
config = Config()
from data.loader import data_loader
from models.STGAT import TrajectoryGenerator
from utils import (
    relative_to_abs,
    get_dset_path,
    plot_trajecotry,
    plot_attention
)
from losses import displacement_error, final_displacement_error
from attrdict import AttrDict
DIR = home +'/Documents/col_aware/191-STGAT_BaseLine-zara2/checkpoint_with_model.pt'
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default=DIR, type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)

seed = config.seed
torch.manual_seed(seed)
np.random.seed(seed)
# import locale
# locale.setlocale(locale.LC_ALL, 'de_DE.utf8')

def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)
    ids = []
    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0).unsqueeze(dim=1)
        _error, id = torch.min(_error, 0)
        ids.append(id.squeeze().item() )
        sum_ += _error.squeeze()
    return sum_, ids


def get_generator(checkpoint):
    model = TrajectoryGenerator(config)
    model.load_state_dict(checkpoint["g_best_state"])
    model.cuda()
    model.eval()
    return model


def cal_ade_fde(pred_traj_gt, pred_traj_fake, val_mask):
    ade = displacement_error(pred_traj_fake, pred_traj_gt, val_mask, mode="raw")
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], val_mask[-1], mode="raw")
    return ade, fde

def fast_coll_counter(pred_batch,seq_start_end, ids, mask):
    ids_of_col_szenes = np.zeros([seq_start_end.shape[0]])
    coll_pro_szene = 0
    # pred_batch = pred_batch * mask[-config.pred_len :]
    stack_of_coll_indeces = []
    for i, (start, end) in enumerate(seq_start_end):
        start = start.item()
        end = end.item()
        if ids:
            pred = pred_batch[ids[i]]
        else:
            pred = pred_batch
        pred = pred #*mask # ToDo Change this if mask is needed!
        currSzene = pred[:, start:end]
        dist_mat = torch.cdist(currSzene, currSzene, p=2.0, compute_mode='donot_use_mm_for_euclid_dist')
        dist_mat_one_triu = torch.triu(dist_mat)
        dist_mat_one_triu = dist_mat_one_triu * mask[:,start:end, start:end]
        filter_zeros = torch.logical_and(0. != dist_mat_one_triu,  dist_mat_one_triu <= config.collision_distance)
        filter_col_pos = torch.logical_and(0. != dist_mat,  dist_mat <= config.collision_distance)
        filter_zeros_sum=filter_zeros.sum().unsqueeze(dim=0).item()
        coll_pro_szene += filter_zeros_sum
        if filter_zeros_sum > 0.:
            ids_of_col_szenes[i] = 1
        stack_of_coll_indeces.append(filter_col_pos)
    # print(filter_zeros)
    count = len(seq_start_end) #* count_empty
    # dist_list.append(filter_zeros[filter_zeros <= config.collision_distance])
    return coll_pro_szene, count, ids_of_col_szenes, stack_of_coll_indeces

def evaluate(args, loader, generator, gt_coll=False, plot_traj=False):
    ade_outer, fde_outer = [], []
    total_traj = 0
    coll_pro_szenes_fake = 0.
    count_szenes_fake = 0.
    coll_pro_szenes = 0.
    count_szenes = 0.
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, val_mask, \
             loss_mask, seq_start_end, nei_num_index, nei_num = batch
            ade, fde = [], []
            # total_traj += pred_traj_gt.size(1)
            total_traj += nei_num.sum()
            batch_pred_traj_fake = []
            att_score_list_batch = []
            model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)
            val_mask = val_mask[-config.pred_len :]
            for _ in range(args.num_samples):
                if plot_traj:
                    pred_traj_fake_rel = generator(model_input, obs_traj, pred_traj_gt,
                                                   seq_start_end, nei_num_index, nei_num, 0, plot_att = False)
                    # att_score_list_batch.append(att_score_list)
                else:
                    pred_traj_fake_rel = generator(model_input, obs_traj, pred_traj_gt,
                                                    seq_start_end, nei_num_index, nei_num, 0, plot_att = False)
                pred_traj_fake_rel = pred_traj_fake_rel[-args.pred_len :]
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                batch_pred_traj_fake.append(pred_traj_fake)
                ade_, fde_ = cal_ade_fde(pred_traj_gt, pred_traj_fake, val_mask)
                ade.append(ade_)
                fde.append(fde_)
            ade_sum, ids = evaluate_helper(ade, seq_start_end)
            fde_sum, _ = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)

            coll_pro_szene_fake, count_fake, ids_of_col_szenes_fake, stack_of_coll_indeces = fast_coll_counter(batch_pred_traj_fake,
                                                                                        seq_start_end, ids, nei_num_index)
            coll_pro_szenes_fake += coll_pro_szene_fake
            count_szenes_fake += count_fake
            if gt_coll:
                coll_pro_szene, count, ids_of_col_szenes_gt,_ = fast_coll_counter(pred_traj_gt, seq_start_end, None, nei_num_index)
                coll_pro_szenes += coll_pro_szene
                count_szenes += count
            if(plot_traj):
           #     plot_trajecotry(obs_traj, pred_traj_gt, batch_pred_traj_fake_rel, ids,  seq_start_end,
           #                     ids_of_col_szenes_fake, config)
                plot_attention(obs_traj, pred_traj_gt, batch_pred_traj_fake, att_score_list_batch, ids,
                               stack_of_coll_indeces, seq_start_end,
                               ids_of_col_szenes_fake, loss_mask, config)

        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        act = coll_pro_szenes_fake / count_szenes_fake
        if count_szenes!=0:
            act_gt = coll_pro_szenes / count_szenes
        else:
            act_gt = 0
        return ade, fde, act, act_gt


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]
    for path in paths:
        checkpoint = torch.load(path)
        coll_distance = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70]
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        path = get_dset_path(_args.dataset_name, args.dset_type)

        _, loader = data_loader(_args, path)
        # ade, fde, act, act_gt = evaluate(_args, loader, generator, gt_coll=True, plot_traj=False)
        for disc in coll_distance:
            config.collision_distance=disc
            ade, fde, act, act_gt = evaluate(_args, loader, generator, gt_coll=False, plot_traj=False)
            # print(
            #     "{:.4f}, {:.2f}".format(
            #          act_gt, disc,
            #     )
            print(
                "{:.4f}".format(
                    act
                )
            )


if __name__ == "__main__":
    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main(args)
