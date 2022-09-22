import os
import time
import torch
import numpy as np
import inspect
from contextlib import contextmanager
import subprocess
from config import *
config = Config()
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
def int_tuple(s):
    return tuple(int(i) for i in s.split(','))


def find_nan(variable, var_name):
    variable_n = variable.data.cpu().numpy()
    if np.isnan(variable_n).any():
        exit('%s has nan' % var_name)


def bool_flag(s):
    if s == '1':
        return True
    elif s == '0':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)


def lineno():
    return str(inspect.currentframe().f_back.f_lineno)


def get_total_norm(parameters, norm_type=2):
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            try:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm**norm_type
                total_norm = total_norm**(1. / norm_type)
            except:
                continue
    return total_norm


@contextmanager
def timeit(msg, should_time=True):
    if should_time:
        torch.cuda.synchronize()
        t0 = time.time()
    yield
    if should_time:
        torch.cuda.synchronize()
        t1 = time.time()
        duration = (t1 - t0) * 1000.0
        print('%s: %.2f ms' % (msg, duration))


def get_gpu_memory():
    torch.cuda.synchronize()
    opts = [
        'nvidia-smi', '-q', '--gpu=' + str(1), '|', 'grep', '"Used GPU Memory"'
    ]
    cmd = str.join(' ', opts)
    ps = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ps.communicate()[0].decode('utf-8')
    output = output.split("\n")[0].split(":")
    consumed_mem = int(output[1].strip().split(" ")[0])
    return consumed_mem


def get_dset_path(dset_name, dset_type):
    _dir = os.path.dirname(__file__)
    _dir = _dir.split("/")[:-1]
    _dir = "/".join(_dir) + "/CoLoss1"
    return os.path.join(_dir, 'datasets', dset_name, dset_type)


def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    if config.nabs:
        # rel_traj = rel_traj.permute(1, 0, 2)
        # start_pos = torch.unsqueeze(start_pos, dim=1)
        # return (rel_traj + start_pos).permute(1, 0, 2)
        return rel_traj * 28.
    else:
        rel_traj = rel_traj.permute(1, 0, 2)
        displacement = torch.cumsum(rel_traj, dim=1)
        start_pos = torch.unsqueeze(start_pos, dim=1)
        abs_traj = displacement + start_pos # traj are rel, so add them up and add to the last known position! so they are rel to last position!
        return abs_traj.permute(1, 0, 2)

DIR = os.path.join(config.home, config.model_path, "plot")
cmap = plt.cm.get_cmap('hsv')

human_colors = cmap(np.linspace(0.5, 1.0, 5))
human_colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']

def plot_multimodal(obs_traj, pred_traj_gt, batch_pred_traj_fake_rel, att_score_list_batch, ids, stack_of_coll_indeces,
                   seq_start_end, ids_of_col_szenes_fake, loss_mask, config, szene_id):
    if config.num_samples > 1:
        all_pred = torch.stack(batch_pred_traj_fake_rel)
    szene_id_batch_offset = szene_id*len(seq_start_end)
    search_ids = [18, 38, 25]
    for k, (start, end) in enumerate(seq_start_end):
        nmp_humans = end-start

        if ids_of_col_szenes_fake[k]:  # and nmp_humans <= 5:               # plot scenes only with collisions
        # if nmp_humans <= 10:                                              # plot scenes with only less than x humans
        # if szene_id_batch_offset + k in search_ids:                       # look for scenes with a specific id
            if config.num_samples > 1:
                all_pred_szene = all_pred[:, :, start:end, :].cpu().numpy()
            obs_traj_szene = obs_traj[:, start:end, :].cpu().numpy()
            loss_mask_szene = loss_mask[:, start:end, :].cpu().numpy()
            ground_truth_input_x_piccoor = (obs_traj_szene[:, :, 0].T)
            ground_truth_input_y_piccoor = (obs_traj_szene[:, :, 1].T)
            fig, ax = plt.subplots()
            ax.axis('off')
            for i in range(end-start):
                obs_traj_i = obs_traj_szene[:,i]
                loss_mask_szene_i = loss_mask_szene[:,i]
                for t, timestep in enumerate(obs_traj_i):
                    if loss_mask_szene_i[t,0]:
                        obs_pos = plt.Circle(timestep, config.collision_distance/2., fill=False, color='black',
                                             label='Traj. History', alpha=.5)
                        ax.add_artist(obs_pos)

                obs_traj_i_filtered = obs_traj_i[np.nonzero(loss_mask_szene_i[0:config.obs_len,0])[0]]
                observed_line = plt.plot(obs_traj_i_filtered[:,0],obs_traj_i_filtered[:,1],"black",linewidth=2,
                                         label="History",alpha=.5)[0]

                observed_line.axes.annotate(
                    "",
                    xytext=(
                        ground_truth_input_x_piccoor[i, -2],
                        ground_truth_input_y_piccoor[i, -2],
                    ),
                    xy=(
                        ground_truth_input_x_piccoor[i, -1],
                        ground_truth_input_y_piccoor[i, -1],
                    ),
                    arrowprops=dict(
                        arrowstyle="-|>", color=observed_line.get_color(), lw=1
                    ),
                    size=15, alpha=1
                )

                if config.num_samples > 1:
                    for s in range(5):
                        pred_traj = all_pred_szene[s]
                        pred_traj = pred_traj[:,i]
                        plt.plot(
                            np.append(
                                obs_traj_i_filtered[-1,0],
                                pred_traj[:, 0],
                            ),
                            np.append(
                                obs_traj_i_filtered[-1,1],
                                pred_traj[:,1],
                            ),
                            color=human_colors[s],
                            linewidth=1,
                            alpha=.5
                        )
            fig.suptitle(szene_id_batch_offset + k, fontsize=20)
            ax.set_aspect("equal")
            # plt.savefig(DIR + 'SGAN2'+ str(szene_id_batch_offset + k) +'.png', dpi=800,bbox_inches='tight')
            plt.show()

        plt.close()

def plot_best(obs_traj, pred_traj_gt, batch_pred_traj_fake_rel, att_score_list_batch, ids, stack_of_coll_indeces,
                   seq_start_end, ids_of_col_szenes_fake, loss_mask, config, szene_id):
    #cmap = plt.cm.get_cmap('hsv', 5)
    if config.num_samples > 1:
        all_pred = torch.stack(batch_pred_traj_fake_rel)
    szene_id_batch_offset = szene_id*len(seq_start_end)
    # search_ids = [18, 38, 25]
    for k, (start, end) in enumerate(seq_start_end):
        nmp_humans = end-start
        # if ids_of_col_szenes_fake[k]:  # and nmp_humans <= 5:               # plot scenes only with collisions
        if nmp_humans <= 10:                                              # plot scenes with only less than x humans
        # if szene_id_batch_offset + k in search_ids:                       # look for scenes with a specific id
            tmp_stack_of_coll_indeces = stack_of_coll_indeces[k].cpu().numpy()
            model_output_traj_best = batch_pred_traj_fake_rel[ids[k]]
            obs_traj_szene = obs_traj[:, start:end, :].cpu().numpy()
            pred_traj_gt_szene = pred_traj_gt[:, start:end, :].cpu().numpy()
            model_output_traj_best_szene = model_output_traj_best[:, start:end, :].cpu().numpy()
            loss_mask_szene = loss_mask[:, start:end, :].cpu().numpy()
            ground_truth_input_x_piccoor = (obs_traj_szene[:, :, 0].T)
            ground_truth_input_y_piccoor = (obs_traj_szene[:, :, 1].T)
            fig, ax = plt.subplots()
            ax.axis('off')
            timesteps = np.linspace(1, 0.1, config.pred_len)

            for i in range(end-start):
                pred_traj_gt_i = pred_traj_gt_szene[:,i]
                obs_traj_i = obs_traj_szene[:,i]
                model_output_traj_best_i = model_output_traj_best_szene[:,i]
                loss_mask_szene_i = loss_mask_szene[:,i]

                for t, timestep in enumerate(model_output_traj_best_i):
                    if loss_mask_szene_i[config.obs_len + t,0]:
                        gt_pos = plt.Circle(timestep, config.collision_distance/2., fill=False, color='b',
                                            label='Prediction', alpha=timesteps[t])
                        ax.add_artist(gt_pos)
                        pred_pos = plt.Circle(pred_traj_gt_i[t], config.collision_distance/2., fill=False, color='r',
                                              label='Ground Truth', alpha=timesteps[t])
                        ax.add_artist(pred_pos)
                        if True in tmp_stack_of_coll_indeces[t,i]:
                            coll_pers_pos = plt.Circle(timestep, config.collision_distance/2., fill=False, color='green',
                                                       label='Collisions')
                            ax.add_artist(coll_pers_pos)
                for t, timestep in enumerate(obs_traj_i):
                    if loss_mask_szene_i[t,0]:
                        obs_pos = plt.Circle(timestep, config.collision_distance/2., fill=False, color='black',
                                             label='Traj. History', alpha=.5)
                        ax.add_artist(obs_pos)

                obs_traj_i_filtered = obs_traj_i[np.nonzero(loss_mask_szene_i[0:config.obs_len,0])[0]]
                mask_tmp = np.nonzero(loss_mask_szene_i[config.obs_len:(config.pred_len+ config.obs_len),0])[0]
                pred_traj_gt_i_filtered = pred_traj_gt_i[mask_tmp]
                model_output_traj_best_i_filtered = model_output_traj_best_i[mask_tmp]
                observed_line = plt.plot(obs_traj_i_filtered[:,0],obs_traj_i_filtered[:,1],"black",linewidth=2,
                                         label="History",alpha=.5)[0]

                observed_line.axes.annotate(
                    "",
                    xytext=(
                        ground_truth_input_x_piccoor[i, -2],
                        ground_truth_input_y_piccoor[i, -2],
                    ),
                    xy=(
                        ground_truth_input_x_piccoor[i, -1],
                        ground_truth_input_y_piccoor[i, -1],
                    ),
                    arrowprops=dict(
                        arrowstyle="-|>", color=observed_line.get_color(), lw=1
                    ),
                    size=15, alpha=1
                )

                plt.plot(
                    np.append(
                        obs_traj_i_filtered[-1,0],
                        pred_traj_gt_i_filtered[:, 0],
                    ),
                    np.append(
                        obs_traj_i_filtered[-1,1],
                        pred_traj_gt_i_filtered[:,1],
                    ),
                    "r",
                    linewidth=2,
                    alpha=.1
                )


                plt.plot(
                    np.append(
                        obs_traj_i_filtered[-1,0],
                        model_output_traj_best_i_filtered[:, 0],
                    ),
                    np.append(
                        obs_traj_i_filtered[-1,1],
                        model_output_traj_best_i_filtered[:,1],
                    ),
                    "b",
                    # ls="--",
                    linewidth=2,
                    alpha=.1
                )
            fig.suptitle(szene_id_batch_offset + k, fontsize=20)
            ax.set_aspect("equal")
            if(not os.path.exists(DIR)):
                os.makedirs(DIR)
            filepath = os.path.join(DIR, str(int(szene_id_batch_offset)) + "_" + str(k) + '.png')
            print("k: " + str(k) + " id: " + str(szene_id_batch_offset) + " path: " + filepath)
            plt.savefig(filepath, dpi=800,bbox_inches='tight')
            #plt.show()

        plt.close()





