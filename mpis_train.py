import os
import sys
import time
import copy
from collections import defaultdict
import torch
import wandb
import config
import logging
from utils import relative_to_abs
import numpy as np
import torch.utils.data
from torch import nn, optim
from data.loader import data_loader
#from models.mpis_cvae_lstm_ar import CVAE
from models.mpis_cvae_lstm_ar_sigma import CVAE
from mpis_evaluate import evaluate
from torch.nn import functional as F
from losses import coll_smoothed_loss
from mpis_losses import loss_function, loss_function_new, L2, loss_function_timo
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

class MPIS:
    def __init__(self):
        self.config = config.Config()
        self.checkpoint = {
            'args': self.config.__dict__,
            'losses': defaultdict(list),
            'losses_ts': [],
            'metrics_val': defaultdict(list),
            'metrics_train': defaultdict(list),
            'sample_ts': [],
            'restore_ts': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'state': None,
            'optim_state': None,
            'best_state': None,
        }

        self.project = "mpis"
        self.name = self.config.experiment_name
        wandb.init(project=self.project, name=self.name, reinit=True, entity="mpis3")
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"


        trainpath = os.path.join("datasets", self.config.dataset_name, "train")
        testpath = os.path.join("datasets", self.config.dataset_name, "testpath")
        valpath = os.path.join("datasets", self.config.dataset_name, "valpath")
        _, self.train_loader = data_loader(self.config, "datasets/zara2/train/", augment=self.config.augment)
        _, self.test_loader = data_loader(self.config, "datasets/zara2/test/", augment=self.config.augment)
        _, self.val_loader = data_loader(self.config, "datasets/zara2/val/")
        self.model = CVAE()
        self.model.type(torch.cuda.FloatTensor)
        print(self.model)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, betas=(0.5, 0.9999))

    def check_accuracy(self, loader, model):  # TODO Change this!
        metrics = {}
        model.eval()

        ade, fde, act, _, prior_mu, prior_logvar = evaluate(self.config, loader, model, plot_traj=False, predict=True)
        metrics['act'] = act
        metrics['ade'] = ade
        metrics['fde'] = fde
        mean = (act + ade) / 2
        metrics['mean'] = mean
        model.train()
        wandb.log({"ade[val]": metrics['ade'], "fde[val]": metrics['fde'], "act[val]": metrics['act'], "Mean[val]": metrics['mean']})
        return metrics

    def test_old(self):
        logger.info('Checking stats on test ...')
        metrics = {}
        self.model = CVAE()
        #self.model.load_state_dict(torch.load("cvae_model.pt"))
        model = torch.load("cvae_model.pkl")
        self.model.cuda()
        self.model.eval()
        loader = self.test_loader
        ade, fde, act, _, prior_mu, prior_logvar = evaluate(self.config, loader, self.model, plot_traj=False, predict=True)
        metrics['act'] = act
        metrics['ade'] = ade
        metrics['fde'] = fde
        mean = (act + ade) / 2
        metrics['mean'] = mean
        wandb.log({"ade[test]": metrics['ade'], "fde[test]": metrics['fde'], "act[test]": metrics['act'], "Mean[test]": metrics['mean']})
        for k, v in sorted(metrics.items()):
            logger.info('  [test] {}: {:.3f}'.format(k, v))

        logger.info('Done.')


    def test(self):

        logger.info('Checking stats on test ...')
        metrics = {}
        modelpath = os.path.join(self.config.home, self.config.checkpoint_start_from)
        checkpoint = torch.load(modelpath)
        logger.info('model loaded from {}'.format(modelpath))
        model = CVAE()
        model.load_state_dict(checkpoint["best_state"])
        model.cuda()
        model.eval()
        self.model.eval()
        loader = self.test_loader
        ade, fde, act, _, prior_mu, prior_logvar = evaluate(self.config, loader, self.model, plot_traj=False, predict=True)
        metrics['act'] = act
        metrics['ade'] = ade
        metrics['fde'] = fde
        mean = (act + ade) / 2
        metrics['mean'] = mean
        wandb.log({"ade[test]": metrics['ade'], "fde[test]": metrics['fde'], "act[test]": metrics['act'], "Mean[test]": metrics['mean']})
        for k, v in sorted(metrics.items()):
            logger.info('  [test] {}: {:.3f}'.format(k, v))

        logger.info('Done.')

    def save_model(self):
        self.checkpoint['counters']['t'] = 0
        self.checkpoint['counters']['epoch'] = self.epoch
        self.checkpoint['sample_ts'].append(0)

        # Check stats on the validation set
        logger.info('Checking stats on val ...')
        metrics_val = self.check_accuracy(self.val_loader, self.model)
        #    self.scheduler.step(metrics_val['ade'])
        for k, v in sorted(metrics_val.items()):
            logger.info('  [val] {}: {:.3f}'.format(k, v))
            self.checkpoint['metrics_val'][k].append(v)

        min_mean = min(self.checkpoint['metrics_val']['mean'])
        if metrics_val['mean'] == min_mean:
            logger.info('New low for mean error')
            self.checkpoint['best_t'] = 0
            self.checkpoint['best_state'] = copy.deepcopy(self.model.state_dict())

        # Save another checkpoint with model weights and
        # optimizer state
        self.checkpoint['state'] = self.model.state_dict()
        self.checkpoint['optim_state'] = self.optimizer.state_dict()
        checkpoint_path = os.path.join(
            self.config.output_dir, '%s_with_model.pt' % self.config.checkpoint_name
        )
        logger.info('Saving checkpoint to {}'.format(checkpoint_path))
        torch.save(self.checkpoint, checkpoint_path)
        torch.save(self.checkpoint, os.path.join(wandb.run.dir, 'model.pt'))
        logger.info('Done.')

        # Save a checkpoint with no model weights by making a shallow
        # copy of the checkpoint excluding some items
        checkpoint_path = os.path.join(
            self.config.output_dir, '%s_no_model.pt' % self.config.checkpoint_name)
        logger.info('Saving checkpoint to {}'.format(checkpoint_path))
        key_blacklist = [
            'state', 'best_state', 'optim_state'
        ]
        small_checkpoint = {}
        for k, v in self.checkpoint.items():
            if k not in key_blacklist:
                small_checkpoint[k] = v
        torch.save(small_checkpoint, checkpoint_path)
        logger.info('Done.')

    def save_model_old(self):
        logger.info('Checking stats on val ...')
        metrics_val = self.check_accuracy(self.val_loader, self.model)
        for k, v in sorted(metrics_val.items()):
            logger.info('[val] {}: {:.3f}'.format(k, v))
        torch.save(self.model.state_dict(), "cvae_model.pt")
        logger.info('Done.')

    def train(self):
        self.model.train()
        for self.epoch in range(self.config.num_epochs):
            logger.info('Starting epoch {}'.format(self.epoch))
            for i, batch in enumerate(self.train_loader):
                batch = [tensor.cuda() for tensor in batch]
                obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, val_mask, loss_mask, seq_start_end, nei_num_index, nei_num = batch
                loss_mask = loss_mask[-self.config.pred_len:]
                for param in self.model.parameters(): param.grad = None

                pred_traj_rel_recon, prior_mu, prior_logvar, recog_mu, recog_logvar, pred_mean, pred_var = self.model(obs_traj_rel, pred_traj_gt_rel, obs_traj, nei_num_index, nei_num, predict=False)
                kl_loss, l2_loss, nll_loss = loss_function_timo(pred_mean, pred_var, recog_mu, recog_logvar, prior_mu, prior_logvar, pred_traj_rel_recon, pred_traj_gt_rel, loss_mask, pred_traj_gt, seq_start_end)
                # For Collision_loss
                print("pred_traj_rel_recon", pred_traj_rel_recon.shape)
                print("obs_traj", obs_traj.shape)
                pred_traj_fake_abs = relative_to_abs(pred_traj_rel_recon, obs_traj[-1])
                coll_loss_count = coll_smoothed_loss(pred_traj_fake_abs, seq_start_end, nei_num_index)
                losses = kl_loss + 0.1 * nll_loss + 1000 * coll_loss_count
                losses.backward()
                self.optimizer.step()
                wandb.log({"Coll Count": coll_loss_count.item(), "Train NLL": nll_loss.item(), "Train KL": kl_loss.item(), "Train Total": losses.item()})

            if self.epoch % self.config.check_after_num_epochs == 0:
                #self.save_model()
                self.save_model_old()





if __name__ == '__main__':
    mpis = MPIS()
    #mpis.train()
    mpis.test_old()
