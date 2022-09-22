
import copy
import logging
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim

import wandb
from config import *
from data.loader import data_loader
from evaluate_model import evaluate
from losses import l2_loss, coll_smoothed_loss
from models.LSTM import LSTM, LSTMaddNoise
from models.Coll_Loss_one import CoLoss
#from models.cvae_var import CVAE
from models.CVAE_TEST_SIZES import CVAE
#from models.CVAE_TEST_SIZES import CVAE
#from models.mpis_cvae_lstm_ar_sigma import CVAE
from utils import get_dset_path, relative_to_abs
from mpis_losses import loss_function_gnll

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)



def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype


class Preparation:
    def __init__(self, model, data=None):
        self.config = Config()
        if data:
            self.config.dataset_name = data
            group_name = self.config.experiment_name
            self.config.experiment_name = self.config.experiment_name + '/' + data #+ "-" + str(self.config.learning_rate) \
                                          #+ "-" + str(self.config.coeff_coll_loss) + "-" + str(self.config.coeff_nll) + "-" + str(self.config.coeff_kldiv)
            path = os.path.join(self.config.DIR, self.config.experiment_name)
            if not os.path.exists(path):
                os.makedirs(path)
            self.config.model_path = path
            self.config.output_dir = self.config.model_path
            self.config.checkpoint_start_from = self.config.model_path + '/checkpoint_with_model.pt'
            wandb.init(project="mpis", entity="mpis3", name=self.config.experiment_name, reinit=True, group=group_name)
        else:
            self.config.experiment_name = self.config.experiment_name  + '/' + self.config.dataset_name + "-" + str(self.config.learning_rate) \
                   + "-" + str(self.config.coeff_coll_loss) + "-" + str(self.config.coeff_nll) + "-" + str(self.config.coeff_kldiv)
            wandb.init(project="mpis", entity="mpis3", name=self.config.experiment_name, reinit=True)
            # print('no_wandb')
        seed = self.config.seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        os.environ['CUDA_VISIBLE_DEVICES'] = self.config.gpu_num
        train_path = get_dset_path(self.config.dataset_name, 'train')
        val_path = get_dset_path(self.config.dataset_name, 'val')
        long_dtype, float_dtype = get_dtypes(self.config)
        logger.info("Initializing train dataset")
        self.train_dset, self.train_loader = data_loader(self.config, train_path, augment=self.config.augment)
        logger.info("Initializing val dataset")
        _, self.val_loader = data_loader(self.config, val_path)
        self.iterations_per_epoch = len(self.train_loader)
        logger.info(
            'There are {} iterations per epoch'.format(self.iterations_per_epoch)
        )

        #self.model = CoLoss(self.config)

        #cvae old
        #in_size = 12
        #cond_size = 8
        #extra_layer_size = 400
        #latent_size = 2
        #self.model = CVAE(in_size, latent_size, cond_size, extra_layer_size)

        self.model = model()

        self.model.type(float_dtype).train()
        logger.info('Here is the generator:')
        logger.info(self.model)

        # Log model
        # wandb.watch(self.generator, log='all')


        if self.config.adam:
            print('Learning with ADAM!')
            betas_d = (0.5, 0.9999)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, betas=betas_d)
        else:
            print('Learning with RMSprop!')
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=0.9)
        restore_path = None
        if self.config.checkpoint_start_from is not None:
            restore_path = self.config.checkpoint_start_from
        elif self.config.restore_from_checkpoint == True:
            restore_path = os.path.join(self.config.output_dir,
                                        '%s_with_model.pt' % self.config.checkpoint_name)

        #if os.path.isfile(restore_path):
        #    logger.info('Restoring from checkpoint {}'.format(restore_path))
        #    self.checkpoint = torch.load(restore_path)
        #    #self.model.load_state_dict(self.checkpoint['state'])
        #    #self.optimizer.load_state_dict(self.checkpoint['optim_state'])
        #    self.t = self.checkpoint['counters']['t']
        #    self.epoch = self.checkpoint['counters']['epoch']
        #    self.checkpoint['restore_ts'].append(self.t)
        #else:
        # Starting from scratch, so initialize checkpoint data structure
        self.t, self.epoch = 0, 0
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

    def check_accuracy(self, loader, generator):  # TODO Change this!
        metrics = {}
        generator.eval()  # will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will work in eval mode instead of training mode.

        ade, fde, act, _ = evaluate(self.config, loader, generator)
        metrics['act'] = act
        metrics['ade'] = ade
        metrics['fde'] = fde
        mean = (act+ ade)/2
        metrics['mean'] = mean
        generator.train()
        wandb.log({"ade":  metrics['ade'], "fde": metrics['fde'],
                   "act_best_ade": metrics['act'], "Mean_ade_act": metrics['mean']})
        return metrics

    def print_stats(self):
        dictlist = 'Epoch = {}, t = {} '.format(self.epoch, self.t) + '[D] '
        dictlist += ' [G] '

        for k, v in sorted(self.losses.items()):
            self.checkpoint['losses'][k].append(v)
            dictlist += ' ' + '{}: {:.6f}'.format(k, v)

        logger.info(dictlist)

        self.checkpoint['losses_ts'].append(self.t)

    def save_model(self):
        self.checkpoint['counters']['t'] = self.t
        print("t: ", self.t)
        self.checkpoint['counters']['epoch'] = self.epoch
        self.checkpoint['sample_ts'].append(self.t)

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
            self.checkpoint['best_t'] = self.t
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

    def coeff_annealing(self, epoch, num_epochs, max_coeff):
        ratio = max_coeff / num_epochs
        period = (epoch + 1) / num_epochs
        step = ratio * period # linear schedule
        coeff = epoch * step
        return coeff

    def model_step(self, batch, epoch):
        #daten aus batch lesen
        #obs_traj = observerd trajectory
        #pred_traj_gt = predicted trajectory ground truth
        #obs_traj_rel = observerd trajectory relative (velocity)
        #pred_traj_gt_rel = predicted trajectory ground truth relative (velocity)
        #val_mask
        #loss_mask
        #seq_start_end
        #nei_num_index
        #nei_num = number of neighbors
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, val_mask,\
        loss_mask, seq_start_end, nei_num_index, nei_num, = batch
        l2_loss_rel = []
        losses = {}

        for param in self.model.parameters(): param.grad = None  # same as optimizer.zero_grad() but faster!
        #loss mask only for last 12(prediction length)
        loss_mask = loss_mask[-self.config.pred_len :]
        # model input is observed(first 8 observed) | groundtruth(last 12 observerd - to be predicted)
        model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)

        for _ in range(self.config.best_k):  # variety loss, look in paper
            #CVAE
            #pred_traj_fake_rel, z, prior_mu, prior_logvar, recog_mu, recog_logvar, pred_mean, pred_var = self.model(
                #obs_traj_rel, pred_traj_gt_rel, obs_traj, nei_num_index, nei_num, testing=False)

            #CVAE bohao
            z, pred_traj_fake_rel, prior_mu, prior_logvar, recog_mu, recog_logvar, pred_mean, pred_var = self.model(
                obs_traj_rel, pred_traj_gt_rel, obs_traj, nei_num_index, nei_num, predict=False)


        mmd, kl_loss, l2_loss, nll_loss = loss_function_gnll(z, pred_mean, pred_var, recog_mu, recog_logvar, prior_mu,
                                                       prior_logvar, pred_traj_fake_rel, pred_traj_gt_rel, loss_mask,
                                                       pred_traj_gt, seq_start_end)

        pred_traj_fake_abs = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
        batchsize = obs_traj.shape[1]
        coll_loss_count = coll_smoothed_loss(pred_traj_fake_abs, seq_start_end, nei_num_index)

        #print("pred rel traj: " +str(pred_traj_fake_rel.shape))
        #print("pred rel traj: " + str(pred_traj_fake_rel))
        #print("mean coll: ", coll_loss_count)

        coeff_mmd = self.coeff_annealing(epoch, self.config.num_epochs, self.config.coeff_mmd)
        #print("mmd coeff: ", coeff_mmd)
        coeff_mmd = torch.tensor(coeff_mmd).cuda()

        loss = coeff_mmd * mmd + self.config.coeff_l2 * l2_loss + self.config.coeff_kldiv * kl_loss + self.config.coeff_nll * nll_loss + self.config.coeff_coll_loss * coll_loss_count

        wandb.log({"MMD Loss": mmd.item(), "L2 Loss": l2_loss.item(), "Coll Count": coll_loss_count.item(), "Train NLL": nll_loss.item(), "Train KL": kl_loss.item(),
                   "Train Total": loss.item()})

        loss.backward()
        self.optimizer.step()


        return losses

    def train(self):
        self.t_step = 0
        while self.epoch < self.config.num_epochs:
            self.t_step = 0
            logger.info('Starting epoch {}'.format(self.epoch))
            for batch in self.train_loader:
                batch = [tensor.cuda() for tensor in batch]
                #print("batch", len(batch))
                #print(batch[0].size())
                #compute losses, but weights are not yet adapted?
                self.losses = self.model_step(batch, self.epoch)
                self.t_step += 1
            if self.epoch % self.config.check_after_num_epochs == 0:
                print("SAVING")
                self.save_model()
            self.epoch += 1

if __name__ == '__main__':
    model = CVAE
    prep = Preparation(model)
    prep.train()
