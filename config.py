import math
import os
from os.path import expanduser
#home = expanduser("sotiroel")


class Config:
    def __init__(self):
        self.coeff_coll_loss = 100
        self.coeff_kldiv = 1
        self.coeff_nll = 100
        #self.DIR = home +'/Documents/MPIS/'
        #self.DIR = "home/WIN-UNI-DUE/sotiroel/Documents/MPIS"
        self.home = "/tmp/CoLoss1/"
        self.DIR = "saved_models"
        self.experiment_name = 'CVAE'
        # Dataset options
        self.dataset_name = 'zara2'
        path = os.path.join(self.DIR, self.experiment_name, self.dataset_name)
        if not os.path.exists(path):
            os.makedirs(path)
        self.model_path = path

        self.nabs = False # absolute position but shift the origin to the latest observed time slot
        self.delim = 'tab'
        self.loader_num_workers = 4
        self.obs_len = 8
        self.pred_len = 12
        self.skip = 1
        self.seed = 42

        #normalization in pre processing
        self.min_x = -7.69
        self.min_y = -10.31
        self.img_h = 28
        self.img_w = 28
        self.collision_distance = 0.2
        # Model Options

        self.l2_loss = 10.

        self.best_k = 1
        #try set to 20
        self.num_samples = 1
        # Model Options
        self.learning_rate = 0.01 #0.000001
        #0.001
        self.clipping_threshold_g = 0.0
        self.adam = 1
        self.augment = True
        self.all_rel_persons = True
        self.gpu_num = "0"
        self.device='cuda'
        # Optimization
        self.batch_size = 32 #32 for local pc?
        self.num_epochs = 100
        # Loss Options
        self.l2_loss_weight = 1
        # Output
        self.output_dir = self.model_path
        self.checkpoint_name = 'checkpoint'
        self.checkpoint_start_from = os.path.join(self.model_path, 'checkpoint_with_model.pt')
        self.restore_from_checkpoint = False
        self.num_samples_check = 5000
        self.check_after_num_epochs = 5
        # Misc
        self.use_gpu = 1
        self.timing = 0
        # GCN
        self.nei_std = 0.01
        self.rela_std = 0.3
        self.WAq_std = 0.05
        self.ifbias_gate = True
        self.WAr_ac = ''
        self.ifbias_WAr = False
        self.input_size = 2
        self.ifbias_nei = False
        self.hidden_dot_size = 16
        self.nei_hidden_size = 16
        self.nei_drop = 0
        self.nei_layers = 1
        self.nei_drop = 0
        self.nei_ac = 'relu'
        self.rela_layers = 1
        self.rela_input = 2
        self.rela_hidden_size = 16
        self.rela_ac = 'relu'
        self.rela_drop = 0.0
        self.std_in = 0.2
        self.std_out = 0.1

        self.passing_time = 1
        self.balance_a = 0.05
        # self.balance_a = 1.
        self.ifdebug = False