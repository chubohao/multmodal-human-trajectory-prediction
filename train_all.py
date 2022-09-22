from train import Preparation
from evaluate_model import main
from models.cvae_var import CVAE
import os
import torch
from config import *

import argparse




parser = argparse.ArgumentParser()

parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)
parser.add_argument('--model_path', default="DIR", type=str)
parser.add_argument('--dset', default="dataset", type=str)

datasets = ['univ', 'hotel', 'zara1', 'zara2', 'eth']

if __name__ == "__main__":
    for data in datasets:
        print("START DATASET: " + str(data))
        model = CVAE
        prep = Preparation(model, data)
        prep.train()
        del prep

        ##evaluate
        config = Config()
        config.experiment_name = config.experiment_name + '/' + data
        path = os.path.join(config.DIR, config.experiment_name)
        config.model_path = path
        config.output_dir = config.model_path
        config.checkpoint_start_from = config.model_path + '/checkpoint_with_model.pt'
        DIR = os.path.join(config.home, config.checkpoint_start_from)



        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        args = parser.parse_args()
        args.model_path = DIR
        args.dset = data
        main(args)
