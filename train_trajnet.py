import json
import torch
import numpy as np
from config import *

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
from losses import l2_loss
from models.LSTMwithNoise import LSTMwithNoise
from utils import get_dset_path

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

def load_data(filepath):


    data = []
    with open(filepath) as f:
        for line in f:
            data.append(json.loads(line))
    #print("data", data)
    return data

#preprocessing: find scenes, persons, neighbors etc.
def preprocess_data(data):
    #first there are primary pedestrians categorizing the scene
    # 1 = static
    # 2 = Linear
    # 3 = Interacting
        #3.1 = Leader Follower
        #3.2 = Collision Avoidance
        #3.3 = Group
        #3.4 = Other
    # 4 = Non Interacting
    new_data = [] #torch.tensor([]).cuda()
    loss_mask = []
    for scene in data:
        scene_traj = torch.tensor([]).cuda()
        scene_loss_mask = torch.tensor([]).cuda()
        #get person
        #get start and end
        #find all frames of person
        #write to tensor all coordinates of persons of scene
        if not 'scene' in scene:
            print("data: ", len(new_data))
            return new_data, loss_mask
        else:
            scene = scene['scene']

            scene_id = scene['id']
            person = scene['p']
            start = scene['s']
            end = scene['e']
            tag = scene['tag']
            #tag2 = scene[tag]
            print("id ", scene['id'])
            print("scene: ", scene)
            person_traj = torch.tensor([]).cuda()
            #for person in scene? how?
            #or for frame in scene
            #make tensors for each person?
            #or dict
            person_ids = []
            person_traj_list = []
            for track in data:
                #dict?
                if 'track' in track:
                    track = track['track']
                    if track['f'] >= start and track['f'] <= end:
                        person_id = track['p']
                        coord_tensor = torch.tensor([track['x'], track['y']]).cuda()

                        if person_id not in person_ids:
                            #print("new person " + str(person_id))
                            person_ids.append(person_id)
                            person_traj = torch.tensor([]).cuda()
                            person_traj = torch.cat((person_traj, coord_tensor.unsqueeze(0)), axis=1)
                            person_traj_list.append(person_traj)
                            #print("updatet traj list", person_traj_list)
                        else:
                            #print("known person " + str(person_id))
                            person_traj = person_traj_list[person_ids.index(person_id)]
                            person_traj = torch.cat((person_traj, coord_tensor.unsqueeze(0)), axis=0)
                            person_traj_list[person_ids.index(person_id)] = person_traj
                            #print("updatet traj list", person_traj_list)

            print("person: ", person_traj.shape)

            #append person traj to tensor
            person_loss_mask = torch.tensor([]).cuda()
            for traj in person_traj_list:
                person_loss_mask = torch.ones([traj.shape[0], 2]).cuda()
                if traj.shape[0] != 21:
                    zero_cat_tensor = torch.zeros([21-traj.shape[0], 2]).cuda()
                    person_loss_mask = torch.cat((person_loss_mask, zero_cat_tensor), axis=0)
                    traj = torch.cat((traj, zero_cat_tensor), axis=0)
                    print("traj: ", traj.shape)
                scene_traj = torch.cat((scene_traj, traj.unsqueeze(0)), axis=0)
                scene_loss_mask = torch.cat((scene_loss_mask, person_loss_mask.unsqueeze(0)), axis=0)

        print("scene: ", scene_traj.unsqueeze(0).shape)
        #new_data = torch.cat((new_data, scene_traj.unsqueeze(0)), axis=0)
        loss_mask.append(scene_loss_mask)
        new_data.append(scene_traj)
    print("data: ", len(new_data))
    return new_data, loss_mask

def get_rel_traj(data):
    rel_traj = []
    for scene in data:
        scene_traj = torch.tensor([]).cuda()
        for person in scene:
            person_traj = torch.tensor([]).cuda()
            prev_step = None
            for step in person:
                if prev_step is None or step is torch.tensor([0,0]).cuda():
                    rel_step = torch.tensor([0, 0]).cuda()
                else:
                    rel_step = step - prev_step
                person_traj = torch.cat((person_traj, rel_step.unsqueeze(0)), axis=0)
                print("rel_step: ", rel_step)
                print("person traj: ", person_traj)
                prev_step = step
            scene_traj = torch.cat((scene_traj, person_traj.unsqueeze(0)), axis=0)
        rel_traj.append(scene_traj)
    return rel_traj

class Trajnet:
    def __init__(self, trainmodel):
        self.obs_len = 9
        self.pred_len = 12
        self.learning_rate = 0.0001
        self.model = trainmodel
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor


        self.model.type(float_dtype).train()

        print('Learning with ADAM!')
        betas_d = (0.5, 0.9999)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=betas_d)

        wandb.init(project="mpis", entity="mpis3", name="trajnet_test", reinit=True, group="trajnet")
    def train(self, traj, traj_rel, loss_mask):
        num_epochs = 100
        epoch = 0
        while epoch < num_epochs:
            logger.info('Starting epoch {}'.format(epoch))
            for scene_num in range(len(traj_rel)):
                loss = self.model_step(traj[scene_num], traj_rel[scene_num], loss_mask[scene_num])
            epoch += 1


    def model_step(self, traj, traj_rel, loss_mask):
        l2_loss_rel = []
        losses = {}

        for param in self.model.parameters(): param.grad = None  # same as optimizer.zero_grad() but faster!
        # loss mask only for last 12(prediction length)
        print("loss_mask: ", loss_mask.shape)
        loss_mask = loss_mask.permute(1,0,2)[-self.pred_len:]
        print("loss_mask new: ", loss_mask.shape)
        print("traj_rel: ", traj_rel.shape)
        obs_traj_rel = traj_rel.permute(1, 0, 2)[0:self.obs_len]
        print("obs_traj_rel: ", obs_traj_rel.shape)
        pred_traj_gt_rel = traj_rel.permute(1,0,2)[-self.pred_len:]
        obs_traj = traj.permute(1,0,2)[0:self.obs_len]
        pred_traj_gt = traj.permute(1,0,2)[-self.pred_len:]
        model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)

        for k in range(3):
            pred_traj_fake_rel = self.model(obs_traj_rel, pred_traj_gt_rel)
            print("pred traj: ", pred_traj_fake_rel.shape)
            print("loss_mask: ", model_input[-self.pred_len:].shape)
            print("loss_mask: ", loss_mask.shape)
            lossfct = torch.nn.MSELoss()
            l2_loss_rel = lossfct(pred_traj_fake_rel, pred_traj_gt_rel)

        loss = l2_loss_rel
        loss.backward()
        losses['L2_loss'] = l2_loss_rel.item()
        # optimize and update weights
        self.optimizer.step()
        wandb.log({"Train L2": l2_loss_rel.item()})
        return losses


if __name__ == '__main__':
    filepath = "./datasets/trajnet/ECCV/train/biwi_hotel.ndjson"
    data = load_data(filepath)
    #puts all traj==21 of persons of scene to tensors
    #or better fill with -999?
    traj,loss_mask = preprocess_data(data)

    traj_rel = get_rel_traj(traj)
    print("tensor: ", len(traj_rel))
    for scene in traj_rel:
        print("#p in scene: ", scene.shape)
    for scene in loss_mask:
        print("loss mask: ", scene)

    model = LSTMwithNoise()
    trajnet = Trajnet(model)
    trajnet.train(traj, traj_rel, loss_mask)
    #model = LSTMwithNoise()
    #trajnet = trajnet(model)
    #trajnet.train(data)