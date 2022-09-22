#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import random
from config import Config
from math import cos, sin, pi


class AugmentSelection:

    def __init__(self, flip=False, ):
        self.flip = flip  # shift y-axis
        self.config = Config()

    @staticmethod
    def random():
        flip = random.uniform(0., 1.) >= 0.5
        return AugmentSelection(flip)

    @staticmethod
    def unrandom():
        flip = False
        return AugmentSelection(flip)

    def affine(self):
        # the main idea: we will do all image transformations with one affine matrix.
        # this saves lot of cpu and make code significantly shorter
        # same affine matrix could be used to transform joint coordinates afterwards
        # look https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/warp_affine/warp_affine.html

        #  width, height = img_shape
        degree = random.uniform(-1., 1.) * 15 if self.flip else 0.
        # degree = 7.

        A = cos(degree / 180. * pi)
        B = sin(degree / 180. * pi)

        rotate = np.array([[A, -B, 0],
                           [B, A, 0],
                           [0, 0, 1.]])

        center2zero = np.array([[1., 0., -(self.config.img_h / 2. -1)],
                                [0., 1., -(self.config.img_h / 2. -1)],
                                [0., 0., 1.]])

        flip = random.uniform(0., 1.) >= 0.5
        flip = -1. if flip else 1.
        flip_v = np.array([[flip, 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]], dtype=np.float32)

        flip = random.uniform(0., 1.) >= 0.5
        flip = -1. if flip else 1.
        flip_h = np.array([[1, 0., 0.],
                           [0., flip, 0.],
                           [0., 0., 1.]], dtype=np.float32)

      #  translatx = random.uniform(-1., 1.) * 15 if self.flip else 0.
      #  translaty = random.uniform(-1., 1.) * 15 if self.flip else 0.

        center2center = np.array([[1., 0., (self.config.img_w / 2. -1)],
                                 [0., 1.,  (self.config.img_w / 2. -1)],
                                 [0., 0., 1.]])

        # order of combination is reversed
        # combined =   center2center  @ rotate @ scale @ flip @center2zero# @ - matmul
       # combined = flip @ rotate @ center2zero  # @ - matmul
       #  combined = center2center @ rotate @ flip_v @ flip_h @ center2zero  # @ - matmul
        combined = center2center @ flip_v @ flip_h @ center2zero  # @ - matmul
       # combined = flip   # @ - matmul

        return combined[0:2]  # 3th row is not important anymore



class Transformer:
    @staticmethod
    def print_traj(data_obs, data_pred_gt):  # only for debug
        data_obs = data_obs[0] #first batch to print
        data_pred_gt = data_pred_gt[0]
        fig, ax = plt.subplots()
        for x_data in data_obs:
            x = x_data[0, :]
            x = x[np.nonzero(x)]
            y = x_data[1, :]
            y = y[np.nonzero(y)]
            ax.plot(x, y, 'g', alpha=.3)
            ax.plot(x[:-1], y[:-1], 'g*', alpha=.5)
            ax.plot(x[-1], y[-1], 'gX', alpha=.5)

        for x_data in data_pred_gt:
            if x_data.any():
                x = x_data[0, :]
                x = x[np.nonzero(x)]
                y = x_data[1, :]
                y = y[np.nonzero(y)]
                ax.plot(x, y, 'g', alpha=.3)
                ax.plot(x[:-1], y[:-1], 'b*', alpha=.5)
                ax.plot(x[-1], y[-1], 'bX', alpha=.5)

        plt.show()
        plt.close

    @staticmethod
    def traj_matmul(original_points, M):
        for j , batch in enumerate(original_points):
            for i, o in enumerate(batch):
               # for k, o in enumerate(tr):
              #  o = o[np.nonzero(o)]
                mask = np.where(o.sum(axis=0) != 0, 1., 0.)
                ones = np.ones_like(o[0, :])
                ones = np.expand_dims(ones, axis=0)

                tmp= np.concatenate((o, ones),axis=0) # we reuse 3rd column in
                # completely different way here, it is hack for matmul with M
                original_points[j][i] = np.matmul(M, tmp)  * mask # transpose for multiplikation
        return original_points

    @staticmethod
    def transform(data):

        aug = AugmentSelection.random()
        M = aug.affine()
        data_obs, data_pred_gt = data
        # Transformer.print_traj( data_obs, data_pred_gt)
        # print(data_obs[0][:,1].max())
        # warp key points
        data_obs = Transformer.traj_matmul(data_obs, M)
        data_pred_gt = Transformer.traj_matmul(data_pred_gt, M)
        # Transformer.print_traj(data_obs, data_pred_gt)
        # print(data_obs[0][:,1].max())


        return data_obs, data_pred_gt
