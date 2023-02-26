import os
import time
import math
import random
import numpy as np
import logging as log
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import mcubes
from utils import *
import datetime
from loadData import loadData
from models.Bas_Relief_Net import im_bas_relief_network


class Validator(object):
    def __init__(self, dataloader, device, net):
        self.dataloader = dataloader
        self.device = device
        self.net = net
    
    # loss
    print("Loss definition")
    def network_loss(self, G, point_value):
        return torch.mean((G-point_value)**2)

    def validate(self, epoch):
        avg_loss_sp = 0
        avg_num = 0

        for n_iter, data in enumerate(self.dataloader):
            # print("iteration: ", n_iter)
            # print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
            # print("epoch: ",epoch,"iter: ",n_iter)
            # 采样点总数
            pts = data[0].to(self.device)
            sdf = data[1].to(self.device)
            # 模型表面的点
            mpts = data[2].to(self.device)
            # 相机参数
            camera = data[3].to(self.device)
            camera = camera.type(torch.float32)

            # pts = pts.to(device=self.device, dtype=torch.float)

            # print("pts: ",pts.shape)
            # print("sdf: ",sdf.shape)
            # print("mpts: ",mpts.shape)
            # print("camera: ", camera.shape)
            _, net_out = self.net(mpts, None, pts, camera, is_training=True)
            net_out = net_out.squeeze(dim=2)
            # print(net_out.shape)
            errSP = self.network_loss(net_out, sdf)

            avg_loss_sp += errSP.item()
            avg_num += 1

        return avg_loss_sp/avg_num
