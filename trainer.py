import os
import time
import math
import random
import numpy as np
import logging as log
from datetime import datetime

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
from validator import Validator
from visualization.sdf_slice import SDF_Slice


from torch.utils.tensorboard import SummaryWriter

class Trainer(object):
	def __init__(self, config):
		self.ef_dim = 32
		self.gf_dim = 128
		self.z_dim = 512
		self.point_dim = 3
		self.camera_dim = 9
		self.checkpoint_dir = config.checkpoint_dir
		self.model_param_path = config.model_param_path

		#获取data loader
		_loadData = loadData(config)
		self.train_dataloader, self.validate_dataloader, self.slice_dataloader= _loadData.load(config)

		#检测GPU
		if torch.cuda.is_available():
			print("gpu")
			self.device = torch.device('cuda')
			torch.backends.cudnn.benchmark = True
		else:
			self.device = torch.device('cpu')

		#build model
		print("Init Network...")
		self.network = im_bas_relief_network(self.ef_dim, self.gf_dim, self.z_dim, self.point_dim, self.camera_dim)
		self.network.to(self.device)

		# init validator
		self.validator = Validator(self.validate_dataloader,self.device,self.network)
		# self.network.double()
		print("Network Prepared!")
		#print params
		#for param_tensor in self.network.state_dict():
		#	print(param_tensor, "\t", self.network.state_dict()[param_tensor].size())
		self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))
		#pytorch does not have a checkpoint manager
		#have to define it myself to manage max num of checkpoints to keep
		self.max_to_keep = 10
		self.checkpoint_path = os.path.join(self.checkpoint_dir, self.model_dir)
		self.checkpoint_name='IM_Bas_Relief.model'
		self.checkpoint_manager_list = [None] * self.max_to_keep
		self.checkpoint_manager_pointer = 0

		# tensor board
		if not os.path.exists(self.checkpoint_path):
			os.makedirs(self.checkpoint_path)
		self.log_fname = f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
		self.log_dir = os.path.join(self.checkpoint_path, "logs"+self.log_fname)
		self.writer = SummaryWriter(self.log_dir, purge_step=0)
		
		#loss
		print("Loss definition")
		def network_loss(G,point_value):
			return torch.mean((G-point_value)**2)
		self.loss = network_loss

	@property
	def model_dir(self):
		return "IM_Bas_Relief_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

	def train(self, config):
		# 需要时启用
		#load previous checkpoint
		# checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
		# if os.path.exists(checkpoint_txt):
		# 	fin = open(checkpoint_txt)
		# 	model_dir = fin.readline().strip()
		# 	fin.close()
		# 	self.network.load_state_dict(torch.load(model_dir))
		# 	print(" [*] Load SUCCESS")
		# else:
		# 	print(" [!] Load failed...")
		
		
		print("\n\n----------net summary----------")
		print("training epoches   ", config.epoch)
		print("-------------------------------\n\n")
		
		self.network.to(self.device)
		start_time = time.time()
		assert config.epoch==0 or config.iteration==0

		for epoch in range(0, config.epoch):
			self.network.train()
			avg_loss_sp = 0
			avg_num = 0


			for n_iter, data in enumerate(self.train_dataloader):
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
				camera=camera.type(torch.float32)
				
				# pts = pts[:,0:10,:]
				# sdf = sdf[:,0:10]
				# camera = camera[:,0:10,:]
				# print("pts: ",pts.shape)
				# print("sdf: ",sdf.shape)
				# print("mpts: ",mpts.shape)
				# print("camera: ", camera.shape)

				self.network.zero_grad()
				_, net_out = self.network(mpts, None, pts, camera, is_training=True)
				net_out = net_out.squeeze(dim=2)
				# print(net_out.shape)
				errSP = self.loss(net_out, sdf)

				errSP.backward()
				self.optimizer.step()

				avg_loss_sp += errSP.item()
				avg_num += 1

			print(" Epoch: [%2d/%2d] time: %4.4f, loss_sp: %.6f" % (epoch, config.epoch, time.time() - start_time, avg_loss_sp/avg_num))
			self.writer.add_scalar('Loss', avg_loss_sp/avg_num, epoch)
			if epoch%10==5:
				# validation
				self.validation(epoch)
			if epoch%20==10:
				if not os.path.exists(self.checkpoint_path):
					os.makedirs(self.checkpoint_path)
				save_dir = os.path.join(self.checkpoint_path,self.checkpoint_name+"-"+str(epoch)+".pth")
				self.checkpoint_manager_pointer = (self.checkpoint_manager_pointer+1)%self.max_to_keep
				#delete checkpoint
				if self.checkpoint_manager_list[self.checkpoint_manager_pointer] is not None:
					if os.path.exists(self.checkpoint_manager_list[self.checkpoint_manager_pointer]):
						os.remove(self.checkpoint_manager_list[self.checkpoint_manager_pointer])
				#save checkpoint
				torch.save(self.network.state_dict(), save_dir)
				#update checkpoint manager
				self.checkpoint_manager_list[self.checkpoint_manager_pointer] = save_dir
				#write file
				checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
				fout = open(checkpoint_txt, 'w')
				for i in range(self.max_to_keep):
					pointer = (self.checkpoint_manager_pointer+self.max_to_keep-i)%self.max_to_keep
					if self.checkpoint_manager_list[pointer] is not None:
						fout.write(self.checkpoint_manager_list[pointer]+"\n")
				fout.close()

		# 写入训练模型
		if not config.slice:
			if not os.path.exists(self.checkpoint_path):
				os.makedirs(self.checkpoint_path)
			save_dir = os.path.join(self.checkpoint_path,self.checkpoint_name+str(config.epoch)+".pth")
			self.checkpoint_manager_pointer = (self.checkpoint_manager_pointer+1)%self.max_to_keep
			#delete checkpoint
			if self.checkpoint_manager_list[self.checkpoint_manager_pointer] is not None:
				if os.path.exists(self.checkpoint_manager_list[self.checkpoint_manager_pointer]):
					os.remove(self.checkpoint_manager_list[self.checkpoint_manager_pointer])
			#save checkpoint
			torch.save(self.network.state_dict(), save_dir)
			#update checkpoint manager
			self.checkpoint_manager_list[self.checkpoint_manager_pointer] = save_dir
			#write file
			checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
			fout = open(checkpoint_txt, 'w')
			for i in range(self.max_to_keep):
				pointer = (self.checkpoint_manager_pointer+self.max_to_keep-i)%self.max_to_keep
				if self.checkpoint_manager_list[pointer] is not None:
					fout.write(self.checkpoint_manager_list[pointer]+"\n")
			fout.close()
			self.writer.close()

	def validation(self, epoch):
		self.network.eval()
		v_loss = self.validator.validate(epoch)
		print("Validation at epoch : ", epoch, "validation! and loss is : ", v_loss)
		self.writer.add_scalar('Validation loss', v_loss, epoch)
		return

	def slice_sdf(self, config):
		slice = SDF_Slice(config, self.network, self.model_param_path, self.slice_dataloader, self.device, self.checkpoint_path)
		slice.get_slice()