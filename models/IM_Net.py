import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import numpy as np


class im_generator(nn.Module):
	def __init__(self, z_dim, point_dim, camera_dim, gf_dim):
		super(im_generator, self).__init__()
		self.z_dim = z_dim
		self.point_dim = point_dim
		self.gf_dim = gf_dim
		# self.linear = nn.Linear(self.z_dim+self.point_dim, 1, bias=True)
		self.linear_1 = nn.Linear(self.z_dim+self.point_dim, self.gf_dim*8, bias=True)
		self.linear_2 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_3 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_4 = nn.Linear(self.gf_dim*8, self.gf_dim*4, bias=True)
		self.linear_5 = nn.Linear(self.gf_dim*4, self.gf_dim*2, bias=True)
		self.linear_6 = nn.Linear(self.gf_dim*2, self.gf_dim*1, bias=True)
		self.linear_7 = nn.Linear(self.gf_dim*1, 1, bias=True)
		nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_1.bias,0)
		nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_2.bias,0)
		nn.init.normal_(self.linear_3.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_3.bias,0)
		nn.init.normal_(self.linear_4.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_4.bias,0)
		nn.init.normal_(self.linear_5.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_5.bias,0)
		nn.init.normal_(self.linear_6.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_6.bias,0)
		nn.init.normal_(self.linear_7.weight, mean=1e-5, std=0.02)
		nn.init.constant_(self.linear_7.bias,0)

	def forward(self, points, camera_param, z, is_training=False):
		zs = z.view(-1,1,self.z_dim).repeat(1,points.size()[1],1)
		# print("camera: ",camera_param)
		# print('x.dtype: ',camera_param.dtype)
		# print("points: ",points)
		# print('x.dtype: ',points.dtype)
		pointz = torch.cat([points,zs],2)
		# print(pointz.shape)
		# print(pointz)
		# pointz = torch.cat([pointz,camera_param],2)
		# print(pointz.shape)
		# print(pointz)
		l1 = self.linear_1(pointz)
		l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)

		l2 = self.linear_2(l1)
		l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)

		l3 = self.linear_3(l2)
		l3 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)

		l4 = self.linear_4(l3)
		l4 = F.leaky_relu(l4, negative_slope=0.02, inplace=True)

		l5 = self.linear_5(l4)
		l5 = F.leaky_relu(l5, negative_slope=0.02, inplace=True)

		l6 = self.linear_6(l5)
		l6 = F.leaky_relu(l6, negative_slope=0.02, inplace=True)

		l7 = self.linear_7(l6)

		# l7 = self.linear(pointz)
		# l7 = torch.clamp(l7, min=-1, max=1)
		# l7 = torch.max(torch.min(l7, l7*0.01+0.99), l7*0.01)
		
		return l7

class im_encoder(nn.Module):
	def __init__(self, ef_dim, z_dim):
		super(encoder, self).__init__()
		self.ef_dim = ef_dim
		self.z_dim = z_dim
		self.conv_1 = nn.Conv3d(1, self.ef_dim, 4, stride=2, padding=1, bias=False)
		self.in_1 = nn.InstanceNorm3d(self.ef_dim)
		self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim*2, 4, stride=2, padding=1, bias=False)
		self.in_2 = nn.InstanceNorm3d(self.ef_dim*2)
		self.conv_3 = nn.Conv3d(self.ef_dim*2, self.ef_dim*4, 4, stride=2, padding=1, bias=False)
		self.in_3 = nn.InstanceNorm3d(self.ef_dim*4)
		self.conv_4 = nn.Conv3d(self.ef_dim*4, self.ef_dim*8, 4, stride=2, padding=1, bias=False)
		self.in_4 = nn.InstanceNorm3d(self.ef_dim*8)
		self.conv_5 = nn.Conv3d(self.ef_dim*8, self.z_dim, 4, stride=1, padding=0, bias=True)
		nn.init.xavier_uniform_(self.conv_1.weight)
		nn.init.xavier_uniform_(self.conv_2.weight)
		nn.init.xavier_uniform_(self.conv_3.weight)
		nn.init.xavier_uniform_(self.conv_4.weight)
		nn.init.xavier_uniform_(self.conv_5.weight)
		nn.init.constant_(self.conv_5.bias,0)

	def forward(self, inputs, is_training=False):
		d_1 = self.in_1(self.conv_1(inputs))
		d_1 = F.leaky_relu(d_1, negative_slope=0.02, inplace=True)

		d_2 = self.in_2(self.conv_2(d_1))
		d_2 = F.leaky_relu(d_2, negative_slope=0.02, inplace=True)
		
		d_3 = self.in_3(self.conv_3(d_2))
		d_3 = F.leaky_relu(d_3, negative_slope=0.02, inplace=True)

		d_4 = self.in_4(self.conv_4(d_3))
		d_4 = F.leaky_relu(d_4, negative_slope=0.02, inplace=True)

		d_5 = self.conv_5(d_4)
		d_5 = d_5.view(-1, self.z_dim)
		d_5 = torch.sigmoid(d_5)

		return d_5