import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from .PointNet import PointNetEncoder
from .IM_Net import im_generator


class im_bas_relief_network(nn.Module):
	def __init__(self, ef_dim, gf_dim, z_dim, point_dim, camera_dim):
		super(im_bas_relief_network, self).__init__()
		self.ef_dim = ef_dim
		self.gf_dim = gf_dim
		self.z_dim = z_dim
		self.point_dim = point_dim
		self.camera_dim = camera_dim
		# self.encoder = encoder(self.ef_dim, self.z_dim)
		self.encoder = PointNetEncoder()
		self.generator = im_generator(self.z_dim, self.point_dim, self.camera_dim, self.gf_dim)

	def forward(self, inputs, z_vector, point_coord, camera_param, is_training=False):
		if is_training:
			z_vector,_,_ = self.encoder(inputs, is_training=is_training)
			# print("z_vector: ",z_vector.shape)
			net_out = self.generator(point_coord, camera_param, z_vector, is_training=is_training)
		else:
			if inputs is not None:
				z_vector = self.encoder(inputs, is_training=is_training)
			if z_vector is not None and point_coord is not None:
				net_out = self.generator(point_coord, camera_param, z_vector, is_training=is_training)
			else:
				net_out = None

		return z_vector, net_out