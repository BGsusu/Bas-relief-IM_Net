# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 15:42:10 2023

@author: 86186
"""
import numpy as np
import os
import torch
from loadData import loadData
import cv2



class SDF_Slice(object):
    def __init__(self, FLAGS, network, param_dir, dataloader, device, save_path):
        self.network= network
        self.param_dir = param_dir
        self.network.load_state_dict(torch.load(self.param_dir))
        self.network.to(device)
        network.eval()
        _loadData = loadData(FLAGS)
        self.dataloader= dataloader
        self.save_path = save_path
        self.device = device

    def get_slice(self):
        for n_iter, data in enumerate(self.dataloader):
            # print("iteration: ", n_iter)
            # print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
            # print("epoch: ",epoch,"iter: ",n_iter)
            # 采样点总数
            pts = data[0].to(self.device)
            print("pts:")
            print(pts.shape)
            print("\n")
            sdf = data[1].to(self.device)
            # 模型表面的点
            mpts = data[2].to(self.device)
            print("mpts:")
            print(mpts.shape)
            print("\n")
            # 相机参数
            camera = data[3].to(self.device)
            print("camera:")
            print(camera.shape)
            print("\n")

            camera=camera.type(torch.float32)
            z_vector, d = self.network(mpts, None, pts, camera, is_training=False)
            print("z_vector shape:")
            print(z_vector.shape)
            print("\n")
            
            out_x =self.sdf_slice(self.network, mpts,z_vector, camera,dim=0,device=self.device)
            out_y =self.sdf_slice(self.network, mpts,z_vector, camera,dim=1,device=self.device)
            out_z =self.sdf_slice(self.network, mpts,z_vector, camera,dim=2,device=self.device)
            cv2.imwrite(os.path.join('out_x.png'), out_x)
            cv2.imwrite(os.path.join('out_y.png'), out_y)
            cv2.imwrite(os.path.join('out_z.png'), out_z)

    def normalized_grid(self, width, height, device='cuda'):
        """Returns grid[x,y] -> coordinates for a normalized window.
        
        Args:
            width, height (int): grid resolution
        """

        # These are normalized coordinates
        # i.e. equivalent to 2.0 * (fragCoord / iResolution.xy) - 1.0
        window_x = torch.linspace(-1, 1, steps=width, device=device) * (width / height)
        window_x += torch.rand(*window_x.shape, device=device) * (1. / width)
        window_y = torch.linspace(1,- 1, steps=height, device=device)
        window_y += torch.rand(*window_y.shape, device=device) * (1. / height)
        coord = torch.stack(torch.meshgrid(window_x, window_y)).permute(1,2,0)
        return coord

    def normalized_slice(self, width, height, dim=0, depth=0.0, device='cuda'):
        """Returns grid[x,y] -> coordinates for a normalized slice for some dim at some depth."""
        window = self.normalized_grid(width, height, device)
        depth_pts = torch.ones(width, height, 1, device=device) * depth

        if dim==0:
            pts = torch.cat([depth_pts, window[...,0:1], window[...,1:2]], dim=-1)
        elif dim==1:
            pts = torch.cat([window[...,0:1], depth_pts, window[...,1:2]], dim=-1)
        elif dim==2:
            pts = torch.cat([window[...,0:1], window[...,1:2], depth_pts], dim=-1)
        else:
            assert(False, "dim is invalid!")
        pts[...,1] *= -1
        return pts

    def sdf_slice(self, net,mpts,z_vector,camera,dim=0,device='cuda', width=250,height=250,depth=0.05):
            pts = self.normalized_slice(width, height, dim=dim, depth=depth, device=device)
            pts=pts.reshape(-1,3)
            #pts=pts.unsqueeze(0)
            pts = pts.unsqueeze(0)
            #pts=pts.repeat(2,1,1)
            print("query pts.shape:")
            print(pts.shape)#(2,num,3)
            print("\n")

            #此处pts的形状存疑
            #new_pts=torch.stack((pts.reshape(-1,3),pts.reshape(-1,3)), dim=0)
            #d = torch.zeros(width * height, 1, device=pts.device)
            #camera.shape:(2,50000,9)
            print("camera")
            _camera=camera[0][0]
            _camera=_camera.repeat(pts.shape[1],1).unsqueeze(0)
            print(_camera.shape)
            print("\n")
            with torch.no_grad():
                #_, d = net(mpts, None, pts.reshape(-1,3), camera, is_training=True)
                _, d = net(mpts, z_vector, pts, _camera, is_training=False)
            print("d shape:")
            print(d.shape)
            print("\n")
            # print(d[0])
            # print("_________________")
            # print(d[1])
            # print("_________________")
            # print(d[0]-d[1])
            # print("_________________")
            # d=d[0]
            d=d.squeeze(1)
            d = d.reshape(width, height, 1)

            d = d.squeeze().cpu().numpy()
            d = np.clip((d + 1.0) / 2.0, 0.0, 1.0)
            blue = np.clip((d - 0.5)*2.0, 0.0, 1.0)
            yellow = 1.0 - blue
            vis = np.zeros([*d.shape, 3])
            vis[...,2] = blue
            vis += yellow[...,np.newaxis] * np.array([0.0, 0.3, 0.4])
            vis += 0.2
            vis[d - 0.5 < 0] = np.array([1.0, 0.38, 0.0])
            for i in range(50):
                vis[np.abs(d - 0.02*i) < 0.0015] = 0.8
            vis[np.abs(d - 0.5) < 0.004] = 0.0
            print(vis.shape)
            print("all last")
            return vis*255
