import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import logging as log


# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class MyDataset(Dataset):
	# 初始化函数，得到数据
    def __init__(self, data_pts, data_sdf, data_mpts, data_camera):
        self.pts = data_pts
        self.sdf = data_sdf
        self.mpts = data_mpts
        self.camera =  data_camera
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        pts = self.pts[index]
        sdf = self.sdf[index]
        mpts = self.mpts[index]
        camera= self.camera[index]
        return pts, sdf, mpts, camera
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.pts)

# 加载数据的类
class loadData():
    def __init__(self,config):
        self.br_dataset_path = config.br_data_dir
        self.m_dataset_path = config.m_data_dir
        self.br_data_type = config.br_data_type
        self.batch_size = config.batch_size
        self.train = config.train
    
    # 获取所有类型文件列表
    def get_filelist(self, dir, Filelist):
        newDir = dir
        if os.path.isfile(dir):
            if self.br_data_type in dir:
                Filelist.append(dir)
            # # 若只是要返回文件夹，使用这个
            # Filelist.append(os.path.basename(dir))
        elif os.path.isdir(dir):
            for s in os.listdir(dir):
                # 如果需要忽略某些文件夹，使用以下代码
                if s == "fbx":
                    continue
                newDir=os.path.join(dir,s)
                self.get_filelist(newDir, Filelist)
        return Filelist
    
    # 读入数据集
    def readAllDataFiles(self, mode):
        # 获取所有npy数据文件
        print("Loading dataset... ")
        #init data
        pts=[]
        sdf=[]
        mpts=[]
        camera=[]
        
        #获取所有bas-relief npy文件
        br_filelist = self.get_filelist(self.br_dataset_path,[])
        print("Loading bas-relief models from path: ", self.br_dataset_path," and total files: ", len(br_filelist))
        
        # 为测试集留下最后200个数据
        if mode is 'train':
            print("data for training")
            br_filelist = br_filelist[0:len(br_filelist)-200]
        if mode is 'test':
            print("data for testing")
            br_filelist = br_filelist[len(br_filelist)-200:len(br_filelist)]
        if mode is 'slice':
            print("data for slicing")
            br_filelist = br_filelist[len(br_filelist)-200+1:len(br_filelist)-200+2]
        
        # for test
        # br_filelist = br_filelist[0:len(br_filelist)-1930]
        
        #loop所有浮雕模型数据
        for idx,file in enumerate(br_filelist):
            # print("relief model file path: ",file)
            data = np.load(file)
            pts.append(np.array(data[:,0:4]))
            sdf.append(np.array(data[:,-1]))
            
            # 根据浮雕文件路径名寻找对应的原始模型
            
            # 三种浮雕模型路径与对应的原始模型路径
            # /home/daipinxuan/bas_relief/AllData/BasRelief/Animation-Relief/1/BaseRelief0_0_sdf.npy
            # /home/daipinxuan/bas_relief/AllData/OriginalData/Animation/sdf_gen/1.npy
            
            # /home/daipinxuan/bas_relief/AllData/BasRelief/Character-Relief/1/BaseRelief0_0_sdf.npy
            # /home/daipinxuan/bas_relief/AllData/OriginalData/Character/sdf_gen/1.npy
            
            # /home/daipinxuan/bas_relief/AllData/BasRelief/relief/1-1/BaseRelief0_0_sdf.npy
            # /home/daipinxuan/bas_relief/AllData/OriginalData/source/obj/sdf_gen/1-1.npy
            
            path_list = file.split("/")
            folder_name = path_list[-2]
            file_name_npy = path_list[-1]

            dirStr, ext = os.path.splitext(file)
            file_name = dirStr.split("/")[-1]
            file_name = file_name.replace("_sdf", "")+".obj"

            model_file = ""
            if "/relief/" in file:
                model_file = file.replace("/BasRelief/relief/"+folder_name+"/"+file_name_npy, "/OriginalData/source/obj/sdf_gen/"+folder_name+".npy")
            elif "/Animation-Relief/" in file:
                model_file = file.replace("/BasRelief/Animation-Relief/"+folder_name+"/"+file_name_npy, "/OriginalData/Animation/sdf_gen/"+folder_name+".npy")
            elif "/Character-Relief/" in file:
                model_file = file.replace("/BasRelief/Character-Relief/"+folder_name+"/"+file_name_npy, "/OriginalData/Character/sdf_gen/"+folder_name+".npy")
            else:
                print("file path wrong!")
            # print("paired original model path: ", model_file)
            data = np.load(model_file)
            mpts.append(np.array(data[:,0:3]))
            
            # 根据浮雕文件名获取相机参数
            camera_file = os.path.dirname(file)+"/out.txt"
            #打开文件
            f=open(camera_file,encoding='utf-8')
            #创建空列表
            text=[]
            #读取全部内容 ，并以列表方式返回
            lines = f.readlines()      
            for line in lines:
                #如果读到空行，就跳过
                if line.isspace():
                    continue
                else:
                    #去除文本中的换行等等，可以追加其他操作
                    line = line.replace("\n","")
                    line = line.replace("\t","")
                    #处理完成后的行，追加到列表中
                    text.append(line)
            # 寻找文件需要的参数
            fc = ""
            pos = "" 
            up = ""
            # print("file name: ", file_name)
            for idx,t in enumerate(text):
                if "Focus center" in t:
                    fc = text[idx+1]
                if file_name in t:
                    pos = text[idx+1]
                    pos = pos.replace("camWorldPos: ","")
                    up = text[idx+2]
                    up = up.replace("camWorldUp: ","")
                
            # print("camera foucus string: ", fc)
            # print("camera world pos: ", pos)
            # print("camera world up axis: ", up)
            fc_list = fc.split(" ")
            # print("fc: ",fc_list)
            if fc_list[0] is '':
                fc_list = [0,0,0]
            else:
                fc_list = [float(x) for x in fc_list]
            pos_list = pos.split(" ")
            pos_list = [float(x) for x in pos_list]
            up_list = up.split(" ")
            up_list = [float(x) for x in up_list]
            
            fc_np = np.array(fc_list)
            pos_np = np.array(pos_list)
            up_np = np.array(up_list)
            # 生成相机空间矩阵，并处理pts到相机空间
            # 1、首先我们求得N = eye – lookat，并把N归一化
            N = pos_np - fc_np
            N_norm = N / np.linalg.norm(N)
            # 2、up和N叉积得到U, U= up X N，归一化U
            U = np.cross(up_np, N_norm)
            U_norm = U / np.linalg.norm(U)
            # 3、然后N和U叉积得到V
            V = np.cross(N_norm, U_norm)
            V_norm = V / np.linalg.norm(V)
            # 4、求出视角坐标系的矩阵表示
            Z = np.append(N_norm,0)
            X = np.append(U_norm,0)
            Y = np.append(V_norm,0)
            P = np.append(pos_np,1)
            np_viewcoord = np.array([X,Y,Z,P])
            M_viewcoord = np.matrix(np_viewcoord)
            # 5、求逆矩阵
            M_view = M_viewcoord.I
            
            # pts = np.array(pts)
            for idx in range(len(pts[0])):
                pts[0][idx][3] = 1
                pts[0][idx] = np.dot(pts[0][idx], M_view)
            # 空间变换完成，但是没有验证，应该是对的
            
            camera_list = pos_list+fc_list+up_list
            camera_list = np.array(camera_list)
            camera_list = np.expand_dims(camera_list,0).repeat(50000,axis=0)
            camera.append(np.array(camera_list))
            
            f.close()
            

        # numpy to tensor
        pts = np.array(pts)
        pts = pts[:,:,0:3]
        sdf = np.array(sdf)
        mpts = np.array(mpts)
        camera = np.array(camera)
        
        # print("All sampled bas-relief points: ", pts.shape)
        # print("All sampled bas-relief SDF: ",sdf.shape)
        # # print("All sampled original model points: ",self.mpts.shape)
        # print("Camera Params: ",camera.shape)
        
        # camera = camera.astype(float)
        
        pts = torch.from_numpy(pts)
        sdf = torch.from_numpy(sdf)
        mpts = torch.from_numpy(mpts)
        camera = torch.from_numpy(camera)
        
        print("Loading Dataset has done!")
        return pts, sdf, mpts, camera
    
    # 读入数据的func
    def load(self, config):
        self.train_dataloader = None
        self.test_dataloader = None
        self.slice_dataloader = None
        # 训练数据集
        if config.train:
            pts,sdf,mpts,camera = self.readAllDataFiles('train')
            self.train_dataset=MyDataset(pts,sdf,mpts,camera)
            self.train_dataloader= DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        if config.train or config.validate:
            # 测试数据集
            pts,sdf,mpts,camera = self.readAllDataFiles('test')
            self.test_dataset=MyDataset(pts,sdf,mpts,camera)
            self.test_dataloader= DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        if config.slice:
            # 切片数据
            pts,sdf,mpts,camera = self.readAllDataFiles('slice')
            self.slice_dataset=MyDataset(pts,sdf,mpts,camera)
            self.slice_dataloader= DataLoader(self.slice_dataset, batch_size=1, shuffle=True, pin_memory=True)
        
        print("Data prepared!")
        return self.train_dataloader, self.test_dataloader, self.slice_dataloader