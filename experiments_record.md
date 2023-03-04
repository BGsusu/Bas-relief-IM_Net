# SDF Bas-Relief 实验记录与checkpoint对应
---
## 网络结构：PointNet Encoder + IM-Net Decoder，无预训练
    path： relief/im-net/IM-NET-pytorch/bas-relief/checkpoint/IM_Bas_Relief_20230226-205934
    tag：相机9个参数与坐标直接拼接
    实验效果：很差，厚板子
    
    path：relief/im-net/IM-NET-pytorch/bas-relief/checkpoint/IM_Bas_Relief_20230227-161150
    tag： 将相机参数转换为视角矩阵，把坐标变换到相机空间
    实验效果：依旧很差，还是厚板子

    path： relief/im-net/IM-NET-pytorch/bas-relief/checkpoint/IM_Bas_Relief_20230228-214444
    tag: 去掉两层decoder线性层
    实验效果：还是很差，板子有所变薄

    path： relief/im-net/IM-NET-pytorch/bas-relief/checkpoint/IM_Bas_Relief_20230228-210334
    tag：极端实验，将decoder减少至一层
    实验效果：还是很差
    
    path： relief/im-net/IM-NET-pytorch/bas-relief/checkpoint/IM_Bas_Relief_20230301-123754
    tag：修改decoder输出，将输出从0~1改为-1~1
    实验效果：厚度变得正常了，但是还是退化成了板子
---
## 网络结构：PointNet Encoder + IM-Net Decoder,进行预训练
    path: relief/im-net/IM-NET-pytorch/bas-relief/checkpoint/IM_Bas_Relief_20230301-223539
    tag: 添加预训练代码，首先进行浮雕模型在本框架下的自编码训练
    实验效果：不行
    
    path: relief/im-net/IM-NET-pytorch/bas-relief/checkpoint/IM_Bas_Relief_20230302-004347
    tag: 将预训练模型数量减少至500加快训练
    实验效果：不行

    path: relief/im-net/IM-NET-pytorch/bas-relief/checkpoint/IM_Bas_Relief_20230302-121403
    tag: 尝试单个模型过拟合
    实验效果：不行

    path: relief/im-net/IM-NET-pytorch/bas-relief/checkpoint/IM_Bas_Relief_20230303-204056
    tag: 尝试修改为无符号距离场
    实验效果：不行