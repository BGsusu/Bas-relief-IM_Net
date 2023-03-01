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
## 网络结构：PointNet Encoder + IM-Net Decoder，预训练PointNet Encoder