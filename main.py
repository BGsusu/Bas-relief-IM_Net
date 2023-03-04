import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np

from trainer import Trainer
from pretrainer import PreTrainer_bas_relief_AE

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", action="store", dest="epoch", default=0, type=int, help="Epoch to train [0]")
parser.add_argument("--iteration", action="store", dest="iteration", default=0, type=int, help="Iteration to train. Either epoch or iteration need to be zero [0]")
parser.add_argument("--learning_rate", action="store", dest="learning_rate", default=0.0001, type=float, help="Learning rate for adam [0.00005]")
parser.add_argument("--beta1", action="store", dest="beta1", default=0.5, type=float, help="Momentum term of adam [0.5]")
parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="checkpoint", help="Directory name to save the checkpoints [checkpoint]")
parser.add_argument("--batch_size", action="store", dest="batch_size", default=4, help="batch size of models")

parser.add_argument("--train", action="store_true", dest="train", default=False, help="True for training, False for testing [False]")
parser.add_argument("--validate", action="store_true", dest="validate", default=False, help="True for validation")
parser.add_argument("--slice", action="store_true", dest="slice", default=False, help="True for slice mode")
parser.add_argument("--pretrain", action="store_true", dest="pretrain", default=False, help="True for pretrain_bas_relief_AE mode")

parser.add_argument("--start", action="store", dest="start", default=0, type=int, help="In testing, output shapes [start:end]")
parser.add_argument("--end", action="store", dest="end", default=16, type=int, help="In testing, output shapes [start:end]")
parser.add_argument("--bas_relief", action="store_true", dest="bas_relief", default=False, help="True for bas_relief [False]")

# data set
parser.add_argument("--br_data_dir", action="store", dest="br_data_dir", default="/home/daipinxuan/bas_relief/AllData/BasRelief", help="Root directory of bas-relief dataset [data]")
parser.add_argument("--m_data_dir", action="store", dest="m_data_dir", default="/home/daipinxuan/bas_relief/AllData/OriginalData", help="Root directory of model dataset [data]")
parser.add_argument("--br_data_type", action="store", dest="br_data_type", default="sdf.npy", help="The type of data")
parser.add_argument("--model_param_path", action="store", dest="model_param_path", default="/home/sujianping/relief/im-net/IM-NET-pytorch/bas-relief/checkpoint/IM_Bas_Relief_20230303-121403/bas_relief_AE.model250.pth", help="model param path .pth")

FLAGS = parser.parse_args()

if FLAGS.bas_relief:
    if FLAGS.train:
        _trainer = Trainer(FLAGS)
        _trainer.train(FLAGS)
    elif FLAGS.validate:
        _trainer = Trainer(FLAGS)
        _trainer.validation(FLAGS)
    elif FLAGS.slice:
        # _trainer = Trainer(FLAGS)
        # _trainer.slice_sdf(FLAGS)
        _pretrainer = PreTrainer_bas_relief_AE(FLAGS)
        _pretrainer.slice_sdf(FLAGS)
    elif FLAGS.pretrain:
        _pretrainer = PreTrainer_bas_relief_AE(FLAGS)
        _pretrainer.train(FLAGS)
else:
	print("Please specify an operation of Net")
