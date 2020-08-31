# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/8/28 -*-


"""
    继承BaseTrainer.py的功能，负责SDNet的具体训练与测试过程
"""
import json
import os
import random
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from Utils.CoQAPreprocess import CoQAPreprocess
from Utils.CoQAUtils import BatchGen, AverageMeter, gen_upper_triangle, score
from Models.Layers import set_dropout_prob
from Models.SDNet import SDNet
from Models.BaseTrainer import BaseTrainer

