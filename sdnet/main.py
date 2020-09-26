# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/9/22 -*-

import argparse  # argparse可以让我们直接在命令行中就可以向程序中传入参数并让程序运行
import os
import sys
import torch
from Models.SDNetTrainer import SDNetTrainer
from Utils.Arguments import Arguments

opt = None

# 创建解析
parser = argparse.ArgumentParser(description='SDNet')

# 添加参数
parser.add_argument('command', help='Command: train')  # 添加程序参数信息，这些信息在parse_args()调用时被存储和使用
parser.add_argument('conf_file', help='Path to conf file.')

# 解析参数
cmd_line_args = parser.parse_args()
command = cmd_line_args.command
conf_file = cmd_line_args.conf_file
conf_args = Arguments(confFile=conf_file)
opt = conf_args.readArguments()
opt['cuda'] = torch.cuda.is_available()
opt['confFile'] = conf_file
opt['datadir'] = os.path.dirname(conf_file)  # conf_file指定数据文件夹的位置

for key, val in cmd_line_args.__dict__.items():
    if val is not None and key not in ['command', 'conf_file']:
        opt[key] = val

model = SDNetTrainer(opt=opt)

print('Select command: ' + command)
model.train()
