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

# argparse可以让我们直接在命令行中就可以向程序中传入参数并让程序运行
parser = argparse.ArgumentParser(description='SDNet')
parser.add_argument('command', help='Command: train')
parser.add_argument('conf_file', help='Path to conf file.')

