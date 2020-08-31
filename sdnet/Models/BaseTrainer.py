# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/8/28 -*-

import os

"""
    定义了基本的训练器操作，包括建立新的文件夹存储模型、配置文件备份和日志文件、写日志文件等功能
"""


class BaseTrainer(object):
    def __init__(self, opt):
        self.opt = opt
        self.isTrain = False
        if self.opt['cuda'] == True:
            self.use_cuda = True
            print('Using CUDA\n')
        else:
            self.use_cuda = False
            print('Using CPU\n')

        self.is_official = 'OFFICIAL' in self.opt
        self.use_spacy = 'SPACY_FEATURE' in self.opt
        self.opt['logFile'] = 'log.txt'

        opt['FEATURE_FOLDER'] = 'conf~/' + ('spacy_intermediate_feature!/' if self.use_spacy else 'intermediate_feature~/')
        opt['FEATURE_FOLDER'] = os.path.join(opt['datadir'], opt['FEATURE_FOLDER'])

    def log(self, s):
        if self.is_official:
            return
        with open(os.path.join(self.saveFolder, self.opt['logFile']), 'a') as f:
            f.write(s + '\n')
        print(s)

    def getSaveFolder(self):
        runid = 1
        while True:
            saveFolder = os.path.join(self.opt['datadir'], 'conf~', 'run_' + str(runid))
            if not os.path.exists(saveFolder):
                self.saveFolder = saveFolder
                os.makedirs(self.saveFolder)
                print('Saving logs, model and evaluation in ' + self.saveFolder)
                return
            runid = runid + 1

    def saveConf(self):
        with open(self.opt['confFile'], encoding='utf-8') as f:
            with open(os.path.join(self.saveFolder, 'conf_copy'), 'w', encoding='utf-8') as f:
                for line in f:
                    fw.write(line + '\n')

    def train(self):
        pass

    def load(self):
        pass
