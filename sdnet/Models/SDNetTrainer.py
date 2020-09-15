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


class SDNetTrainer(BaseTrainer):
    def __init__(self, opt):
        super(SDNetTrainer, self).__init__(opt)
        print('SDNet Model Trainer')
        set_dropout_prob(0.0 if not 'DROPOUT' in opt else float(opt['DROPOUT']))
        self.seed = int(opt['SEED'])
        self.data_prefix = 'coqa-'
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.preproc = CoQAPreprocess(self.opt)
        if self.use_cuda:
            torch.cuda.manual_seed_all(self.seed)

    def offical(self, model_path, test_data):
        print('-----------------------------------------------')
        print('Initializing model...')
        self.setup_model(self.preproc.train_embedding)
        self.load_model(model_path)

        print('Predicting in batches...')
        test_batches = BatchGen(
            self.opt,
            test_data['data'],
            self.use_cuda,
            self.preproc.train_vocab,
            self.preproc.train_char_vocab,
            evaluation=True
        )
        predictions = []
        confidence = []
        final_json = []
        cnt = 0
        for j, test_batch in enumerate(test_batches):
            cnt += 1
            if cnt % 50 == 0:
                print(cnt, '/', len(test_batches))
            phrase, phrase_score, pred_json = self.predict(test_batch)
            predictions.extend(phrase)  # 在已知列表中追加新的内容
            confidence.extend(phrase_score)
            final_json.extend(pred_json)
        return predictions, confidence, final_json

    def train(self):
        self.isTrain = True
        self.getSaveFolder()
        self.saveConf()
        self.vocab, self.char_vocab, vocab_embedding = self.preproc.load_data()
        self.log('-----------------------------------------------')
        self.log('Initializing model...')
        self.setup_model(vocab_embedding)

        if 'RESUME' in self.opt:
            model_path = os.path.join(self.opt['datadir'], self.opt['MODEL_PATH'])
            self.load_model(model_path)

        print('Loading train json')
        with open(os.path.join(self.opt['FEATURE_FOLDER'], self.data_prefix + 'train-preprocessed.json'), 'r') as f:
            train_data = json.load(f)

        print('Loading dev json')
        with open(os.path.join(self.opt['FEATURE_FOLDER'], self.data_prefix + 'dev-preprocessed.json'), 'r') as f:
            dev_data = json.load(f)

        best_f1_score = 0.0
        numEpochs = self.opt['EPOCH']
        for epoch in range(self.epoch_start, numEpochs):
            self.log('Epoch {}'.format(epoch))
            self.network.train()
            startTime = datetime.now()
            train_batches = BatchGen(self.opt, train_data['data'], self.use_cuda, self.vocab, self.char_vocab)
            dev_batches = BatchGen(self.opt, dev_data['data'], self.use_cuda, self.vocab, self.char_vocab, evaluation=True)
            for i, batch in enumerate(train_batches):
                if i == len(train_batches) - 1 or (epoch == 0 and i == 0 and ('RESUME' in self.opt)) or (i > 0 and i % 1500 == 0):
                    print('Saving folder is', self.saveFolder)
                    print('Evaluating on dev set...')
                    predictions = []
                    confidence = []
                    dev_answer = []
                    final_json = []
                    for j, dev_batch in enumerate(dev_batches):
                        phrase, phrase_score, pred_json = self.predict(dev_batch)
                        final_json.extend(pred_json)
                        predictions.extend(phrase)
                        confidence.extend(phrase_score)
                        dev_answer.extend(dev_batch[-3])  # answer_str
                    result, all_f1s = score(pred=predictions, truth=dev_answer, final_json=final_json)













































