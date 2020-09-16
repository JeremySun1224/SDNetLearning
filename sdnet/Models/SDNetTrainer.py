# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/8/28 -*-


"""
    继承BaseTrainer.py的功能，负责SDNet的具体训练与测试过程。
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

    def setup_model(self, vocab_embedding):  # 初始化模型
        self.train_loss = AverageMeter()
        self.network = SDNet(self.opt, vocab_embedding)
        if self.use_cuda:
            self.log('Putting model to GPU')
            self.network.cuda()

        parameters = [p for p in self.network.parameters() if p.requires_grad]
        self.optimizer = optim.Adamax(params=parameters)
        if 'ADAM2' in self.opt:
            print('ADAM2')
            self.optimizer = optim.Adam(params=parameters, lr=0.0001)

        self.updates = 0
        self.epoch_start = 0
        self.loss_func = F.cross_entropy

    def load_model(self, model_path):
        print('Loading model from', model_path)
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['state_dict']
        new_state = set(self.network.state_dict().keys())
        for k in list(state_dict['network'].keys()):
            if k not in new_state:
                del state_dict['network'][k]
        for k, v in list(self.network.state_dict().items()):
            if k not in state_dict['network']:
                state_dict['network'][k] = v
        self.network.load_state_dict(state_dict['network'])
        print('Loading finished', model_path)

    def save(self, filename, epoch, prev_filename):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'updates': self.updates
            },
            'train_loss': {
                'val': self.train_loss.val,
                'avg': self.train_loss.avg,
                'sum': self.train_loss.sum,
                'count': self.train_loss.count
            },
            'config': self.opt,
            'epoch': epoch
        }
        try:
            torch.save(params, filename)
            self.log('model saved to {}'.format(filename))
            if os.path.exists(prev_filename):
                os.remove(prev_filename)
        except BaseException:
            self.log(['WARN: Saving failed... continuing anyway.'])

    def save_for_predict(self, filename, epoch):
        network_state = dict(
            [(k, v) for k, v in self.network.state_dict().items() if
             k[0: 4] != 'CoVe' and k[0: 4] != 'ELMo' and k[0: 9] != 'AllenELMo' and k[0: 4] != 'Bert']
        )
        if 'eval_embed.weight' in network_state:
            del network_state['eval_embed.weight']
        if 'fixed_embedding' in network_state:
            del network_state['fixed_embedding']
        params = {
            'state_dict': {
                'network': network_state
            },
            'config': self.opt
        }
        try:
            torch.save(params, filename)
            self.log('model saved to {}'.format(filename))
        except BaseException:
            self.log(['WARN: Saving failed...continuing anyway.'])

    def update(self, batch):
        self.network.train()  # 进入训练模式
        self.network.drop_emb = True
        # 从batch中获得文章、问题、答案的所有信息，包括单词编号、词性标注、BERT分词编号等
        x, x_mask, x_char, x_char_mask, x_features, x_pos, x_ent, x_bert, x_bert_mask, x_bert_offsets, \
        query, query_mask, query_bert, query_char, query_char_mask, query_bert_mask, query_bert_offsets, \
        ground_truth, context_str, context_words, _, _, _, _ = batch
        # 进行前向计算，获得模型预测答案
        # 1) 在文本每个位置开始和结束的概率score_s, score_e
        # 2) 是Yes/No/No answer的概率为score_yes, score_no, score_no_answer
        # 其中score_s和score_e的维度为batch * context_word_num, score_yes, score_no, score_no_answer的维度均为batch * 1
        score_s, score_e, score_yes, score_no, score_no_answer = self.network(
            x, x_mask, x_char, x_char_mask, x_features, x_pos, x_ent, x_bert, x_bert_mask, x_bert_offsets,
            query, query_mask, query_char, query_char_mask, query_bert, query_bert_mask, query_bert_offsets, len(context_words))
        # 答案最长长度在配置文件中定义
        max_len = self.opt['max_len'] or score_s.size(1)
        batch_size = score_s.shape[0]
        context_len = score_s.size(1)
        expand_score = gen_upper_triangle(score_s, score_e, max_len, self.use_cuda)
        scores = torch.cat((expand_score, score_no, score_yes, score_no_answer), dim=1)
        


    def train(self):
        """
            train()函数进行批次处理，即对于一个batch的数据，计算当前预测结果并求导更新参数。
            每训练1500个batch，利用predict()函数在验证数据上进行一次预测并计算准确率得分。
            当前得分最高的模型参数保存在run_id文件夹中。
        """
        self.isTrain = True  # 标记训练模式
        self.getSaveFolder()
        self.saveConf()
        self.vocab, self.char_vocab, vocab_embedding = self.preproc.load_data()  # 从CoQAPreprocess中获得词表和编码
        self.log('-----------------------------------------------')
        self.log('Initializing model...')
        self.setup_model(vocab_embedding)  # 初始化模型

        if 'RESUME' in self.opt:  # 在继续训练模式时，读取之前存储的模型
            model_path = os.path.join(self.opt['datadir'], self.opt['MODEL_PATH'])
            self.load_model(model_path)

        print('Loading train json')  # 读取处理好的训练数据
        with open(os.path.join(self.opt['FEATURE_FOLDER'], self.data_prefix + 'train-preprocessed.json'), 'r') as f:
            train_data = json.load(f)

        print('Loading dev json')  # 读取处理好的验证数据
        with open(os.path.join(self.opt['FEATURE_FOLDER'], self.data_prefix + 'dev-preprocessed.json'), 'r') as f:
            dev_data = json.load(f)

        best_f1_score = 0.0  # 训练中得到的验证集上的最高的F1得分
        numEpochs = self.opt['EPOCH']  # 配置文件中EPOCH为训练轮数
        for epoch in range(self.epoch_start, numEpochs):
            self.log('Epoch {}'.format(epoch))
            # 训练模式，开启Dropout等功能
            self.network.train()
            startTime = datetime.now()
            # 获得训练数据的batch迭代器
            train_batches = BatchGen(self.opt, train_data['data'], self.use_cuda, self.vocab, self.char_vocab)
            # 获得验证数据的batch迭代器
            dev_batches = BatchGen(self.opt, dev_data['data'], self.use_cuda, self.vocab, self.char_vocab, evaluation=True)
            for i, batch in enumerate(train_batches):
                # 每轮结束时或继续训练模式的第一个batch或每1500个batch，在验证数据上预测并计算得分
                if i == len(train_batches) - 1 or (epoch == 0 and i == 0 and ('RESUME' in self.opt)) or (i > 0 and i % 1500 == 0):
                    print('Saving folder is', self.saveFolder)
                    print('Evaluating on dev set...')
                    predictions = []
                    confidence = []
                    dev_answer = []
                    final_json = []
                    for j, dev_batch in enumerate(dev_batches):
                        # 预测的结果包括答案文本、答案可能性打分以及JSON格式结果
                        phrase, phrase_score, pred_json = self.predict(dev_batch)
                        final_json.extend(pred_json)
                        predictions.extend(phrase)
                        confidence.extend(phrase_score)
                        dev_answer.extend(dev_batch[-3])  # answer_str
                    # 计算精确匹配EM和F1得分
                    result, all_f1s = score(pred=predictions, truth=dev_answer, final_json=final_json)
                    f1 = result['f1']
                    # 如果F1得分高于之前的所有模型，则存储此模型
                    if f1 > best_f1_score:
                        model_file = os.path.join(self.saveFolder, 'best_model.pt')
                        self.save_for_predict(model_file, epoch)
                        best_f1_score = f1
                        pred_json_file = os.path.join(self.saveFolder, 'prediction.json')
                        with open(pred_json_file, 'w') as output_file:
                            json.dump(final_json, output_file)
                        score_per_instance = []
                        for instance, s in zip(final_json, all_f1s):
                            score_per_instance.append({
                                'id': instance['id'],
                                'turn_id': instance['turn_id'],
                                'f1': s
                            })
                        score_per_instance_json_file = os.path.join(self.saveFolder, 'score_per_instance.json')
                        with open(score_per_instance_json_file, 'w') as output_file:
                            json.dump(score_per_instance, output_file)
                    self.log('Epoch {0} - dev F1: {1:.3f} (best F1: {2:.3f})'.format(epoch, f1, best_f1_score))
                    self.log('Results breakdown\n{0}'.format(result))
                # 对本批次进行计算、求导和参数更新
                self.update(batch)
                if i % 100 == 0:
                    self.log('updates[{0: 6}] train loss[{1: .5f}] remaining[{2}]'.format(
                        self.updates, self.train_loss.avg,
                        str((datetime.now() - startTime) / (i + 1) * (len(train_batches) - i - 1)).split('.')[0]))
                print('PROGRESS: {0:.2F}%'.format(100.0 * (epoch + 1) / numEpochs))
                print('Config file is at ' + self.opt['confFile'])
