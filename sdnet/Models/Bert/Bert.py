# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/9/5 -*-


import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from Models.Bert.modeling import BertModel

"""
    Bert在SDNet中的使用，包括子词编码平均等功能。Bert.py实现了BERT上下文编码的生成过程。在BatchGen中，
    每个单词宝贝Bert自带的WordPiece工具分成若干个子词。BERT预训练模型为每个子词生成多层上下文编码。SDNet
    模型将一个词对应的所有子词的每一层上下文编码求平均，从而得到这个单词在每一层的编码。
    
    此外，BERT的预训练模型只能接收长度为512个子词的输入。但是CoQA中的一部分文章超出了这个限制。所以，模型
    将文章分成若干块，每块含512个子词（最后一块可能少于512个子词），然后分别计算Bert编码，最后拼接得到结果。
"""


class Bert(nn.Module):
    def __init__(self, opt):
        super(Bert, self).__init__()
        print('Loading BERT model...')
        self.BERT_MAX_LEN = 512
        self.linear_combine = 'BERT_LINEAR_COMBINE' in opt

        if 'BERT_LARGE' in opt:
            print('Using BERT Large model')
            model_file = os.path.join(opt['datadir'], opt['BERT_large_model_file'])
            print('Loading BERT model from', model_file)
            self.bert_model = BertModel.from_pretrained(model_file)
            self.bert_dim = 1024
            self.bert_layer = 24
        else:
            print('Using BERT base model')
            model_file = os.path.join(opt['datadir'], opt['BERT_model_file'])
            print('Loading BERT model from', model_file)
            self.bert_model = BertModel.from_pretrained(model_file)
            self.bert_dim = 768
            self.bert_layer = 12

        self.bert_model.cuda()
        self.bert_model.eval()

        print('Finished loading')

    def forward(self, x_bert, x_bert_mask, x_bert_offset, x_mask):
        """
        :param x_bert: batch * max_bert_sent_len (ids)
        :param x_bert_mask:
        :param x_bert_offset:
        :param x_mask:
        :return:
        """
        
















































