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
        :param x_bert_mask: batch * max_bert_sent_len (0/1)
        :param x_bert_offset: batch * max_real_word_num * 2
        :param x_mask: batch * max_real_word_num
        :return: embedding: batch * max_real_word_num * bert_dim
        """
        if self.linear_combine:
            return self.combine_forward(x_bert, x_bert_mask, x_bert_offset, x_mask)

        last_layers = []
        bert_sent_len = x_bert.shape[1]  # 实际子词个数
        # 每次处理self.BERT_MAX_LEN=512个子词
        p = 0
        while p < bert_sent_len:
            all_encoder_layers, _ = self.bert_model(
                x_bert[:, p: (p + self.BERT_MAX_LEN)],
                token_type_ids = None,
                attention_mask = x_bert_mask[:, p: (p + self.BERT_MAX_LEN)]
            )  # bert_layer * batch * max_bert_sent_len * bert_dim
            last_layers.append(all_encoder_layers[-1])  # batch * up_to_512 * bert_dim
            p += self.BERT_MAX_LEN
        # 通过拼接得到文本所有单词的BERT编码
        bert_embedding = torch.cat(last_layers, 1)

        batch_size = x_mask.shape[0]
        max_word_num = x_mask.shape[1]
        output = Variable(torch.zeros(batch_size, max_word_num, self.bert_dim))
        # 如果一个单词有T个子词，则将这个单词对应的所有子词每一层的编码除以T。这样进行求和后就可以直接得到单词的BERT编码
        for i in range(batch_size):
            for j in range(max_word_num):
                if x_mask[i, j] == 0:
                    continue
                # 单词的所有子词在BERT分词中的位置范围为[st, ed)
                st = x_bert_offset[i, j, 0]
                ed = x_bert_offset[i, j, 1]
                if st + 1 == ed:
                    output[i, j, :] = bert_embedding[i. st, :]
                else:
                    



        
















































