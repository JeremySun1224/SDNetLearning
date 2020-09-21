# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/8/28 -*-


"""
    SDNet核心的模型代码SDNet.py定义了SDNet的网络结构和计算方法。PyTorch中的网络模型类继承nn.Module，
    注册需要优化的参数。然后，在类中实现构造器函数__init__()和前向计算函数forward()。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from Utils.CoQAUtils import POS, ENT
from Models.Bert.Bert import Bert
from Models.Layers import MaxPooling, CNN, dropout, RNN_from_opt, set_dropout_prob, weighted_avg, set_seq_dropout, Attention, \
    DeepAttention, LinearSelfAttn, GetFinalScores


class SDNet(nn.Module):
    def __init__(self, opt, word_embedding):  # word_embedding为SDNet构建的词表中单词的GloVe编码，用于初始化编码层的权重
        super(SDNet, self).__init__()
        print('SDNet model\n')

        self.opt = opt
        set_dropout_prob(0. if not 'DROPOUT' in opt else float(opt['DROPOUT']))  # 设置Dropout比率
        set_seq_dropout('VARIATIONAL_DROPOUT' in self.opt)

        x_input_size = 0  # 统计文章单词(x)的feature维度总和
        ques_input_size = 0  # 统计问题单词(ques)的feature维度总和

        self.vocab_size = int(opt['vocab_size'])  # 词表大小
        vocab_dim = int(opt['vocab_dim'])  # GloVe编码维度
        self.vocab_embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=vocab_dim, padding_idx=1)
        self.vocab_embed.weight.data = word_embedding  # 用GloVe编码初始化编码层权重

        x_input_size += vocab_dim
        ques_input_size += vocab_dim

        if 'CHAR_CNN' in self.opt:
            print('CHAR_CNN')
            char_vocab_size = int(opt['char_vocab_size'])
            char_dim = int(opt['char_emb_size'])
            char_hidden_size = int(opt['char_hidden_size'])
            self.char_embed = nn.Embedding(num_embeddings=char_vocab_size, embedding_dim=char_dim, padding_idx=1)
            self.char_cnn = CNN(input_size=char_dim, window_size=3, output_size=char_hidden_size)
            self.maxpooling = MaxPooling()
            x_input_size += char_hidden_size
            ques_input_size += char_hidden_size

        if 'TUNE_PARTIAL' in self.opt:
            print('TUNE_PARTIAL')
            self.fixed_embedding = word_embedding[opt['tune_partial']:]
        else:
            self.vocab_embed.weight.data.requires_grad = False

        cdim = 0
        self.use_contextual = False

        if 'BERT' in self.opt:
            print('Using BERT')
            self.Bert = Bert(self.opt)
            if 'LOCK_BERT' in self.opt:
                print('Lock BERT\'s weights')
                for p in self.Bert.parameters():  # 锁定BERT权重不进行更新
                    p.requires_grad = False
            if 'BERT_LARGE' in self.opt:
                print('BERT_LARGE')
                bert_dim = 1024
                bert_layers = 24
            else:
                bert_dim = 768
                bert_layers = 12

            print('BERT dim:', bert_dim, 'BERT_LAYERS:', bert_layers)

            if 'BERT_LINEAR_COMBINE' in self.opt:
                print('BERT_LINEAR_COMBINE')  # 如果对BERT每层输出的编码计算加权和，则需要定义权重alpha和gamma
                self.alphaBERT = nn.Parameter(torch.Tensor(bert_layers), requires_grad=True)
                self.gammaBERT = nn.Parameter(torch.Tensor(1, 1), requires_grad=True)
                torch.nn.init.constant(self.alphaBERT, 1.0)
                torch.nn.init.constant(self.gammaBERT, 1.0)

            cdim = bert_dim
            x_input_size += bert_dim
            ques_input_size += bert_dim
        # 单词注意力层
        self.pre_align = Attention(input_size=vocab_dim, hidden_size=opt['prealign_hidden'], correlation_func=3, do_similarity=True)
        x_input_size += vocab_dim
        # 词性和命名实体标注编码
        pos_dim = opt['pos_dim']
        ent_dim = opt['ent_dim']
        self.pos_embedding = nn.Embedding(num_embeddings=len(POS), embedding_dim=pos_dim)
        self.ent_embedding = nn.Embedding(num_embeddings=len(ENT), embedding_dim=ent_dim)
        # 文章单词的4维feature，包括词频、精确匹配等
        x_feat_len = 4
        if 'ANSWER_SPAN_IN_CONTEXT_FEATURE' in self.opt:
            print('ANSWER_SPAN_IN_CONTEXT_FEATURE')
            x_feat_len += 1

        x_input_size += pos_dim + ent_dim + x_feat_len

        print('Initially, the vector_sizes [doc, query] are', x_input_size, ques_input_size)

        additional_feat = cdim if self.use_contextual else 0

        # 文章RNN层
        self.context_rnn, context_rnn_output_size = RNN_from_opt(
            input_size_=x_input_size, hidden_size_=opt['hidden_size'],
            num_layers=opt['in_rnn_layers'], concat_rnn=opt['concat_rnn'], add_feat=additional_feat)

        # 问题RNN层
        self.ques_rnn, ques_rnn_output_size = RNN_from_opt(
            input_size_=ques_input_size, hidden_size_=opt['hidden_size'],
            num_layers=opt['in_rnn_layers'], concat_rnn=opt['concat_rnn'], add_feat=additional_feat)
        # RNN层输出大小
        print('After Input LSTM, the vector_sizes [doc, query] are [', context_rnn_output_size, ques_rnn_output_size, '] * ', opt['in_rnn_layers'])

        # 全关注互注意力
        self.deep_attn = DeepAttention(
            opt=opt, abstr_list_cnt=opt['in_rnn_layers'],
            deep_att_hidden_size_per_abstr=opt['deep_att_hidden_size_per_abstr'], correlation_func=3, word_hidden_size=vocab_dim + additional_feat)
        self.deep_attn_input_size = self.deep_attn.rnn_input_size
        self.deep_attn_output_size = self.deep_attn.output_size

        # 问题理解层
        self.high_lvl_ques_rnn, high_lvl_ques_rnn_output_size = RNN_from_opt(
            input_size_=ques_rnn_output_size * opt['in_rnn_layers'], hidden_size_=opt['highlvl_hidden_size'],
            num_layers=opt['question_high_lvl_rnn_layers'], concat_rnn=True)
        # 统计当前文章单词历史维度
        self.after_deep_attn_size = self.deep_attn_output_size + self.deep_attn_input_size + additional_feat + vocab_dim
        self.self_attn_input_size = self.after_deep_attn_size
        self_attn_output_size = self.deep_attn_output_size

        # 文章单词自注意力层
        self.highlvl_self_attn = Attention(
            input_size=self.self_attn_input_size, hidden_size=opt['deep_att_hidden_size_per_abstr'], correlation_func=3)
        print('Self deep-attention input is {}-dim'.format(self.self_attn_input_size))

        # 文章单词高级RNN层
        self.high_lvl_context_rnn, high_lvl_context_rnn_output_size = RNN_from_opt(
            input_size_=self.deep_attn_output_size + self_attn_output_size, hidden_size_=opt['highlvl_hidden_size'], num_layers=1, concat_rnn=False)
        # 文章单词最终维度
        context_final_size = high_lvl_context_rnn_output_size

        # 问题自注意力层
        print('Do Question self attention')
        self.ques_self_attn = Attention(
            input_size=high_lvl_ques_rnn_output_size, hidden_size=opt['query_self_attn_hidden_size'], correlation_func=3)
        # 问题单词的最终维度
        ques_final_size = high_lvl_ques_rnn_output_size
        print('Before answer span finding, hidden size are', context_final_size, ques_final_size)

        # 线性注意力层，用于获得问题的向量表示
        self.ques_merger = LinearSelfAttn(input_size=ques_final_size)

        # 分数输出层
        self.get_answer = GetFinalScores(x_size=context_final_size, h_size=ques_final_size)































































