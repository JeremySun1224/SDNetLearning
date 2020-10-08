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

    def forward(self, x, x_single_mask, x_char, x_char_mask, x_features, x_pos, x_ent, x_bert, x_bert_mask, x_bert_offsets,
                q, q_mask, q_char, q_char_mask, q_bert, q_bert_mask, q_bert_offsets, context_len):
        """
        forward()前向计算函数以BatchGen()产生的批次数据作为输入，经过编码层、交互层和输出层计算得到最终的打分结果
        :param x: [1, x_len] (word_ids)
        :param x_single_mask: [1, x_len]
        :param x_char: [1, x_len, char_len] (char_ids)
        :param x_char_mask: [1, x_len, char_len]
        :param x_features: [batch_size, x_len, feature_len] (5 if answer_span_in_context_feature 4 otherwise)
        :param x_pos: [1, x_len] (POS id)
        :param x_ent: [1, x_len] (ENT id)
        :param x_bert: [1, x_bert_token_len]
        :param x_bert_mask: [1, x_bert_token_len]
        :param x_bert_offsets: [1, x_len, 2]
        :param q: [batch, q_len] (word_ids)
        :param q_mask: [batch, q_len]
        :param q_char: [batch, q_len, char_len] (char ids)
        :param q_char_mask: [batch, q_len, char_len]
        :param q_bert: [1, q_bert_token_len]
        :param q_bert_mask: [1, q_bert_token_len]
        :param q_bert_offsets: [1, q_len, 2]
        :param context_len: number of words in context (only one per batch)
        :return:
            score_s: [batch, context_len]
            score_e: [batch, context_len]
            score_no: [batch, 1]
            score_yes: [batch, 1]
            score_noanswer: [batch, 1]
        """
        batch_size = q.shape[0]
        # 由于同一个batch中的问答共享一篇文章，x_single_mask只有一行，这里将x_single_mask重复batch_size行，与问题数据对齐
        x_mask = x_single_mask.expand(batch_size, -1)
        # 获得文章单词编码，同样重复batch_size行
        x_word_embed = self.vocab_embed(x).expand(batch_size, -1, -1)  # [batch, x_len, vocab_dim]
        # 获得问题单词编码
        ques_word_embed = self.vocab_embed(q)  # [batch, q_len, vocab_dim]
        # 文章单词历史
        x_input_list = [dropout(x=x_word_embed, p=self.opt['dropout_emb'], training=self.drop_emb)]  # [batch, x_len, vocab_dim]
        # 问题单词历史
        ques_input_list = [dropout(x=x_word_embed, p=self.opt['dropout_emb'], training=self.drop_emb)]  # [batch, q_len, vocab_dim]
        # 上下文编码层
        x_cemb = ques_cemb = None
        if 'BERT' in self.opt:
            x_cemb = ques_cemb = None

            if 'BERT_LINEAR_COMBINE' in self.opt:
                # 得到BERT每一层输出的文章单词编码
                x_bert_output = self.Bert(x_bert, x_bert_mask, x_bert_offsets, x_single_mask)
                # 计算加权和
                x_cemb_mid = self.linear_sum(x_bert_output, self.alphaBERT, self.gammaBERT)
                # 得到BERT每一层输出的问题单词编码
                ques_bert_output = self.Bert(q_bert, q_bert_mask, q_bert_offsets, q_mask)
                # 计算加权和
                ques_cemb_mid = self.linear_sum(ques_bert_output, self.alphaBERT, self.gammaBERT)
                x_cemb_mid = x_cemb_mid.expand(batch_size, -1, -1)
            else:
                # 不计算加权和的情况
                x_cemb_mid = self.Bert(x_bert, x_bert_mask, x_bert_offsets, x_single_mask)
                x_cemb_mid = x_cemb_mid.expand(batch_size, -1, -1)
                ques_cemb_mid = self.Bert(q_bert, q_bert_mask, q_bert_offsets, q_mask)

            # 上下文编码加入单词历史
            x_input_list.append(x_cemb_mid)
            ques_input_list.append(ques_cemb_mid)

        if 'CHAR_CNN' in self.opt:
            x_char_final = self.character_cnn(x_char, x_char_mask)
            x_char_final = x_char_final.expand(batch_size, -1, -1)
            ques_char_final = self.character_cnn(q_char, q_char_mask)
            x_input_list.append(x_char_final)
            ques_input_list.append(ques_char_final)

        # 单词注意力层
        x_prealign = self.pre_align(x_word_embed, ques_word_embed, q_mask)
        x_input_list.append(x_prealign)  # [batch, x_len, vocab_dim + cdim + vocab_dim]
        # 词性编码
        x_pos_emb = self.pos_embedding(x_pos).expand(batch_size, -1, -1)  # [batch, x_len, pos_dim]
        # 命名实体编码
        x_ent_emb = self.ent_embedding(x_ent).expand(batch_size, -1, -1)  # [batch, x_len, ent_dim]
        x_input_list.append(x_pos_emb)
        x_input_list.append(x_ent_emb)
        # 加入文章单词的词频和精确匹配特征
        x_input_list.append(x_features)  # [batch_size, x_len, vocab_dim + cdim + vocab_dim + pos_dim, ent_dim, feature_dim]
        # 将文章答案的单词历史向量拼接起来
        x_input = torch.cat(x_input_list, 2)  # [batch_size, x_len, vocab_dim + cdim + vocab_dim + pos_dim + ent_dim + feature_dim]
        # 将问题答案的单词历史向量拼接起来
        ques_input = torch.cat(ques_input_list, 2)  # [batch_size, q_len, vocab_dim + cdim]
        # Multi-layer RNN, 获得文章和问题RNN层的输出
        _, x_rnn_layers = self.context_rnn(x_input, x_mask, return_list=True, x_additional=x_cemb)  # [layer, batch, x_len, context_rnn_output_size]
        _, ques_rnn_layers = self.ques_rnn(ques_input, q_mask, return_list=True, x_additional=ques_cemb)  # [layer, batch, q_len, ques_rnn_output_size]
        # 问题理解层
        ques_highlvl = self.high_lvl_ques_rnn(torch.cat(ques_rnn_layers, 2), q_mask)  # [batch, q_len, high_lvl_ques_rnn_output_size]
        ques_rnn_layers.append(ques_highlvl)  # (layer + 1) layers

        # deep multilevel inter-attention, 全关注互注意力层的输入
        if x_cemb is None:
            x_long = x_word_embed
            ques_long = ques_word_embed
        else:
            x_long = torch.cat([x_word_embed, x_cemb], 2)  # [batch, x_len, vocab_dim + cdim]
            ques_long = torch.cat([ques_word_embed, ques_cemb], 2)  # [batch, q_len, vocab_dim + cdim]
        # 文章单词经过全关注互注意力层, x_rnn_after_inter_attn: [batch, x_len, deep_attn_output_size], x_inter_attn: [batch, x_len, deep_attn_input_size]
        x_rnn_after_inter_attn, x_inter_attn = self.deep_attn([x_long], x_rnn_layers, [ques_long], ques_rnn_layers, x_mask, q_mask, return_bef_rnn=True)

        # deep self attention, 全关注自注意力层的输入, x_self_attn_input: [batch, x_len, deep_attn_output_size + deep_attn_input_size + cdim + vocab_dim]
        if x_cemb is None:
            x_self_attn_input = torch.cat([x_rnn_after_inter_attn, x_inter_attn, x_word_embed], 2)
        else:
            x_self_attn_input = torch.cat([x_rnn_after_inter_attn, x_inter_attn, x_cemb, x_word_embed], 2)
        # 文章经过全关注自注意力层
        x_self_attn_output = self.highlvl_self_attn(x_self_attn_input, x_self_attn_input, x_mask, x3=x_rnn_after_inter_attn,
                                                    drop_diagonal=True)  # [batch, x_len, deep_attn_output_size]

        # 文章单词经过高级RNN层
        x_highlvl_output = self.high_lvl_context_rnn(torch.cat([x_rnn_after_inter_attn, x_self_attn_output], 2), x_mask)

        # 文章单词的最终编码x_final
        x_final = x_highlvl_output  # [batch, x_len, high_lvl_context_rnn_output_size]

        # 问题单词的自注意力层
        ques_final = self.ques_self_attn(ques_highlvl, ques_highlvl, q_mask, x3=None, drop_diagonal=True)  # [batch, q_len, high_lvl_ques_rnn_output_size]

        # merge questions, 获得问题的向量表示
        q_merge_weights = self.ques_merger(ques_final, q_mask)
        ques_merged = weighted_avg(ques_final, q_merge_weights)  # [batch, ques_final_size], 按照q_merge_weights计算ques_final的加权和

        # 获得答案在文章每个位置开始和结束的概率以及三种特殊答案“是/否/没有答案”的概率
        score_s, score_e, score_no, score_yes, score_noanswer = self.get_answer(x_final, ques_merged, x_mask)

        return score_s, score_e, score_no, score_yes, score_noanswer

    def character_cnn(self, x_char, x_char_mask):
        """
        :param x_char: [batch, word_num, char_num]
        :param x_char_mask: [batch, word_num, char_num]
        :return: [batch, word_num, char_cnn_hidden_size]
        """
        x_char_embed = self.char_embed(x_char)  # [batch, word_num, char_num, char_dim]
        batch_size = x_char_embed.shape[0]
        word_num = x_char_embed.shape[1]
        char_num = x_char_embed.shape[2]
        char_dim = x_char_embed.shape[3]
        # x_char_cnn: [batch * word_num, char_num, char_cnn_hidden_size]
        x_char_cnn = self.char_cnn(x_char_embed.contiguous().view(-1, char_num, char_dim), x_char_mask)
        # x_char_cnn_final: [batch, word_num, char_cnn_hidden_size]
        x_char_cnn_final = self.maxpooling(x_char_cnn, x_char_mask.contiguous().view(-1, char_num)).contiguous().view(batch_size, word_num, -1)
        return x_char_cnn_final

    # 对BERT每层的输出计算加权和
    def linear_sum(self, output, alpha, gamma):
        alpha_softmax = F.softmax(alpha)  # 对alpha权重归一化
        for i in range(len(output)):
            t = output[i] * alpha_softmax[i] * gamma  # 第i层的权重系数是alpha_softmax[i] * gamma
            if i == 0:
                res = t
            else:
                res += t
        res = dropout(x=res, p=self.opt['dropout_emb'], training=self.drop_emb)  # Dropout后输出
        return res
