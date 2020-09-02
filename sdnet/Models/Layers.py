# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/8/28 -*-


import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.parameter import Parameter

"""
    SDNet的计算使用了Attention、DeepAttention、GetFinalScores等类。这些类封装了较为复杂的子网络结构，
    实现了注意力、全关注注意力、答案输出层等功能。
"""


def set_dropout_prob(p):
    global dropout_p
    dropout_p = p


def set_seq_dropout(option):  # option = True or False
    global do_seq_dropout
    do_seq_dropout = option


def seq_dropout(x, p=0, training=False):
    # x: (batch * len * input_size) or (any other shape)
    if do_seq_dropout and len(x.size()) == 3:  # if x is (batch * len * input_size)
        return seq_dropout(x, p=p, training=training)
    else:
        return F.dropout(x, p=p, training=training)


def dropout(x, p=0, training=False):
    # x: (batch * len * input_size) or (any other shape)
    if do_seq_dropout and len(x.size()) == 3:  # if x is (batch * len * input_size)
        return seq_dropout(x, p=p, training=training)
    else:
        return F.dropout(x, p=p, training=training)


class CNN(nn.Module):
    def __init__(self, input_size, window_size, output_size):
        super(CNN, self).__init__()
        if window_size % 2 != 1:
            raise Exception('window size must be an odd number')
        padding_size = int((window_size - 1) / 2)
        self._output_size = output_size
        self.cnn = nn.Conv2d(1, output_size, kernel_size=(window_size, input_size), padding=(padding_size, 0), bias=False)
        init.xavier_uniform(self.cnn.weight)

    @property  # 该装饰器的作用是可以直接把output_size()这个方法作为属性去直接调用，至少可以不用输入()
    def output_size(self):
        return self._output_size

    """
         x_unsqueeze: num_items x 1 x max_subitem_size x input_size  
         x_conv: num_items x output_size x max_subitem_size
         x_output: num_items x max_subitem_size x output_size
    """

    def forward(self, x, x_mask):
        """
        (item, subitem) can be (word, characters) or (sentence, words)
        :param x: num_items * max_subitem * input_size
        :param x_mask: num_items * max_subitem_size (not used but put here to align with RNN format)
        :return: num_items * max_subitem_size * output_size
        """
        x = F.dropout(x, p=dropout_p, training=self.training)
        x_unsqueeze = x.unsqueeze(1)
        x_conv = F.tanh(self.cnn(x_unsqueeze)).squeeze(3)
        x_output = torch.transpose(x_conv, 1, 2)
        return x_output


class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()
        self.MIN = -1e6

    """
        x_output: num_items * input_size * 1 --> num_items * input_size
    """

    def forward(self, x, x_mask):
        """
        (item, subitem) can be (word, characters) or (sentence, words)
        :param x: num_items * max_subitem_size * input_size
        :param x_mask: num_items * max_subitem_size
        :return: num_items * input_size
        """
        empty_mask = x_mask.eq(0).unsqueeze(2).expand_as(x)
        x_now = x.clone()
        x_now.data.masked_fill_(empty_mask.data, self.MIN)
        x_output = x_now.max(1)[0]
        x_output.data.masked_fill_(x_output.data.eq(self.MIN), 0)
        return x_output


class AveragePooling(nn.Module):
    def __init__(self):
        super(AveragePooling, self).__init__()

    """
        x_output: num_items * input_size * 1 --> num_items * input_size
    """

    def forward(self, x, x_mask):
        """
        (item, subitem) can be (word, characters) or (sentence, words)
        :param x: num_items * max_subitem_size * input_size
        :param x_mask: num_items * max_subitem_size
        :return: num_items * input_size
        """
        x_now = x.clone()
        empty_mask = x_mask.eq(0).unsqueeze(2).expand_as(x_now)
        x_now.data.masked_fill_(empty_mask.data, 0)
        x_sum = torch.sum(x_now, 1)
        x_num = torch.sum(x_mask.eq(1).float(), 1).unsqueeze(1).expand_as(x_sum)
        x_num = torch.clamp(x_num, min=1)
        return x_sum / x_num


class StackedBiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, rnn_type=nn.LSTM, concat_layers=False, bidirectional=True, add_feat=0):
        super(StackedBiRNN, self).__init__()
        self.bidir_coef = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.hidden_size = hidden_size
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            in_size = input_size if i == 0 else (self.bidir_coef * hidden_size + add_feat if i == 1 else self.bidir_coef * hidden_size)
            rnn = rnn_type(input_size=in_size, hidden_size=hidden_size, num_layers=1, bidirectional=bidirectional, batch_first=True)
            self.rnns.append(rnn)

    @property
    def output_size(self):
        if self.concat_layers:
            return self.num_layers * self.bidir_coef * self.hidden_size
        else:
            return self.bidir_coef * self.hidden_size

    def forward(self, x, x_mask, return_list=False, x_additional=None):
        """
        Multi-layer Bi-RNN
        :param x: a Float Tensor of size (batch * wordnum * input_dim).
        :param x_mask: a Byte Tensor of mask for the input tensor (batch * wordnum).
        :param x_additional: a Byte Tensor of mask for the additional input tensor (batch * wordnum * additional_dim).
        :return: a Float Tensor of size (batch * wordnum * output_size).
        """
        hiddens = [x]
        for i in range(self.num_layers):
            rnn_input = hiddens[-1]
            if i == 1 and x_additional is not None:
                rnn_input = torch.cat((rnn_input, x_additional), dim=2)
            if dropout_p > 0:
                rnn_input = dropout(rnn_input, p=dropout_p, training=self.training)
            rnn_output = self.rnns[i](rnn_input)[0]
            hiddens.append(rnn_output)

        if self.concat_layers:
            output = torch.cat(hiddens[1:], 2)
        else:
            output = hiddens[-1]

        if return_list:
            return output, hiddens[1:]
        else:
            return output


# 计算注意力分数
class AttentionScore(nn.Module):
    """
    相关函数score(x1, x2)计算方法：
        correlation_func = 1, sij = x1^Tx2
        correlation_func = 2, sij = (Wx1)D(Wx2)
        correlation_func = 3, sij = Relu(Wx1)DRelu(Wx2)
        correlation_func = 4, sij = x1^TWx2
        correlation_func = 5, sij = Relu(Wx1)DRelu(Wx2)
    """

    def __init__(self, input_size, hidden_size, correalition_func=1, do_similarity=False):
        super(AttentionScore, self).__init__()
        self.correlation_func = correalition_func
        self.hidden_size = hidden_size  # 隐状态维度，即U矩阵的行数

        # 实现公式：score(x1, x2) = ReLU(Ux1)^TDReLU(Ux2)。以下是矩阵U的初始设定
        if correalition_func == 2 or correalition_func == 3:
            self.linear = nn.Linear(input_size, hidden_size, bias=True)  # self.linear即矩阵U
            if do_similarity:  # do_similarity控制初始化参数是否除以维度的平方根（类似Transformer中的Attention），以及是否更新D的参数
                # 应用Parameter()将self.diagonal，即对角矩阵D，绑定到模型中，所以在训练的时候其是可优化的
                self.diagonal = Parameter(torch.ones(1, 1, 1) / (hidden_size ** 0.5), requires_grad=False)
            else:
                self.diagonal = Parameter(torch.ones(1, 1, hidden_size), requires_grad=True)

        if correalition_func == 4:
            self.linear = nn.Linear(input_size, input_size, bias=False)  # 不含矩阵U，即不含隐状态

        if correalition_func == 5:
            self.linear = nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, x1, x2):
        """
        计算x1和x2向量组的注意力分数
        :param x1: batch * word_num1 * dim
        :param x2: batch * word_num2 * dim
        :return: scores: batch * word_num1 * word_num2
        """
        x1 = dropout(x1, p=dropout_p, training=self.training)
        x2 = dropout(x2, p=dropout_p, training=self.training)

        x1_rep = x1
        x2_rep = x2
        batch = x1_rep.size(0)
        word_num1 = x1_rep.size(1)
        word_num2 = x2_rep.size(1)
        dim = x1_rep.size(2)
        # 计算x1_rep和x2_rep
        if self.correlation_func == 2 or self.correlation_func == 3:
            x1_rep = self.linear(x1_rep.contiguous().view(-1, dim)).view(batch, word_num1, self.hidden_size)
            x2_rep = self.linear(x2_rep.contiguous().view(-1, dim)).view(batch, word_num2, self.hidden_size)
            if self.correlation_func == 3:  # ReLU(Wx1)DReLU(Wx2)
                x1_rep = F.relu(x1_rep)
                x2_rep = F.relu(x2_rep)
            # x1_rep is (W1D) or ReLU(Wx1)D and the shape is batch * word_num1 * dim(corr=1) or hidden_size(corr=2, 3)
            x1_rep = x1_rep * self.diagonal.expand_as(x1_rep)

        if self.correlation_func == 4:
            x2_rep = self.linear(x2_rep.contiguous().view(-1, dim)).view(batch, word_num2, self.hidden_size)

        if self.correlation_func == 5:
            x1_rep = self.linear(x1_rep.contiguous().view(-1, dim)).view(batch, word_num1, self.hidden_size)
            x2_rep = self.linear(x2_rep.contiguous().view(-1, dim)).view(batch, word_num2, self.hidden_size)
            x1_rep = F.relu(x1_rep)
            x2_rep = F.relu(x2_rep)

        scores = x1_rep.bmm(x2_rep.transpose(1, 2))
        return scores


# 第2个类Attention类的输入为x1、x2和x3。它利用AttentionScore获得x1和x2的注意力分数，经softmax得到权重后，计算x3的加权和得到注意力向量
class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, correlation_func=1, do_similarity=False):
        super(Attention, self).__init__()
        self.scoring = AttentionScore(input_size, hidden_size, correlation_func, do_similarity)

    def forward(self, x1, x2, x2_mask, x3=None, drop_diagonal=False):
        """
        对于x1中的每个单词，使用在x1和x2之间计算出的分数，获得x3的注意力线性组合（即注意力向量）。如果x3不存在，则使用x2。
        :param x1: batch * word_num1 * dim
        :param x2: batch * word_num2 * dim
        :param x2_mask: batch * word_num2
        :param x3: if not None, batch * word_num2 * dim_3
        :param drop_diagonal: 为True时，将对角线位置的注意力分数置为负无穷，即第i个元素和自身不计算注意力系数
        :return: batch * word_num1 * dim_3
        """
        batch = x1.size(0)
        word_num1 = x1.size(1)
        word_num2 = x2.size(1)

        if x3 is None:
            x3 = x2

        scores = self.scoring(x1, x2)  # 得到注意力分数scores

        # 按照x2的掩码将所有补齐符号位置的注意力分数置为负无穷
        empty_mask = x2_mask.eq(0).unsqueeze(1).expand_as(scores)
        scores.data.masked_fill_(empty_mask.data, -float('inf'))

        if drop_diagonal:
            assert (scores.size(1) == scores.size(2))
            diag_mask = torch.diag(scores.data.new(scores.size(1)).zero_() + 1).byte().unsqueeze(0).expand_as(scores)  # 将Tensor投射成byte类型
            scores.data.masked_fill_(diag_mask, -float('inf'))

        # 用softmax计算注意力分数，所有负无穷的位置获得系数0
        alpha_flat = F.softmax(scores.view(-1, x2.size(1)), dim=1)
        alpha = alpha_flat.view(-1, x1.size(1), x2.size(1))
        # 将注意力系数与x3相乘得到注意力向量attented
        attented = alpha.bmm(x3)
        return attented











































































