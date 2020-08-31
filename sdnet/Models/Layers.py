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
        self.concat_layers = num_layers
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
