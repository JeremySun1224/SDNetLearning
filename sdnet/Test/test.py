# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/9/16 -*-

import torch
use_cuda = ['GPU:0' if torch.cuda.is_available() else 'CPU']

def gen_upper_triangle(score_s, score_e, max_len, use_cuda):
    batch_size = score_s.shape[0]
    context_len = score_s.shape[1]  # context_word_num
    expand_score = score_s.unsqueeze(2).expand([batch_size, context_len, context_len]) + \
                   score_e.unsqueeze(1).expand([batch_size, context_len, context_len])
    score_mask = torch.ones(context_len)
    if use_cuda:
        score_mask = score_mask.cuda()
    score_mask = torch.ger(score_mask, score_mask).triu().tril(max_len - 1)
    empty_mask = score_mask.eq(0).unsqueeze(0).expand_as(expand_score)
    expand_score.data.masked_fill_(empty_mask.data, -float('inf'))
    return expand_score.contiguous().view(batch_size, -1)  # batch x context_len x context_len



