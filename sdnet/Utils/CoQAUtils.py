# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 20/8/19 -*-

"""
    SDNet用到的功能函数
"""

import torch
import os
import random
import numpy as np
from collections import Counter
from Models.Bert.tokenization import BertTokenizer
from Utils.GeneralUtils import normalize_text, nlp
from torch.autograd import Variable

POS = {w: i for i, w in enumerate([''] + list(nlp.tagger.labels))}  # 所有词性及其索引
ENT = {w: i for i, w in enumerate([''] + nlp.entity.move_names)}  # 所有实体及其索引


# 生成词表中单词的编码
def build_embedding(embed_file, targ_vocab, wv_dim):
    vocab_size = len(targ_vocab)
    emb = np.random.uniform(low=-1, high=1, size=(vocab_size, wv_dim))  # 随机编码的所有维度为-1~1之间的等概率分布
    emb[0] = 0  # 0号单词<PAD>的单词编码为全零
    w2id = {w: i for i, w in enumerate(targ_vocab)}
    lineCnt = 0
    with open(file=embed_file, encoding='utf-8') as f:  # 读入GloVe编码文件
        for line in f:
            lineCnt = lineCnt + 1
            if lineCnt % 100000 == 0:
                print('.', end='', flush=True)
            elems = line.split()
            token = normalize_text(''.join(elems[0: -wv_dim]))  # 文件每一列最后300列是编码，之前是单词字符串
            if token in w2id:  # 如果是词表中的单词，则将其编码特换为GloVe编码
                emb[w2id[token]] = [float(v) for v in elems[-wv_dim:]]
    return emb


# 根据单词获得词表中的编号
def token2id_sent(sent, w2id, unk_id=None, to_lower=False):
    if to_lower:
        sent = sent.lower()
    w2id_len = len(w2id)
    # 如果单词w在词表w2id中出现则返回单词编号w2id[w]否则的话返回unk_id
    ids = [w2id[w] if w in w2id else unk_id for w in sent]
    return ids


def char2id_sent(sent, c2id, unk_id=None, to_lower=False):
    if to_lower:
        sent = sent.lower()
    cids = [[c2id['<STA>'] + [c2id[c] if c in c2id else unk_id for c in w] + [c2id['<END>']]] for w in sent]
    return cids


def token2id(w, vocab, unk_id=None):
    return vocab[w] if w in vocab else unk_id


"""
    文章单词的feature包括词频，原词，小写变形和词形还原是否在问题中出现；
    精确匹配向量(如果这个单词在问题中出现，值为1，否则为0)。
"""


def feature_gen(context, question):
    counter_ = Counter(w.text.lower() for w in context)
    total = sum(counter_.values())
    term_freq = [counter_[w.text.lower()] / total for w in context]  # 获得文章中每个单词出现的概率
    question_word = {w.text for w in question}  # 集合
    question_lemma = {w.lemma_ if w != '-PRON-' else w.text.lower() for w in question}  # To lemma_, the base form of the word.
    match_origin = [w.text in question_word for w in context]
    match_lower = [w.text.lower() in question_word for w in context]
    match_lemma = [(w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in question_lemma for w in context]
    C_features = list(zip(term_freq, match_origin, match_lower, match_lemma))
    return C_features


"""
    得到从开始得分到结束得分的上三角矩阵
"""


def gen_upper_triangle(score_s, score_e, max_len, use_cuda):
    batch_size = score_s.shape[0]
    context_len = score_s.shape[1]
    expand_score = score_s.unsqueeze(2).expand([batch_size, context_len, context_len]) + \
                   score_e.unsqueeze(2).expand([batch_size, context_len, context_len])
    score_mask = torch.ones(context_len)
    if use_cuda:
        score_mask = score_mask.cuda()
    score_mask = torch.ger(score_mask, score_mask).triu().tril(max_len - 1)  # 上三角
    empty_mask = score_mask.eq(0).unsqueeze(0).expand_as(expand_score)  # eq(0), 判断是否与0相等，等为1，否则为0
    expand_score.data.masked_fill_(empty_mask.data, -float('inf'))
    return expand_score.contiguous().view(batch_size, -1)  # [batch_size, context_len * context_len]


"""
    生成Batch。每轮开始时，BatchGen将所有文章打乱顺序，然后将一个batch定义为一篇文章和与它相关的所有轮问题和答案。
"""


class BatchGen(object):
    def __init__(self, opt, data, use_cuda, vocab, char_vocab, evaluation=False):
        self.opt = opt
        self.data = data
        self.use_cuda = use_cuda
        self.vocab = vocab
        self.char_vocab = char_vocab
        self.evaluation = evaluation
        if 'PREV_ANS' in self.opt:
            self.prev_ans = self.opt['PREV_ANS']
        else:
            self.prev_ans = 2
        if 'PREV_QUES' in self.opt:
            self.prev_ques = self.opt['PREV_QUES']
        else:
            self.prev_ques = 0
        self.use_char_cnn = 'CHAR_CNN' in self.opt
        self.bert_tokenizer = None
        if 'BERT' in self.opt:
            if 'BERT_LARGE' in opt:
                print('Using BERT Large model')
                tokenizer_file = os.path.join(opt['datadir'], opt['BERT_large_tokenizer_file'])
                print('Loading tokenizer from', tokenizer_file)
                self.bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name=tokenizer_file)
            else:
                print('Using BERT base model')
                tokenizer_file = os.path.join(opt['datadir'], opt['BERT_tokenizer_file'])
                print('Loading tokenizer from', tokenizer_file)
                self.bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name=tokenizer_file)
        self.answer_span_in_context = 'ANSWER_SPAN_IN_CONTEXT_FEATURE' in self.opt
        self.ques_max_len = (30 + 1) * self.prev_ans + (25 + 1) * (self.prev_ques + 1)  # 限定拼接后新问题的最大长度
        self.char_max_len = 30
        print('*****************')
        print('prev_ques   :', self.prev_ques)
        print('prev_ans    :', self.prev_ans)
        print('ques_max_len:', self.ques_max_len)
        print('*****************')
        c2id = {c: i for i, c in enumerate(char_vocab)}
        if not evaluation:
            indices = list(range(len(self.data)))
            random.shuffle(indices)  # 训练采用随机梯度下降，因此训练模式中需要随机打乱数据顺序
            self.data = [self.data[i] for i in indices]

    def __len__(self):
        return len(self.data)

    # 将一段文本的spaCy分词继续分成BERT字词
    def bertify(self, words):
        if self.bert_tokenizer is None:
            return None
        bpe = ['[CLS]']  # bpe存储BERT子词分词结果，BERT规定第一个子词是特殊开始字符[CLS]
        x_bert_offsets = []  # 存储每个单词的第一个和最后一个子词在bpe中的位置
        for word in words:
            now = self.bert_tokenizer.tokenize(word)
            x_bert_offsets.append([len(bpe), len(bpe) + len(now)])
            bpe.extend(now)
        bpe.append('[SEP]')  # BERT规定最后一个子词为特殊分隔符[SEP]
        x_bert = self.bert_tokenizer.convert_tokens_to_ids(bpe)  # 将子词转化为BERT规定的编码
        return x_bert, x_bert_offsets

    def __iter__(self):
        """
        BatchGen提供一个迭代器，使得它可以在循环语句中使用。Python中用__iter__作为迭代器的函数名，
        函数用yield命令向外层的循环语句提供循环变量内容。
        :return: SDNet中，BatchGen的迭代器提供网络前向计算时使用的一个批次中的文章、问题和答案的单词编号、掩码、特征等信息。
        """
        data = self.data
        MAX_ANS_SPAN = 15  # 答案可以包含的最多单词个数
        for datum in data:
            if not self.evaluation:  # 训练模式下，忽略包含超长答案的数据
                datum['qas'] = [qa for qa in datum['qas'] if
                                len(qa['annotated_answer']['word']) == 1 or qa['answer_span'][1] - qa['answer_span'][0] < MAX_ANS_SPAN]
            if len(datum['qas']) == 0:
                continue
            context_len = len(datum['annotated_context']['wordid'])
            x_len = context_len  # 一个batch的数据全部来自同一篇文章，context_len为该文章的单词个数
            qa_len = len(datum['qas'])  # qa_len为问答轮数，即batch_size
            batch_size = qa_len
            x = torch.LongTensor(1, x_len).fill_(0)  # x为文章单词的编号
            x_char = torch.LongTensor(1, x_len, self.char_max_len).fill_(0)
            if 'BERT' in self.opt:  # 获得bertiy函数的结果
                x_bert, x_bert_offsets = self.bertify(datum['annotated_context']['word'])
                x_bert_mask = torch.LongTensor(1, len(x_bert)).fill_(1)
                x_bert = torch.tensor([x_bert], dtype=torch.long)
                x_bert_offsets = torch.tensor([x_bert_offsets], dtype=torch.long)
            x_pos = torch.LongTensor(1, x_len).fill_(0)  # x_pos为文章单词词性标注的编号
            x_ent = torch.LongTensor(1, x_len).fill_(0)  # x_ent为文章命名实体标注的编号
            if self.answer_span_in_context:
                x_features = torch.Tensor(batch_size, x_len, 5).fill_(0)
            else:
                x_features = torch.Tensor(batch_size, x_len, 4).fill_(0)
            query = torch.LongTensor(batch_size, self.ques_max_len).fill_(0)  # 问题单词的编号
            query_char = torch.LongTensor(batch_size, self.ques_max_len, self.char_max_len).fill_(0)
            query_bert_offsets = torch.LongTensor(batch_size, self.ques_max_len, 2).fill_(0)
            q_bert_list = []
            ground_truth = torch.LongTensor(batch_size, 2).fill_(-1)  # 正确答案在文章中的开始位置和结束位置
            context_id = datum['id']  # 文章的ID
            context_str = datum['context']
            context_words = datum['anbotated_context']['word']
            context_word_offsets = datum['raw_context_offsets']
            answer_strs = []
            turn_ids = []
            x[0, :context_len] = torch.LongTensor(datum['annotated_context']['wordid'])
            if self.use_char_cnn:
                for j in range(context_len):
                    t = min(len(datum['annotated_context']['charid'][j]), self.char_max_len)
                    x_char[0, j, :t] = torch.LongTensor(datum['annotated_context']['charid'][j][:t])
            x_pos[0, :context_len] = torch.LongTensor(datum['annotated_context']['pos_id'])
            x_ent[0, :context_len] = torch.LongTensor(datum['annotated_context']['ent_id'])
            for i in range(qa_len):
                x_features[i, :context_len, :4] = torch.Tensor(datum['qas'][i]['context_features'])
                turn_ids.append(int(datum['qas'][i]['turn_id']))
                p = 0
                ques_words = []
                for j in range(i - self.prev_ans, i + 1):
                    if j < 0:
                        continue
                    if not self.evaluation and datum['qas'][j]['answer_span'][0] == -1:
                        continue
                    q = [2] + datum['qas'][j]['annotated_question']['wordid']
                    q_char = [[0]] + datum['qas'][j]['annotated_question']['charid']
                    if j >= i - self.prev_ques and p + len(q) <= self.ques_max_len:
                        ques_words.extend(['<Q>'] + datum['qas'][j]['annotated_question']['word'])
                        query[i, p: (p + len(q))] = torch.LongTensor(q)
                        if self.use_char_cnn:
                            for k in range(len(q_char)):
                                t = min(self.char_max_len, len(q_char[k]))
                                query_char[i, p + k, :t] = torch.LongTensor(q_char[k][:t])
                        ques = datum['qas'][j]['question'].lower()
                        p += len(q)
                    a = [3] + datum['qas'][j]['annotated_answer']['wordid']
                    a_char = [[0]] + datum['qas'][j]['annotated_answer']['charid']
                    if j < i and j >= i - self.prev_ans and p + len(a) <= self.ques_max_len:
                        ques_words.extend(['<A>'] + datum['qas'][j]['annotated_answer']['word'])
                        query[i, p: (p + len(a))] = torch.LongTensor(a)
                        if self.use_char_cnn:
                            for k in range(len(a_char)):
                                t = min(self.char_max_len, len(a_char[k]))
                                query_char[i, p + k, :t] = torch.LongTensor(a_char[k][:t])
                        p += len(a)
                        if self.answer_span_in_context:
                            st = datum['qas'][j]['answer_span'][0]
                            ed = datum['qas'][j]['answer_span'][1] + 1
                            x_features[i, st: ed, 4] = 1.0
                if 'BERT' in self.opt:
                    now_bert, now_bert_offsets = self.bertify(ques_words)
                    query_bert_offsets[i, :len(now_bert_offsets), :] = torch.tensor(now_bert_offsets, dtype=torch.long)
                    q_bert_list.append(now_bert)
                # 标准答案在文中开始的位置和结束的位置
                ground_truth[i, 0] = datum['qas'][i]['answer_span'][0]
                ground_truth[i, 1] = datum['qas'][i]['answer_span'][1]
                answer = datum['qas'][i]['raw_answer']
                # 下面是特殊类型的答案，包括“是”，“否”和“没有答案”
                if answer.lower() in ['yes', 'yes.']:
                    ground_truth[i, 0] = -1
                    ground_truth[i, 1] = 0
                    answer_str = 'yes'
                if answer.lower() in ['no', 'no.']:
                    ground_truth[i, 0] = 0
                    ground_truth[i, 1] = -1
                    answer_str = 'no'
                if answer.lower() in ['unknown', 'unknown.']:
                    ground_truth[i, 0] = -1
                    ground_truth[i, 1] = -1
                    answer_str = 'unknown'
                if ground_truth[i, 0] >= 0 and ground_truth[i, 1] >= 0:
                    answer_str = answer
                all_viable_answers = [answer_str]
                if 'additional_answers' in datum['qas'][i]:
                    all_viable_answers.extend(datum['qas'][i]['additional_answers'])
                answer_strs.append(all_viable_answers)
            if 'BERT' in self.opt:
                bert_len = max([len(s) for s in q_bert_list])
                query_bert = torch.LongTensor(batch_size, bert_len).fill_(0)
                query_bert_mask = torch.LongTensor(batch_size, bert_len).fill_(0)
                for i in range(len(q_bert_list)):
                    query_bert[i, :len(q_bert_list[i])] = torch.LongTensor(q_bert_list[i])
                    query_bert_mask[i, :len(q_bert_list[i])] = 1
                if self.use_cuda:
                    x_bert = Variable(x_bert.cuda(async=True))
                    x_bert_mask = Variable(x_bert_mask.cuda(async=True))
                    query_bert = Variable(query_bert.cuda(async=True))
                    query_bert_mask = Variable(query_bert_mask.cuda(async=True))
                else:
                    x_bert = Variable(x_bert)
                    x_bert_mask = Variable(x_bert_mask)
                    query_bert = Variable(query_bert)
                    query_bert_mask = Variable(query_bert_mask)
            else:
                x_bert = None
                x_bert_mask = None
                x_bert_offsets = None
                query_bert = None
                query_bert_mask = None
                query_bert_offsets = None
            if self.use_char_cnn:
                x_char_mask = 1 - torch.eq(x_char, 0)
                query_char_mask = 1 - torch.eq(query_char, 0)
                if self.use_cuda:
                    x_char = Variable(x_char.cuda(async=True))
                    x_char_mask = Variable(x_char_mask.cuda(async=True))
                    query_char = Variable(query_char.cuda(async=True))
                    query_char_mask = Variable(query_char_mask.cuda(async=True))
                else:
                    x_char = Variable(x_char)
                    x_char_mask = Variable(x_char_mask)
                    query_char = Variable(query_char)
                    query_char_mask = Variable(query_char_mask)
            else:
                x_char = None
                x_char_mask = None
                query_char = None
                query_char_mask = None
            # 由于补齐符号<PAD>的编号为0，因此可以根据词编号是否为0生成掩码
            x_mask = 1 - torch.eq(x, 0)
            query_mask = 1 - torch.eq(query, 0)
            # 将相关张量放入GPU
            if self.use_cuda:
                x = Variable(x.cuda(async=True))
                x_mask = Variable(x_mask.cuda(async=True))
                x_features = Variable(x_features.cuda(async=True))
                x_pos = Variable(x_pos.cuda(async=True))
                x_ent = Variable(x_ent.cuda(async=True))
                query = Variable(query.cuda(async=True))
                query_mask = Variable(query_mask.cuda(async=True))
                ground_truth = Variable(ground_truth.cuda(async=True))
            else:
                x = Variable(x)
                x_mask = Variable(x_mask)
                x_features = Variable(x_features)
                x_pos = Variable(x_pos)
                x_ent = Variable(x_ent)
                query = Variable(query)
                query_mask = Variable(query_mask)
                ground_truth = Variable(ground_truth)
            yield (x, x_mask, x_char, x_char_mask, x_features, x_pos, x_ent, x_bert, x_bert_offsets,
                   query, query_mask, query_char, query_char_mask, query_bert, query_bert_mask, query_bert_offsets,
                   ground_truth, context_str, context_words, context_word_offsets, answer_strs, context_id, turn_ids)
