import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import prettytable as pt
from transformers import AutoTokenizer, BertTokenizer
import os
import utils
from joblib import Parallel, delayed

os.environ["TOKENIZERS_PARALLELISM"] = "false"

dis2idx = np.zeros((1000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


class Vocabulary(object):
    PAD = '<pad>'
    UNK = '<unk>'
    SUC = '<suc>'

    def __init__(self):
        self.label2id = {self.PAD: 0, self.SUC: 1}
        self.id2label = {0: self.PAD, 1: self.SUC}

    def add_label(self, label):
        label = label.lower()
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label

        assert label == self.id2label[self.label2id[label]]

    def __len__(self):
        return len(self.token2id)

    def label_to_id(self, label):
        label = label.lower()
        return self.label2id[label]

    def id_to_label(self, i):
        return self.id2label[i]


def collate_fn(data):
    bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text = map(list, zip(*data))

    max_tok = np.max(sent_length)
    sent_length = torch.LongTensor(sent_length)
    max_pie = np.max([x.shape[0] for x in bert_inputs])
    bert_inputs = pad_sequence(bert_inputs, True)
    batch_size = bert_inputs.size(0)

    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data

    dis_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    dist_inputs = fill(dist_inputs, dis_mat)
    labels_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    grid_labels = fill(grid_labels, labels_mat)
    mask2d_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.bool)
    grid_mask2d = fill(grid_mask2d, mask2d_mat)
    sub_mat = torch.zeros((batch_size, max_tok, max_pie), dtype=torch.bool)
    pieces2word = fill(pieces2word, sub_mat)

    return bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text


class RelationDataset(Dataset):
    def __init__(self, bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text):
        self.bert_inputs = bert_inputs
        self.grid_labels = grid_labels
        self.grid_mask2d = grid_mask2d
        self.pieces2word = pieces2word
        self.dist_inputs = dist_inputs
        self.sent_length = sent_length
        self.entity_text = entity_text

    def __getitem__(self, item):
        return torch.LongTensor(self.bert_inputs[item]), \
               torch.LongTensor(self.grid_labels[item]), \
               torch.LongTensor(self.grid_mask2d[item]), \
               torch.LongTensor(self.pieces2word[item]), \
               torch.LongTensor(self.dist_inputs[item]), \
               self.sent_length[item], \
               self.entity_text[item]

    def __len__(self):
        return len(self.bert_inputs)

def process_bert(data, tokenizer: BertTokenizer, vocab, config):
    results = Parallel(n_jobs=config.n_jobs, verbose=True)(
        [delayed(to_example)(instance, tokenizer, vocab) for instance in data])
    results = [list(rel) for rel in zip(*results)]
    return results #bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text


def to_example(instance, tokenizer: BertTokenizer, vocab):
    tokens = [tokenizer.tokenize(word) for word in instance['sentence']]
    pieces = [piece for pieces in tokens for piece in pieces]
    _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
    _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])

    """
    tokens = [tokenizer.tokenize(word) for word in instance['sentence']]
    _bert_inputs = tokenizer(instance['sentence'], is_split_into_words=True, truncation=True, max_length=512)
    _bert_inputs = np.array(_bert_inputs['input_ids'])
    """

    length = len(instance['sentence'])
    _grid_labels = np.zeros((length, length), dtype=np.int)
    _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=np.bool)
    _dist_inputs = np.zeros((length, length), dtype=np.int)
    _grid_mask2d = np.ones((length, length), dtype=np.bool)

    if tokenizer is not None:
        start = 0
        for i, pieces in enumerate(tokens):
            if len(pieces) == 0:
                continue
            pieces = list(range(start, start + len(pieces)))
            _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
            start += len(pieces)

    for k in range(length):
        _dist_inputs[k, :] += k
        _dist_inputs[:, k] -= k

    for i in range(length):  # TODO 这段骚操作没看懂
        for j in range(length):
            if _dist_inputs[i, j] < 0:
                _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
            else:
                _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
    _dist_inputs[_dist_inputs == 0] = 19

    for entity in instance["ner"]:
        index = entity["index"]
        for i in range(len(index)):
            if i + 1 >= len(index):
                break
            _grid_labels[index[i], index[i + 1]] = 1
        _grid_labels[index[-1], index[0]] = vocab.label_to_id(entity["type"])

    _entity_text = set([utils.convert_index_to_text(e["index"], vocab.label_to_id(e["type"]))
                        for e in instance["ner"]])
    return _bert_inputs, _grid_labels, _grid_mask2d, _pieces2word, _dist_inputs, length, _entity_text


def fill_vocab(vocab, dataset):
    entity_num = 0
    for instance in dataset:
        for entity in instance["ner"]:
            vocab.add_label(entity["type"])
        entity_num += len(instance["ner"])
    return entity_num


def load_data_bert(config):
    with open('./data/{}/train.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('./data/{}/dev.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    with open('./data/{}/test.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    train_data = [data for data in train_data if len(data['sentence']) != 0]
    dev_data = [data for data in dev_data if len(data['sentence']) != 0]
    test_data = [data for data in test_data if len(data['sentence']) != 0]

    tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir="./cache/")

    vocab = Vocabulary()
    train_ent_num = fill_vocab(vocab, train_data)
    dev_ent_num = fill_vocab(vocab, dev_data)
    test_ent_num = fill_vocab(vocab, test_data)

    table = pt.PrettyTable([config.dataset, 'sentences', 'entities'])
    table.add_row(['train', len(train_data), train_ent_num])
    table.add_row(['dev', len(dev_data), dev_ent_num])
    table.add_row(['test', len(test_data), test_ent_num])
    config.logger.info("\n{}".format(table))

    config.label_num = len(vocab.label2id)
    config.vocab = vocab
    # save

    save_path_token = os.path.join(config.save_root_path, 'tokenizer.pt')
    torch.save(tokenizer, save_path_token)
    save_path_vocab = os.path.join(config.save_root_path, 'vocab.pt')
    torch.save(vocab, save_path_vocab)
    config.logger.info('tokenizer saved to {}.'.format(save_path_token))
    config.logger.info('entity vocab saved to {}.'.format(save_path_vocab))

    config.logger.info("preprocessing original datasets.")
    train_dataset = RelationDataset(*process_bert(train_data, tokenizer, vocab, config))
    dev_dataset = RelationDataset(*process_bert(dev_data, tokenizer, vocab,config))
    test_dataset = RelationDataset(*process_bert(test_data, tokenizer, vocab, config))
    return (train_dataset, dev_dataset, test_dataset), (train_data, dev_data, test_data)

def process_bert_predict(texts, tokenizer):
    bert_inputs = []
    grid_mask2d = []
    dist_inputs = []
    pieces2word = []
    sent_length = []
    for index, text in enumerate(texts):
        # 这里直接是以字为单位
        tokens = [tokenizer.tokenize(word) for word in text]
        pieces = [piece for pieces in tokens for piece in pieces]
        _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
        # 将字符转换为bert需要的token
        _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])

        length = len(text)
        _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=np.bool)
        _dist_inputs = np.zeros((length, length), dtype=np.int)
        _grid_mask2d = np.ones((length, length), dtype=np.bool)

        if tokenizer is not None:
            start = 0
            # tokens:[['高'], ['勇'], ['：'], ['男'], ['，'], ['中'], ['国'], ['国'], ['籍'], ['，'], ['无'], ['境'], ['外'], ['居'], ['留'], ['权'], ['，']]
            for i, pieces in enumerate(tokens):
                if len(pieces) == 0:
                    continue
                # 这里的start表示的是第i个token的起始位置
                pieces = list(range(start, start + len(pieces)))
                _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                start += len(pieces)

        for k in range(length):
            _dist_inputs[k, :] += k
            _dist_inputs[:, k] -= k

        for i in range(length):
            for j in range(length):
                if _dist_inputs[i, j] < 0:
                    _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                else:
                    _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
        _dist_inputs[_dist_inputs == 0] = 19

        sent_length.append(length)
        bert_inputs.append(_bert_inputs)
        grid_mask2d.append(_grid_mask2d)
        dist_inputs.append(_dist_inputs)
        pieces2word.append(_pieces2word)

    return bert_inputs, grid_mask2d, pieces2word, dist_inputs, sent_length, texts


def collate_fn_predict(data):
    bert_inputs, grid_mask2d, pieces2word, dist_inputs, sent_length, texts = map(list, zip(*data))

    max_tok = np.max(sent_length)
    sent_length = torch.LongTensor(sent_length)
    max_pie = np.max([x.shape[0] for x in bert_inputs])
    bert_inputs = pad_sequence(bert_inputs, True)
    batch_size = bert_inputs.size(0)

    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data

    dis_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    dist_inputs = fill(dist_inputs, dis_mat)
    mask2d_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.bool)
    grid_mask2d = fill(grid_mask2d, mask2d_mat)
    sub_mat = torch.zeros((batch_size, max_tok, max_pie), dtype=torch.bool)
    pieces2word = fill(pieces2word, sub_mat)

    return bert_inputs, grid_mask2d, pieces2word, dist_inputs, sent_length, texts


class RelationDatasetPredict(Dataset):
    def __init__(self, bert_inputs, grid_mask2d, pieces2word, dist_inputs, sent_length, texts):
        self.bert_inputs = bert_inputs
        self.grid_mask2d = grid_mask2d
        self.pieces2word = pieces2word
        self.dist_inputs = dist_inputs
        self.sent_length = sent_length
        self.texts = texts

    def __getitem__(self, item):
        return torch.LongTensor(self.bert_inputs[item]), \
               torch.LongTensor(self.grid_mask2d[item]), \
               torch.LongTensor(self.pieces2word[item]), \
               torch.LongTensor(self.dist_inputs[item]), \
               self.sent_length[item], \
               self.texts[item]

    def __len__(self):
        return len(self.bert_inputs)


def load_data_bert_predict(texts, config):
    if isinstance(texts, str):
        texts = [texts]

    # load model data
    tokenizer = torch.load(config.tokenizer)

    predict_dataset = RelationDatasetPredict(*process_bert_predict(texts, tokenizer))
    return predict_dataset
