""" PyTorch dataset class"""

import pickle
import random
import torch
import numpy as np
from torch import LongTensor
# from nltk.corpus import wordnet


class Dataset:
    """ Dataset Class : callable """
    def __init__(self):
        super().__init__()

    def __iter__(self):
        raise NotImplementedError


class ClassificationDataset(Dataset):
    """ Classification Dataset wrapper """
    def __init__(self, file_name: str, batch_size: int, vocab: dict, datatype: str = 'train', gpu: bool = True, max_pad_size: int = 5):
        super().__init__()
        self.batch_size = batch_size
        self.file_name = file_name
        self.vocab = vocab
        self.ivocab = dict(zip(vocab.values(), vocab.keys()))
        self.datatype = datatype
        self.gpu = gpu
        self.max_pad_size = 0
        self.data = pickle.load(open(file_name, 'rb'))

    def __len__(self):
        return len(self.data[0])

    def __iter__(self):
        batch = []
        for d in zip(*self.data):
            if self.gpu:
                d = [t.cuda() for t in d]
            batch.append(d)
            if len(batch) == self.batch_size:
                batch = [torch.stack(d) for d in zip(*batch)]
                yield batch
                batch.clear()

        if batch:
            batch = [torch.stack(d) for d in zip(*batch)]
            yield batch


class ClassificationDataset_old(Dataset):
    """ Classification Dataset wrapper """
    def __init__(self, file_name: str, batch_size: int, vocab: dict, datatype: str = 'train', max_pad_size: int = 5):
        super().__init__()
        self.batch_size = batch_size
        self.file_name = file_name
        self.vocab = vocab
        self.ivocab = dict(zip(vocab.values(), vocab.keys()))
        self.datatype = datatype
        self.max_pad_size = 0

        self.data = []
        with open(self.file_name, mode='r', encoding='utf-8') as f:
            for line in f:
                self.data.append(line)
        self.real_label = {}
        # self.new_map = pickle.load(open('diff.pkl', 'rb'))
        self.replace = False

    def set_replacement(self, val):
        self.replace = val

    def get_real_label(self, idx):
        return self.real_label[int(idx)]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        # with open(self.file_name, mode='r', encoding='utf-8') as f:
        x, y, xlen = [], [], []
        c = []
        ids = []
        max_seq = 0
        for line in self.data:
            line = line.replace('\n', '')
            line = list(map(int, line.split(',')))

            if self.datatype == 'train':
                idx, label, label_sample, text = line[0], line[1], line[2], line[3:]

                is_corrupted = (label_sample != label)
                c.append(is_corrupted)

                ids.append(idx)
                x.append(text)
                y.append(label_sample)
                xlen.append(len(text))
                max_seq = max(max_seq, len(text))
                if idx not in self.real_label:
                    self.real_label[idx] = label

            elif self.datatype == 'test':
                idx, label, text = line[0], line[1], line[2:]

                ids.append(idx)
                x.append(text)
                y.append(label)
                xlen.append(len(text))
                max_seq = max(max_seq, len(text))

            if len(x) == self.batch_size:
                for i, d in enumerate(x):
                    npad = (max_seq - len(d) + self.max_pad_size)
                    d.extend([self.vocab['<pad>']] * npad)
                    x[i] = d

                yield LongTensor(x), LongTensor(y), LongTensor(xlen), LongTensor(ids), LongTensor(c)

                x.clear()
                y.clear()
                xlen.clear()
                c.clear()
                ids.clear()
                max_seq = 0

        if len(x) == 0:
            np.random.shuffle(self.data)
            return
        else:

            for i, d in enumerate(x):
                npad = (max_seq - len(d) + self.max_pad_size)
                d.extend([self.vocab['<pad>']] * npad)
                # d = list(reversed(d))
                # d.extend([self.vocab['<pad>']] * self.max_pad_size)
                # d = list(reversed(d))
                x[i] = d
            np.random.shuffle(self.data)
            return LongTensor(x), LongTensor(y), LongTensor(xlen), LongTensor(ids), LongTensor(c)


class SequenceDataset(Dataset):
    """ Sequence Dataset wrapper """
    def __init__(self, file_name: str, batch_size: int, vocab: dict, datatype: str = 'train', max_pad_size: int = 5):
        super().__init__()
        self.batch_size = batch_size
        self.file_name = file_name
        self.vocab = vocab
        self.ivocab = dict(zip(vocab.values(), vocab.keys()))
        self.datatype = datatype
        self.max_pad_size = 0

        self.data = []
        with open(self.file_name, mode='r', encoding='utf-8') as f:
            for line in f:
                self.data.append(line)
        self.real_label = {}
        # self.new_map = pickle.load(open('diff.pkl', 'rb'))
        self.replace = False

    def get_real_label(self, idx):
        idx = int(idx)
        return self.real_label[int(idx)]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        # with open(self.file_name, mode='r', encoding='utf-8') as f:
        x, y, xlen = [], [], []
        c = []
        ids = []
        max_seq = 0
        for line in self.data:
            line = line.replace('\n', '')
            line = line.split('#')

            if self.datatype == 'train':
                idx, label, label_sample, text = line[0], line[1], line[2], line[3]

                idx = int(idx)
                text = list(map(int, text.split(',')))
                label_sample = list(map(int, label_sample.split(',')))
                label = list(map(int, label.split(',')))

                for l, lc in zip(label, label_sample):
                    c.append(l == lc)

                ids.append(idx)
                x.append(text)
                y.append(label_sample)
                xlen.append(len(text))
                max_seq = max(max_seq, len(text))
                if idx not in self.real_label:
                    self.real_label[idx] = label

            elif self.datatype == 'test':
                idx, label, text = line[0], line[1], line[2]

                idx = int(idx)
                text = list(map(int, text.split(',')))
                label = list(map(int, label.split(',')))

                ids.append(idx)
                x.append(text)
                y.append(label)
                xlen.append(len(text))

                max_seq = max(max_seq, len(text))

            if len(x) == self.batch_size:
                for i, d in enumerate(x):
                    npad = (max_seq - len(d))
                    x[i].extend([self.vocab['<pad>']] * npad)
                    y[i].extend([self.vocab['<unk>']] * npad)
                yield LongTensor(x), LongTensor(y), LongTensor(xlen), LongTensor(ids), LongTensor(c)

                x.clear()
                y.clear()
                xlen.clear()
                c.clear()
                ids.clear()
                max_seq = 0

        if len(x) == 0:
            np.random.shuffle(self.data)
            return

        for i, d in enumerate(x):
            npad = (max_seq - len(d) + self.max_pad_size)
            x[i].extend([self.vocab['<pad>']] * npad)
            y[i].extend([self.vocab['<unk>']] * npad)
        np.random.shuffle(self.data)
        yield LongTensor(x), LongTensor(y), LongTensor(xlen), LongTensor(ids), LongTensor(c)
