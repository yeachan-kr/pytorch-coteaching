# coding=utf-8
import torch

# for vision
import torchvision
import torchvision.transforms as transforms

# for natural language
from torchtext import data, datasets
from torchtext.vocab import GloVe
from torchtext.data import Iterator

import os
import pickle
import numpy as np


def load_dataset(dataset: str = 'MNIST', datapath: str = './data/'):
    """ Download and load dataset (MNIST, CIFAR10, CIFAR100) """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if dataset.__eq__('CIFAR10'):
        trainset = torchvision.datasets.CIFAR10(root=datapath, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=datapath, train=False, download=True, transform=transform)
        return trainset, testset
    if dataset.__eq__('TREC'):

        # set up fields
        TEXT = data.Field(lower=True, include_lengths=False, batch_first=True, fix_length=300)
        LABEL = data.Field(sequential=False)

        # make splits for data
        train, test = datasets.TREC.splits(TEXT, LABEL)

        # build the vocabulary
        TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
        LABEL.build_vocab(train)

        train_loader = Iterator(train, batch_size=1, repeat=False)

        test_loader = Iterator(test, batch_size=1, repeat=False)

        trainset = []
        for d in train_loader:
            trainset.append((d.text[-1], d.label[-1] - 1))  # for 'unk' token

        testset = []
        for d in test_loader:
            testset.append((d.text[-1], d.label[-1] - 1))  # for 'unk' token
        return trainset, testset, TEXT.vocab


def build_uniform_noise(num_class: int, noise_prob: float, noise_type: str) -> np.ndarray:
    """ """
    mat_size = (num_class, num_class)

    if noise_type.__eq__('sym'):
        noise_matrix = (1 - noise_prob) * np.identity(num_class) + (noise_prob / (num_class - 1)) * (np.ones(mat_size) - np.eye(*mat_size))
        print(noise_matrix)

    return noise_matrix


def corrupt_dataset(noise_matrix: np.ndarray, data):
    corrupt_data = []
    for i, item in enumerate(data):
        img, label = item
        sampled_label = np.random.multinomial(1, noise_matrix[label, :]).argmax()
        corrupt_data.append((img, label, sampled_label))
    return corrupt_data


def split_train_valid(data: list, valid_ratio: float):
    np.random.shuffle(data)
    nvalid = int(len(data) * valid_ratio)
    train = data[nvalid:]
    valid = data[:nvalid]
    return train, valid


def preprocess(FLAGS):
    if FLAGS.dataset.__eq__('TREC'):
        train, test, vocab = load_dataset(dataset=FLAGS.dataset, datapath=FLAGS.datapath)
    else:
        train, test = load_dataset(dataset=FLAGS.dataset, datapath=FLAGS.datapath)
    noise_matrix = build_uniform_noise(num_class=FLAGS.num_class, noise_prob=FLAGS.noise_prob,
                                       noise_type=FLAGS.noise_type)
    train = corrupt_dataset(noise_matrix=noise_matrix, data=train)
    train, valid = split_train_valid(data=train, valid_ratio=FLAGS.valid_ratio)

    pickle.dump([train, valid, test], open(os.path.join(FLAGS.datapath, FLAGS.dataset + '_{}_{}.pkl'.format(FLAGS.noise_prob, FLAGS.noise_type)), 'wb'))
    if 'vocab' in vars():
        pickle.dump(vocab, open(os.path.join(FLAGS.datapath, FLAGS.dataset + '_emb.pkl'), 'wb'))
