""" Noisy-NLP model training/evaluating codes """
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from absl import logging
from models.modeling import CNN
from dataset import ClassificationDataset


def eval(FLAGS):

    # load vocabulary
    vocab = {}
    file_name = os.path.join(FLAGS.data_path, FLAGS.dataset + '_vocab.txt')
    with open(file_name, mode='r', encoding='utf-8') as f:
        for line in f:
            w = line.replace('\n', '')
            vocab[w] = len(vocab)
    logging.info('vocabulary successfully loaded [number of words = {}]'.format(len(vocab)))

    file_name = os.path.join(FLAGS.data_path, FLAGS.dataset + '_test.txt')
    data_loader = ClassificationDataset(file_name=file_name,
                                        batch_size=FLAGS.batch_size,
                                        vocab=vocab)
    logging.info('{} dataloader successfully loaded'.format(FLAGS.dataset))

    torch.manual_seed(FLAGS.seed)
    model = TextCNN(emb_dim=FLAGS.emb_dim,
                    num_filters=FLAGS.num_filters,
                    size_filters=FLAGS.size_filters,
                    vocab=vocab,
                    num_class=FLAGS.num_class,
                    drop_rate=FLAGS.drop_rate)
    model.load_state_dict(torch.load(os.path.join(FLAGS.save_dir, 'model_{}.pt'.format(FLAGS.dataset))))
    model = model.eval()
    if FLAGS.gpu:
        model = model.cuda()

    nstep = 0
    avg_accuracy = 0.
    for x, y in data_loader:
        if FLAGS.gpu:
            x, y = x.cuda(), y.cuda()

        out = model(x)
        acc = torch.eq(torch.argmax(out, 1), y)
        acc = acc.cpu().numpy()
        acc = np.mean(acc)
        avg_accuracy += acc
        nstep += 1
    logging.info('Test average accuracy {}'.format(avg_accuracy/nstep))