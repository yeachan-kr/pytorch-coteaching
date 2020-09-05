""" Noisy-NLP model training/evaluating codes """
import os
import pickle
import numpy as np
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from absl import logging
from models.modeling import CNN
from tensorboardX import SummaryWriter
from utils import EarlyStopping

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_step(data_loader, gpu: bool, model, optimizer, criterion):
    global_step = 0
    avg_accuracy = 0.
    avg_loss = 0.

    model = model.train()
    for x, y, y_hat in data_loader:
        # Forward and Backward propagation
        x, y, y_hat = x.to(device), y.to(device), y_hat.to(device)

        out = model(x)
        model_loss = criterion(out, y_hat)

        # loss exchange
        optimizer.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        avg_loss += model_loss.item()

        # Compute accuracy
        acc = torch.eq(torch.argmax(out, 1), y).float()
        avg_accuracy += acc.mean()
        global_step += 1
    return avg_accuracy / global_step, avg_loss / global_step, model


def test_step(data_loader, gpu: bool, model):
    model = model.eval()
    global_step = 0
    avg_accuracy = 0.

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        acc = torch.eq(torch.argmax(logits, 1), y)
        acc = acc.cpu().numpy()
        acc = np.mean(acc)
        avg_accuracy += acc
        global_step += 1
    return avg_accuracy / global_step


def valid_step(data_loader, gpu: bool, model):
    model = model.eval()
    global_step = 0
    avg_accuracy = 0.

    for x, y, y_hat in data_loader:
        x, y, y_hat = x.to(device), y.to(device), y_hat.to(device)
        
        logits = model(x)
        acc = torch.eq(torch.argmax(logits, 1), y)
        acc = acc.cpu().numpy()
        acc = np.mean(acc)
        avg_accuracy += acc
        global_step += 1
    return avg_accuracy / global_step


def train(FLAGS):

    # load dataset (train)
    train, valid, test = pickle.load(open(os.path.join(FLAGS.datapath, FLAGS.dataset + '.pkl'), 'rb'))
    train_data_loader = torch.utils.data.DataLoader(train, batch_size=FLAGS.batch_size, shuffle=True, num_workers=2)
    valid_data_loader = torch.utils.data.DataLoader(valid, batch_size=FLAGS.batch_size, shuffle=True, num_workers=2)
    test_data_loader = torch.utils.data.DataLoader(test, batch_size=FLAGS.batch_size, shuffle=False, num_workers=2)
    logging.info('{} dataloader successfully loaded'.format(FLAGS.dataset))

    torch.manual_seed(FLAGS.seed)
    model = CNN(num_class=FLAGS.num_class, dropout_rate=FLAGS.drop_rate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    # learning history
    train_acc_list = []
    test_acc_list = []

    early_stopping = EarlyStopping(patience=FLAGS.stop_patience, verbose=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=FLAGS.lr)
    for e in range(FLAGS.epochs):
        # training step
        train_accuracy, avg_loss, model = train_step(data_loader=train_data_loader,
                                                          gpu=FLAGS.gpu,
                                                          model=model,
                                                          optimizer=optimizer,
                                                          criterion=criterion)

        # testing/valid step
        test_accuracy = test_step(data_loader=test_data_loader,
                                  gpu=FLAGS.gpu,
                                  model=model)

        dev_accuracy = valid_step(data_loader=valid_data_loader,
                                  gpu=FLAGS.gpu,
                                  model=model)

        train_acc_list.append(train_accuracy)
        test_acc_list.append(test_accuracy)

        logging.info('{} epoch, Train Loss {}, Train accuracy {}, Dev accuracy {}, Test accuracy {}'.format(e,
                                                                                                            avg_loss,
                                                                                                            train_accuracy,
                                                                                                            dev_accuracy,
                                                                                                            test_accuracy))
        early_stopping(-dev_accuracy, model, test_acc=test_accuracy)
        if early_stopping.early_stop:
            logging.info('Training stopped! Best accuracy = {}'.format(max(early_stopping.acc_list)))
            break

    torch.save(model.state_dict(),  # save model object before nn.DataParallel
               os.path.join(FLAGS.save_dir, 'model_{}.pt'.format(FLAGS.dataset)))
