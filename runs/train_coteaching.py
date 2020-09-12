import os
import pickle
import numpy as np
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from absl import logging
from models.modeling import CNN, TextCNN
from tensorboardX import SummaryWriter
from utils import EarlyStopping
from matplotlib import pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def co_teaching_loss(model1_loss, model2_loss, rt):
    _, model1_sm_idx = torch.topk(model1_loss, k=int(int(model1_loss.size(0)) * rt), largest=False)
    _, model2_sm_idx = torch.topk(model2_loss, k=int(int(model2_loss.size(0)) * rt), largest=False)

    # co-teaching
    model1_loss_filter = torch.zeros((model1_loss.size(0))).cuda()
    model1_loss_filter[model2_sm_idx] = 1.0
    model1_loss = (model1_loss_filter * model1_loss).sum()

    model2_loss_filter = torch.zeros((model2_loss.size(0))).cuda()
    model2_loss_filter[model1_sm_idx] = 1.0
    model2_loss = (model2_loss_filter * model2_loss).sum()

    return model1_loss, model2_loss


def train_step(data_loader, gpu: bool, model_list: list, optimizer, criterion, rt):
    global_step = 0
    avg_accuracy = 0.
    avg_loss = 0.

    model1, model2 = model_list
    model1 = model1.train()
    model2 = model2.train()
    for x, y, y_hat in data_loader:
        # Forward and Backward propagation
        x, y, y_hat = x.to(device), y.to(device), y_hat.to(device)

        out1 = model1(x)
        out2 = model2(x)

        model1_loss = criterion(out1, y_hat)
        model2_loss = criterion(out2, y_hat)
        model1_loss, model2_loss = co_teaching_loss(model1_loss=model1_loss, model2_loss=model2_loss, rt=rt)

        # loss exchange
        optimizer.zero_grad()
        model1_loss.backward()
        torch.nn.utils.clip_grad_norm_(model1.parameters(), 5.0)
        optimizer.step()

        optimizer.zero_grad()
        model2_loss.backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), 5.0)
        optimizer.step()

        avg_loss += (model1_loss.item() + model2_loss.item())

        # Compute accuracy
        acc = torch.eq(torch.argmax(out1, 1), y).float()
        avg_accuracy += acc.mean()
        global_step += 1

    return avg_accuracy / global_step, avg_loss / global_step, [model1, model2]


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


def update_reduce_step(cur_step, num_gradual, tau=0.5):
    return 1.0 - tau * min(cur_step / num_gradual, 1)


def train(FLAGS):
    # load dataset (train)
    train, valid, test = pickle.load(open(os.path.join(FLAGS.datapath, FLAGS.dataset + '_{}_{}.pkl'.format(FLAGS.noise_prob, FLAGS.noise_type)), 'rb'))

    if FLAGS.dataset.__eq__('TREC'):
        vocab = pickle.load(open(os.path.join(FLAGS.datapath, FLAGS.dataset + '_emb.pkl'), 'rb'))
    train_data_loader = torch.utils.data.DataLoader(train, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4)
    valid_data_loader = torch.utils.data.DataLoader(valid, batch_size=FLAGS.batch_size, shuffle=False, num_workers=4)
    test_data_loader = torch.utils.data.DataLoader(test, batch_size=FLAGS.batch_size, shuffle=False, num_workers=4)
    logging.info('{} dataloader successfully loaded'.format(FLAGS.dataset))

    if FLAGS.dataset.__eq__('TREC'):
        model1 = TextCNN(vocab=vocab.stoi, num_class=FLAGS.num_class, drop_rate=FLAGS.drop_rate,
                         pre_weight=vocab.vectors)
        model2 = TextCNN(vocab=vocab.stoi, num_class=FLAGS.num_class, drop_rate=FLAGS.drop_rate,
                         pre_weight=vocab.vectors)
    else:
        model1 = CNN(num_class=FLAGS.num_class, dropout_rate=FLAGS.drop_rate)
        model2 = CNN(num_class=FLAGS.num_class, dropout_rate=FLAGS.drop_rate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model1 = nn.DataParallel(model1)
        model2 = nn.DataParallel(model2)

    model1.to(device)
    model2.to(device)

    # learning history
    train_acc_list = []
    test_acc_list = []

    early_stopping = EarlyStopping(patience=FLAGS.stop_patience, verbose=False)
    criterion = nn.CrossEntropyLoss(reduce=False)
    optimizer = optim.Adam(chain(model1.parameters(), model2.parameters()), lr=FLAGS.lr)
    for e in range(FLAGS.epochs):
        # update reduce step
        rt = update_reduce_step(cur_step=e, num_gradual=FLAGS.num_gradual, tau=FLAGS.tau)

        # training step
        train_accuracy, avg_loss, model_list = train_step(data_loader=train_data_loader,
                                                          gpu=FLAGS.gpu,
                                                          model_list=[model1, model2],
                                                          optimizer=optimizer,
                                                          criterion=criterion,
                                                          rt=rt)
        model1, model2 = model_list

        # testing/valid step
        test_accuracy = test_step(data_loader=test_data_loader,
                                  gpu=FLAGS.gpu,
                                  model=model1)

        dev_accuracy = valid_step(data_loader=valid_data_loader,
                                  gpu=FLAGS.gpu,
                                  model=model1)

        train_acc_list.append(train_accuracy)
        test_acc_list.append(test_accuracy)

        logging.info(
            '{} epoch, Train Loss {}, Train accuracy {}, Dev accuracy {}, Test accuracy {}, Reduce rate {}'.format(e + 1,
                                                                                                                avg_loss,
                                                                                                                train_accuracy,
                                                                                                                dev_accuracy,
                                                                                                                test_accuracy,
                                                                                                                rt))

        # early_stopping(-dev_accuracy, model1, test_acc=test_accuracy)
        # if early_stopping.early_stop:
        #     logging.info('Training stopped! Best accuracy = {}'.format(max(early_stopping.acc_list)))
        #     break

    # learning curve plot
    xrange = [(i + 1) for i in range(FLAGS.epochs)]
    plt.plot(xrange, train_acc_list, 'b', label='training accuracy')
    plt.plot(xrange, test_acc_list, 'r', label='test accuracy')
    plt.legend()
    plt.title('Learning curve')
    plt.savefig('l_curve.png')

    if not os.path.exists(FLAGS.save_dir):
        os.mkdir(FLAGS.save_dir)
    torch.save(model1.state_dict(),  # save model object before nn.DataParallel
               os.path.join(FLAGS.save_dir,
                            '{}_{}_{}_{}.pt'.format(FLAGS.dataset, FLAGS.model, FLAGS.noise_prob, FLAGS.noise_type)))
