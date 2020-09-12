from __future__ import absolute_import, division, print_function
from absl import app, flags, logging

from runs.train_coteaching import train
from runs.train_coteaching import train
from runs.preprocess import preprocess
from runs.eval import eval

import sys
import pprint

FLAGS = flags.FLAGS

# dataset
flags.DEFINE_string('dataset', 'TREC', 'the name of dataset, available: CIFAR10 (image), TREC (text)')
flags.DEFINE_string('datapath', 'data/', 'the dataset path to be downloaded')
flags.DEFINE_float('valid_ratio', '0.1', 'validation ratio out of total dataset')

# model parameters
flags.DEFINE_float('drop_rate', 0.5, 'dropout settings')

# training parameters
flags.DEFINE_integer('epochs', 30, 'the number of epochs for training')
flags.DEFINE_float('lr', 0.001, 'learning rate')
flags.DEFINE_integer('batch_size', 256, 'the number of batch for training')
flags.DEFINE_integer('num_class', 6, 'the number of class (category) in training data')
flags.DEFINE_integer('stop_patience', 3, 'the number of patience for early stopping')

# noisy parameters
flags.DEFINE_float('tau', 0.5, 'the estimated noise ratio')
flags.DEFINE_integer('num_gradual', 5, 'the number of gradual step (T_k = 5, 10, 15), default: 15')
flags.DEFINE_float('noise_prob', 0.5, 'noise probability in training data')
flags.DEFINE_string('noise_type', 'sym', 'noise type (sym, asym), default: sym')

# misc
flags.DEFINE_bool('gpu', True, '')
flags.DEFINE_string('run_mode', 'train', 'current mode (train, preprocess, eval)')
flags.DEFINE_string('model', 'coteach', 'training model type (coteach, normal), default: coteach')
flags.DEFINE_string('save_dir', 'pretrained/', 'the path of directory for trained models')



def main(argv):
    del argv  # Unused.
    logging.info('Running under Python {0[0]}.{0[1]}.{0[2]}'.format(sys.version_info))
    pprint.pprint(FLAGS.flag_values_dict(), indent=4)

    if FLAGS.run_mode == 'train':
        if FLAGS.model == 'normal':
            from runs.train_normal import train
        elif FLAGS.model == 'coteach':
            from runs.train_coteaching import train
        train(FLAGS)
    if FLAGS.run_mode == 'preprocess':
        preprocess(FLAGS)
    if FLAGS.run_mode == 'eval':
        eval(FLAGS)


if __name__ == '__main__':
    app.run(main)
