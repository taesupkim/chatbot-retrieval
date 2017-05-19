from __future__ import print_function

import sys
import argparse
from keras_base.model import *
from keras import optimizers

def get_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--voca_size', action="store", default=91620, dest="voca_size", type=int)
    parser.add_argument('--emb_size', action="store", default=100, dest="emb_size", type=int)
    parser.add_argument('--hid_size', action="store", default=150, dest="hid_size", type=int)
    parser.add_argument('--ctx_max_len', action="store", default=160, dest="ctx_max_len", type=int)
    parser.add_argument('--ans_max_len', action="store", default=20, dest="ans_max_len", type=int)

    parser.add_argument('--vocab_path', action="store", default=None, dest="vocab_path")

    parser.add_argument('--learn_rate', action="store", default=0.001, dest="learn_rate", type=float)
    parser.add_argument('--optimizer', action="store", default="adam", dest="optimizer")
    parser.add_argument('--batch_size', action="store", default=32, dest="batch_size", type=int)
    parser.add_argument('--l2', action="store", default=0.0, dest="l2", type=float)

    opts = parser.parse_args(sys.argv[1:])
    return opts


def main():
    options = get_params()

    model = baseline_model(ctx_shape=(None, options.ctx_max_len),
                           ans_shape=(None, options.ans_max_len),
                           vocab_size=options.voca_size,
                           emb_size=options.emb_size,
                           hid_size=options.hid_size,
                           weight_regularizer=options.l2)

    model.summary()

    model.compile(optimizer=optimizers.get(options.optimizer)(options.learn_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x=, )



if __name__ == '__main__':
