from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def build_trainer(model_fn,
                  ctx_idx_data,
                  ans_idx_data,
                  gt_label_data,
                  optimizer=tf.train.AdadeltaOptimizer,
                  init_lr=0.01,
                  grad_clip=1.0):

    # model output
    model_logits, model_probs = model_fn(ctx_idx_data, ans_idx_data)

    # model parameters
    model_params = tf.trainable_variables()

    # model cost
    model_sample_cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=model_logits, labels=tf.to_float(gt_label_data))
    model_mean_cost = tf.reduce_mean(model_sample_cost)

    # model gradients
    model_grads = tf.gradients(ys=model_mean_cost, xs=model_params)
    model_grads, model_grad_norm = tf.clip_by_global_norm(t_list=model_grads, clip_norm=grad_clip)

    # model updater
    global_step = tf.contrib.framework.get_or_create_global_step()
    learn_rate = tf.Variable(initial_value=init_lr, trainable=False)
    model_optimizer = optimizer(learning_rate=learn_rate, name='optimizer')

    # update op
    update_op = model_optimizer.apply_gradients(grads_and_vars=zip(model_grads, model_params),
                                                global_step=global_step)

    return update_op, model_mean_cost, global_step









