from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import tensorflow as tf
from new_models.models import base_model
from new_models.trainer import build_trainer
from new_models.data import create_input_fn
FLAGS = tf.app.flags.FLAGS


# set input
is_training = tf.placeholder('bool', [], name='is_training')

train_inputs, train_target = create_input_fn(input_files=[FLAGS.train_file],
                                             batch_size=FLAGS.batch_size,
                                             num_epochs=FLAGS.num_epochs,
                                             mode=tf.contrib.learn.ModeKeys.TRAIN)()

valid_inputs, valid_target = create_input_fn(input_files=[FLAGS.valid_file],
                                             batch_size=FLAGS.batch_size,
                                             num_epochs=1,
                                             mode=tf.contrib.learn.ModeKeys.EVAL)

ctx_idx_data, ans_idx_data, gt_label_data = tf.cond(is_training,
                                                    lambda: (train_inputs['context'], train_inputs['answer'], train_target),
                                                    lambda: (valid_inputs['context'], valid_inputs['answer'], valid_target))

# set model
model_fn = base_model(rnn_size=FLAGS.rnn_size,
                      vocab_size=FLAGS.vocab_size,
                      emb_size=FLAGS.emb_size)

# set train updater
update_op, train_loss, global_step = build_trainer(model_fn=model_fn,
                                                   ctx_idx_data=ctx_idx_data,
                                                   ans_idx_data=ans_idx_data,
                                                   gt_label_data=gt_label_data,
                                                   optimizer=FLAGS.optimizer,
                                                   init_lr=FLAGS.init_lr,
                                                   grad_clip=FLAGS.grad_clip)
train_op_list = [update_op, train_loss]

# set eval operations
eval_op_list = []
for k in [1, 2, 5, 10]:
    eval_op_list.append(tf.metrics.recall_at_k(labels=tf.to_float(gt_label_data),
                                               predictions=model_fn(ctx_idx_data, ans_idx_data)[0],
                                               k=k,
                                               name="recall_at_%d" % k))

# set model saver
model_saver = tf.train.Saver(tf.all_variables())

# set summary op
summary_op = tf.merge_all_summaries()

with tf.Session() as sess:

    # if resume training
    if FLAGS.resume:
        latest = tf.train.latest_checkpoint(FLAGS.train_dir)
        if not latest:
            print('No checkpoint to continue from in', FLAGS.train_dir)
            sys.exit(1)
        else:
            print('resume', latest)
            model_saver.restore(sess, latest)
    else:
        # initialize model
        sess.run(tf.global_variables_initializer())

    # set summary writer
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)


    # for each step
    for x in xrange(FLAGS.max_steps + 1):
        # check time
        start_time = time.time()

        # update global step
        step = sess.run(global_step, {is_training: True})

        # run ops
        train_outputs = sess.run(train_op_list)

        loss_value = train_outputs[-1]

        # check time
        end_time = time.time()
        step_time = end_time - start_time

        # if step for summary write
        write_summary = step % 100 and step > 1
        if write_summary:
            summary_outputs = sess.run(summary_op)
            summary_writer.add_summary(summary_outputs, step)

        # if checkpoint
        if step > 1 and step % 100 == 0:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            model_saver.save(sess=sess,
                             save_path=os.path.join(FLAGS.train_dir, "model_%d.ckpt" % step),
                             global_step=global_step)

        # if validation
        if step > 1 and step % 100 == 0:
            recall_k_list = sess.run(eval_op_list, {is_training: False})
            print('Validation top1 error %.2f' % top1_error_value)






