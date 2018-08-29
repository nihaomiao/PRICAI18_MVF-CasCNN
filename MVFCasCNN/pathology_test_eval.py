# used by pathology_test_test.py
# Author: Haomiao Ni
# Acknowledgement: We borrow some codes from the following url:
# https://github.com/tensorflow/models/blob/master/research/
# inception/inception/inception_eval.py
from datetime import datetime
import math
import time
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import save_npz
import tensorflow as tf

from MVFCasCNN import pathology_test_preprocessing
from MVFCasCNN import inception_model as inception

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('map_mat_dir', '',
                           """Directory where to save heatmap sparse matrix.""")

tf.app.flags.DEFINE_integer('num_classes', 2,
                            """the total number of classes.""")

tf.app.flags.DEFINE_string('specific_model', '',
                           """Directory where to read specific model checkpoints.""")


def _eval_once(saver, pred_op, pos_ws_op, pos_hs_op,
               num_examples, img_height, img_width, dirname):
    with tf.Session() as sess:
        if FLAGS.specific_model != '':
            saver.restore(sess, FLAGS.specific_model)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/imagenet_train/model.ckpt-0,
            # extract global_step from it.
            global_step = FLAGS.specific_model.split('/')[-1].split('-')[-1]
            print('Successfully loaded model from %s at step=%s.' %
                  (FLAGS.specific_model, global_step))
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(float(num_examples) / FLAGS.batch_size))
            # Counts the number of correct predictions.
            heatmap = lil_matrix((img_height, img_width), dtype=np.float32)
            step = 0

            print('%s: starting evaluation on (%s).' % (datetime.now(), dirname))
            start_time = time.time()
            while step < num_iter and not coord.should_stop():
                pred, pos_ws, pos_hs = sess.run([pred_op, pos_ws_op, pos_hs_op])
                heatmap[pos_hs, pos_ws] = pred[:, 1]

                step += 1
                if step % 10 == 0:
                    duration = time.time() - start_time
                    sec_per_batch = duration / 10.0
                    examples_per_sec = FLAGS.batch_size / sec_per_batch
                    print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                          'sec/batch)' % (datetime.now(), step, num_iter,
                                          examples_per_sec, sec_per_batch))
                    start_time = time.time()

            # save heatmap
            MapName = dirname.split('/')[-1] + '_' + str(global_step)
            save_npz(FLAGS.map_mat_dir + '/' + MapName + '_MAP.npz',
                     heatmap.tocsr())
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate(dirname, num_examples, img_height, img_width):
    if num_examples == 0:
        heatmap = lil_matrix((img_height, img_width), dtype=np.float32)
        MapName = dirname.split('/')[-1] + '_' + str(FLAGS.specific_model.split('/')[-1].split('-')[-1])
        save_npz(FLAGS.map_mat_dir + '/' + MapName + '_MAP.npz',
                 heatmap.tocsr())
        return
    with tf.Graph().as_default():
        # Get images and labels from the dataset.
        images, pos_ws, pos_hs = pathology_test_preprocessing.inputs(dirname)

        num_classes = FLAGS.num_classes

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits, _ = inception.inference(images, num_classes)

        # Calculate predictions.
        pred = tf.nn.softmax(logits)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            inception.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        _eval_once(saver, pred, pos_ws, pos_hs,
                   num_examples, img_height, img_width, dirname)
