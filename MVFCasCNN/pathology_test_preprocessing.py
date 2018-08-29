# used by pathology_test_eval.py
# Preprocessing the TFRecords
# Author: Haomiao Ni
# Acknowledgement: We borrow some codes from the following url:
# https://github.com/tensorflow/models/blob/master/research/
# inception/inception/image_processing.py
import os
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 256,#16 for LVF-CNN OR 256 for SVF-CNN, please
                            # setting it according to the memory of your GPU.
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('image_size', 299,#1196 for LVF-CNN OR 299 for SVF-CNN
                            """Provide square images of this size.""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """Number of preprocessing threads per tower. """
                            """Please make this a multiple of 4.""")
tf.app.flags.DEFINE_integer('num_readers', 4,
                            """Number of parallel readers during train.""")
tf.app.flags.DEFINE_integer('input_queue_memory_factor', 1,
                            """Size of the queue of preprocessed images. """
                            """Default is ideal but try smaller values, e.g. """
                            """4, 2 or 1, if host memory is constrained. See """
                            """comments in code for more details.""")

def parse_example_proto(example_serialized):
    # Dense features in Example proto.
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/pos_width': tf.FixedLenFeature([1], dtype=tf.int64,
                                                default_value=-1),
        'image/pos_height': tf.FixedLenFeature([1], dtype=tf.int64,
                                              default_value=-1),
    }

    features = tf.parse_single_example(example_serialized, feature_map)
    pos_width = tf.cast(features['image/pos_width'], dtype=tf.int32)
    pos_height = tf.cast(features['image/pos_height'], dtype=tf.int32)

    return features['image/encoded'], pos_width, pos_height

def decode_jpeg(image_buffer, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.

    Args:
      image_buffer: scalar string Tensor.
      scope: Optional scope for name_scope.
    Returns:
      3-D float Tensor with values ranging from [0, 1).
    """
    with tf.name_scope(values=[image_buffer], name=scope,
                       default_name='decode_jpeg'):
        # Decode the string as an RGB JPEG.
        # Note that the resulting image contains an unknown height and width
        # that is set dynamically by decode_jpeg. In other words, the height
        # and width of image is unknown at compile-time.
        image = tf.image.decode_jpeg(image_buffer, channels=3)

        # After this point, all image pixels reside in [0,1)
        # until the very end, when they're rescaled to (-1, 1).  The various
        # adjust_* ops all require this range for dtype float.
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image

def image_preprocessing(image_buffer, thread_id=0):
    image = decode_jpeg(image_buffer)
    height = FLAGS.image_size
    width = FLAGS.image_size

    image.set_shape([height, width, 3])

    # Finally, rescale to [-1,1] instead of [0, 1)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image

def batch_inputs(dirname, batch_size, num_preprocess_threads=None,
                 num_readers=1):
    with tf.name_scope('batch_processing'):
        tf_record_pattern = os.path.join(dirname, '%s-*' % dirname.split('/')[-1])
        data_files = tf.gfile.Glob(tf_record_pattern)
        if data_files is None:
            raise ValueError('No data files found for this directory')

        # Create filename_queue
        filename_queue = tf.train.string_input_producer(data_files,
                                                            shuffle=False,
                                                            capacity=1)
        if num_preprocess_threads is None:
            num_preprocess_threads = FLAGS.num_preprocess_threads

        if num_preprocess_threads % 4:
            raise ValueError('Please make num_preprocess_threads a multiple '
                             'of 4 (%d % 4 != 0).', num_preprocess_threads)

        if num_readers is None:
            num_readers = FLAGS.num_readers

        if num_readers < 1:
            raise ValueError('Please make num_readers at least 1')

        # Approximate number of examples per shard.
        examples_per_shard = 6000
        examples_queue = tf.FIFOQueue(
                    capacity=examples_per_shard + 3 * batch_size,
                    dtypes=[tf.string])

        # Create multiple readers to populate the queue of examples.
        if num_readers > 1:
            enqueue_ops = []
            for _ in range(num_readers):
                reader = tf.TFRecordReader()
                _, value = reader.read(filename_queue)
                enqueue_ops.append(examples_queue.enqueue([value]))

            tf.train.queue_runner.add_queue_runner(
                tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
            example_serialized = examples_queue.dequeue()
        else:
            reader = tf.TFRecordReader()
            _, example_serialized = reader.read(filename_queue)

        images_and_positions = []
        for thread_id in range(num_preprocess_threads):
            # Parse a serialized Example proto to extract the image and metadata.
            image_buffer, pos_width, pos_height = parse_example_proto(
                example_serialized)
            image = image_preprocessing(image_buffer, thread_id)
            images_and_positions.append([image, pos_width, pos_height])

        images, pos_w_batch, pos_h_batch = tf.train.batch_join(
            images_and_positions,
            batch_size=batch_size,
            capacity=2 * num_preprocess_threads * batch_size)

        # Reshape images into these desired dimensions.
        height = FLAGS.image_size
        width = FLAGS.image_size
        depth = 3

        images = tf.cast(images, tf.float32)
        images = tf.reshape(images, shape=[batch_size, height, width, depth])

        # Display the training images in the visualizer.
        tf.summary.image('images', images, 10)

        return images, tf.reshape(pos_w_batch, [batch_size]), tf.reshape(pos_h_batch, [batch_size])

def inputs(dirname, batch_size=None, num_preprocess_threads=None):
    if not batch_size:
        batch_size = FLAGS.batch_size

    # Force all input processing onto CPU in order to reserve the GPU for
    # the forward inference and back-propagation.
    with tf.device('/cpu:0'):
        images, pos_w_batches, pos_h_batches = batch_inputs(
            dirname, batch_size,
            num_preprocess_threads=num_preprocess_threads,
            num_readers=1)

    return images, pos_w_batches, pos_h_batches


