# Building Test TFRecord for the Test Patches
# Author: Haomiao Ni
# Acknowledgement: We borrow some codes from the following url:
# https://github.com/tensorflow/models/blob/master/
# research/inception/inception/data/build_image_data.py
import os
# GPU Forbidden
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from datetime import datetime
import sys
import threading
import tensorflow as tf
import numpy as np
from math import ceil
import time

# Setting Paths
tf.app.flags.DEFINE_string('test_directory', '',
                           'Testing data directory')

tf.app.flags.DEFINE_string('output_directory', '',
                           'Output data directory')

tf.app.flags.DEFINE_string('log_directory', '',
                           'log directory')

tf.app.flags.DEFINE_integer('down_sample_rate', 128,
                            'down_sample_rate')

tf.app.flags.DEFINE_integer('patch_side', 299, # 1196 for LVF-CNN OR 299 for SVF-CNN
                            'the side of patch')

tf.app.flags.DEFINE_integer('num_threads', 32, 'the maximum number of threads')

FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_to_example(filename, image_buffer, height, width, pos_width, pos_height):
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/pos_width': _int64_feature(pos_width),
        'image/pos_height': _int64_feature(pos_height),
        'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
        'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
    return example

class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

def _is_png(filename):
    """Determine if a file contains a PNG format image.
    Args:
      filename: string, path of the image file.
    Returns:
      boolean indicating if the image is a PNG.
    """
    return '.png' in filename

def _process_image(filename, coder):
    """Process a single image file.
    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()

    # Read the position
    tmp_name = filename.split('_')
    pos_width = (int(tmp_name[-2])+FLAGS.patch_side/2)/FLAGS.down_sample_rate
    pos_height = (int((tmp_name[-1])[:-4])+FLAGS.patch_side/2)/FLAGS.down_sample_rate

    # Convert any PNG to JPEG's for consistency.
    if _is_png(filename):
        print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width, pos_width, pos_height

def _process_image_files_batch(coder, thread_index, ranges, filenames, num_shards):
    """Processes and saves list of images as TFRecord in 1 thread.
    Args:
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
      thread_index: integer, unique batch to run index is within [0, len(ranges)).
      ranges: list of pairs of integers specifying ranges of each batches to
        analyze in parallel.
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      num_shards: integer number of shards for this data set.
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    name = filenames[0].split('/')[-2]
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'Test_130-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        TargetDir = os.path.join(FLAGS.output_directory, name)
        output_file = os.path.join(TargetDir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]

            try:
                image_buffer, height, width, pos_width, pos_height = _process_image(filename, coder)
            except Exception as e:
                print(e)
                print('SKIPPED: Unexpected eror while decoding %s.' % filename)
                continue

            example = _convert_to_example(filename, image_buffer, height, width, pos_width, pos_height)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()

def _process_image_files(filenames, num_shards, global_num_threads):
    """Process and save list of images as TFRecord of Example protos.
    Args:
      filenames: list of strings; each string is a path to an image file
      num_shards: integer number of shards for this data set.
    """
    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), global_num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (global_num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, filenames,
                num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(filenames)))
    sys.stdout.flush()

def _find_image_files(data_dir, log):
    print('Determining list of input files from %s.' % data_dir)

    filenames = []

    # Construct the list of JPEG files.
    jpeg_file_path = '%s/*' % (data_dir)
    matching_files = tf.gfile.Glob(jpeg_file_path)
    filenames.extend(matching_files)

    if len(filenames) == 0:
        print('Found %d JPEG files inside %s.' %
              (len(filenames), data_dir))
        log.writelines("PatchesNum:" + str(len(filenames)) + '\n')
        return filenames, None, None

    num_shards = int(ceil(len(filenames)/6000.0))
    num_shards = num_shards + num_shards%2
    # set the num of threads
    global_num_threads = 2
    for i in range(2, FLAGS.num_threads+1):
        if not num_shards%i:
            global_num_threads = i

    assert not num_shards % global_num_threads, (
        'Please make the global_num_threads commensurate with FLAGS.train_shards')
    print('Found %d JPEG files inside %s.' %
          (len(filenames), data_dir))
    log.writelines("PatchesNum:"+str(len(filenames))+'\n')
    return filenames, num_shards, global_num_threads

def _process_dataset(directory, log):
    filenames, num_shards, global_num_threads = _find_image_files(directory, log)
    if len(filenames) == 0:
        return
    _process_image_files(filenames, num_shards, global_num_threads)

def main(unused_argv):
    print('Saving results to %s' % FLAGS.output_directory)

    DirList = os.listdir(FLAGS.test_directory)

    # Setting Log
    LogPath = FLAGS.log_directory
    localtime = time.localtime(time.time())
    LogName = "Test_TFRecord_log_"+str(localtime.tm_year)+"_"+str(localtime.tm_mon)+"_"+str(localtime.tm_mday)\
                +"_"+str(localtime.tm_hour)+"_"+str(localtime.tm_min)+"_"+str(localtime.tm_sec)
    LogFile = os.path.join(LogPath, LogName)

    with open(LogFile, 'a', 0) as log:
        total_start = time.time()
        for DirName in DirList:
            DirPath = os.path.join(FLAGS.test_directory, DirName)
            if os.path.isdir(DirPath):
                start = time.time()
                print "Building TFRecords from " + DirPath
                log.writelines("Building TFRecords from " + DirPath + '\n')
                TargetDir = os.path.join(FLAGS.output_directory, DirName)
                if not os.path.exists(TargetDir):
                    os.mkdir(TargetDir)
                _process_dataset(DirPath, log)

                stop = time.time()
                print "processing time:", stop - start
                log.writelines("processing time:" + str(stop - start) + '\n')
                log.writelines('\n')

        total_stop = time.time()
        print "total processing time:", total_stop - total_start
        log.writelines("total processing time:" + str(total_stop - total_start) + '\n')

if __name__ == '__main__':
    tf.app.run()