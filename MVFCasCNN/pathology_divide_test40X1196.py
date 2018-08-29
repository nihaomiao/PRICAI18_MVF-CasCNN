# Dividing 40X1196 Large Visual Field Testing Patches using multi-thread
# Author: Haomiao Ni

import os
# GPU Forbidden
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from datetime import datetime
import sys
import threading
import tensorflow as tf
import numpy as np
import openslide
from skimage import filters
import time
from scipy.sparse import coo_matrix
import math

# Setting Paths
tf.app.flags.DEFINE_string('test_directory', '',
                           'Testing data directory')

tf.app.flags.DEFINE_string('patch_directory', '',
                           'patch directory')

tf.app.flags.DEFINE_string('log_directory', '',
                           'log directory')

tf.app.flags.DEFINE_integer('down_sample_rate', 128,
                            'down sample rate')

tf.app.flags.DEFINE_integer('global_num_threads', 32, 'the number of threads')

FLAGS = tf.app.flags.FLAGS


def __divide_patches_batch(slide, tiff_dir, TargetDir, FileName, thread_index,
                           ranges, sparse_s_bin, num_patches, log):
    num_patches_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(ranges[thread_index][0], ranges[thread_index][1]):
        shard = s
        w = sparse_s_bin.row[shard]
        h = sparse_s_bin.col[shard]

        try:
            patch = slide.read_region((FLAGS.down_sample_rate * w - 1196 / 2, FLAGS.down_sample_rate * h - 1196 / 2), 0, (1196, 1196))
        except:
            slide = openslide.open_slide(tiff_dir)
            print "Can not read the point (" + str(FLAGS.down_sample_rate * w - 1196 / 2) + ',' + str(FLAGS.down_sample_rate * h - 1196 / 2) + ") for " + FileName
            log.writelines("Can not read the point (" + str(FLAGS.down_sample_rate * w - 1196 / 2) + ',' + str(
                FLAGS.down_sample_rate * h - 1196 / 2) + ") for " + FileName + '\n')
            continue
        else:
            patch_name = os.path.join(TargetDir, FileName[:-4] + '_' + str(FLAGS.down_sample_rate * w - 1196 / 2) + '_' + str(
                FLAGS.down_sample_rate * h - 1196 / 2) + '.jpg')
            patch = patch.convert('RGB')
            patch.save(patch_name, "JPEG")
            # print patch_name
            counter += 1

    print('%s [thread %d]: Wrote %d images to %d shards.' %
            (datetime.now(), thread_index, counter, num_patches_in_thread))
    sys.stdout.flush()

def _divide_patches(tiff_dir, TargetDir, FileName, sparse_s_bin, num_patches,
                    global_num_threads, log):
    slide = openslide.open_slide(tiff_dir)
    spacing = np.linspace(0, len(sparse_s_bin.data), global_num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (global_num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    threads = []
    for thread_index in range(len(ranges)):
        args = (slide, tiff_dir, TargetDir, FileName, thread_index, ranges, sparse_s_bin,
                num_patches, log)
        t = threading.Thread(target=__divide_patches_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished dividing all %d patches.' %
          (datetime.now(), len(sparse_s_bin.data)))
    sys.stdout.flush()

def _preprocessing_tiff(tiff_dir, log):
    # Deleting the background of the slide
    # And count the number of patches
    slide = openslide.open_slide(tiff_dir)
    low_dim_level = slide.get_best_level_for_downsample(FLAGS.down_sample_rate)
    assert low_dim_level == math.log(FLAGS.down_sample_rate, 2)
    low_dim_size = slide.level_dimensions[low_dim_level]
    low_dim_img = slide.read_region((0, 0), low_dim_level, low_dim_size)

    # --transform to hsv space
    low_hsv_img = low_dim_img.convert("HSV")
    _, low_s, _ = low_hsv_img.split()

    # --OSTU threshold
    low_s_thre = filters.threshold_otsu(np.array(low_s))
    low_s_bin = low_s > low_s_thre
    low_s_bin = low_s_bin.transpose()

    # sample rate : 512
    width = low_s_bin.shape[0]
    height = low_s_bin.shape[1]
    sample_mat = np.zeros((width, height), dtype=np.bool)
    sample_mat[0:width:4, 0:height:4] = True
    low_s_bin = np.logical_and(low_s_bin, sample_mat)

    num_patches = np.sum(low_s_bin)
    assert num_patches > 0

    sparse_s_bin = coo_matrix(low_s_bin)
    log.writelines("PatchNum:" + str(np.sum(low_s_bin)) + '\n')
    assert num_patches==len(sparse_s_bin.data)
    # set the num of threads
    global_num_threads = FLAGS.global_num_threads
    log.writelines("ThreadNum:"+str(global_num_threads)+'\n')

    return sparse_s_bin, num_patches, global_num_threads

def _process_tiff(directory, TargetDir, FileName, log):
    sparse_s_bin, num_patches, global_num_threads = _preprocessing_tiff(directory, log)
    _divide_patches(directory, TargetDir, FileName, sparse_s_bin, num_patches, global_num_threads, log)

def main(unused_argv):
    print('Saving patches to %s' % FLAGS.patch_directory)

    FileList = os.listdir(FLAGS.test_directory)

    # Setting Log
    LogPath = FLAGS.log_directory
    localtime = time.localtime(time.time())
    LogName = "Test_LVF_Patches_log_" + str(localtime.tm_year) + "_" + str(localtime.tm_mon) + "_" + str(localtime.tm_mday) \
              + "_" + str(localtime.tm_hour) + "_" + str(localtime.tm_min) + "_" + str(localtime.tm_sec)
    LogFile = os.path.join(LogPath, LogName)

    with open(LogFile, 'a', 0) as log:
        total_start = time.time()
        for FileName in FileList:
            if os.path.splitext(FileName)[1] == '.tif':
                start = time.time()
                FilePath = os.path.join(FLAGS.test_directory, FileName)

                print "Dividing patches from " + FilePath
                log.writelines("processing:" + FilePath + '\n')

                TargetDir = os.path.join(FLAGS.patch_directory, FileName[:-4])
                if not os.path.exists(TargetDir):
                    os.mkdir(TargetDir)
                # divide test patches
                _process_tiff(FilePath, TargetDir, FileName, log)

                stop = time.time()
                print "processing time:", stop - start
                log.writelines("processing time:" + str(stop - start) + '\n')
                log.writelines('\n')

        total_stop = time.time()
        print "total processing time:", total_stop - total_start
        log.writelines("total processing time:" + str(total_stop - total_start) + '\n')


if __name__ == '__main__':
    tf.app.run()