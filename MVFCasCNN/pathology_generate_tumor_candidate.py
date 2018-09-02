# Using heatmaps from LVF-CNN to extract patches of SVF
import os
# GPU Forbidden
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from datetime import datetime
import sys
import threading
import tensorflow as tf
import numpy as np
import openslide
import time
from scipy.sparse import coo_matrix, load_npz
import glob

tf.app.flags.DEFINE_string('test_directory', '',
                           'Testing data directory')

tf.app.flags.DEFINE_string('patch_directory', '',
                           'patch directory')

tf.app.flags.DEFINE_string('log_directory', '',
                           'log directory')

tf.app.flags.DEFINE_integer('down_sample_rate', 128,
                            'down_sample_rate')

tf.app.flags.DEFINE_string('lvfmap_directory', '',
                           'large visual field map directory')

tf.app.flags.DEFINE_float('lvf_thre', 0.3,
                            'threshold for large visual field')

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
            patch = slide.read_region((FLAGS.down_sample_rate * w - 299 / 2, FLAGS.down_sample_rate * h - 299 / 2), 0, (299, 299))
        except:
            slide = openslide.open_slide(tiff_dir)
            print "Can not read the point (" + str(FLAGS.down_sample_rate * w - 299 / 2) + ',' + str(FLAGS.down_sample_rate * h - 299 / 2) + ") for " + FileName
            log.writelines("Can not read the point (" + str(FLAGS.down_sample_rate * w - 299 / 2) + ',' + str(
                FLAGS.down_sample_rate * h - 299 / 2) + ") for " + FileName + '\n')
            continue
        else:
            patch_name = os.path.join(TargetDir, FileName[:-4] + '_' + str(FLAGS.down_sample_rate * w - 299 / 2) + '_' + str(
                FLAGS.down_sample_rate * h - 299 / 2) + '.jpg')
            patch = patch.convert('RGB')
            patch.save(patch_name, "JPEG")
            #print patch_name
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
    # Selecting the target patches using large visual field.
    # And count the patch num
    SlideName = (tiff_dir.split('/')[-1]).split('.')[0]
    LVFMapName = os.path.join(FLAGS.lvfmap_directory, SlideName + '_*.npz')
    sparse_s_bin = load_npz(glob.glob(LVFMapName)[0])

    sparse_s_bin = coo_matrix(sparse_s_bin.transpose()>FLAGS.lvf_thre)
    num_patches = len(sparse_s_bin.data)
    log.writelines("PatchNum:" + str(num_patches) + '\n')
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

    LogPath = FLAGS.log_directory
    localtime = time.localtime(time.time())
    LogName = "Test_Patches_log_" + str(localtime.tm_year) + "_" + str(localtime.tm_mon) + "_" + str(localtime.tm_mday) \
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
