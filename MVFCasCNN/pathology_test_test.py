# Implementing LVF-CNN OR SVF-CNN
# Author: Haomiao Ni
# Remember to set parameters in pathology_test_eval.py and pathology_test_preprocessing.py
import os
# only use one GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import openslide
from MVFCasCNN import pathology_test_eval
import time
import math

# Setting Path
tf.app.flags.DEFINE_string('TifPath', '',
                           'TIFF directory')
tf.app.flags.DEFINE_string('PatchPath', '',
                           'TFRecord data directory')
tf.app.flags.DEFINE_string('TFRecordPath', '',
                           'TFRecord data directory')
tf.app.flags.DEFINE_string('LogPath', '',
                           'Log directory')
tf.app.flags.DEFINE_integer('down_sample_rate', 128,
                            'down sample rate')

FLAGS = tf.app.flags.FLAGS

def main(unused_argv=None):
    DirList = os.listdir(FLAGS.TFRecordPath)

    LogPath = FLAGS.LogPath
    localtime = time.localtime(time.time())
    LogName = "Test_HeatMap_log_"+str(localtime.tm_year)+"_"+str(localtime.tm_mon)+"_"+str(localtime.tm_mday)\
                +"_"+str(localtime.tm_hour)+"_"+str(localtime.tm_min)+"_"+str(localtime.tm_sec)
    LogFile = os.path.join(LogPath, LogName)

    with open(LogFile, 'a', 0) as log:
        total_start = time.time()
        for DirName in DirList:
            DirPath = os.path.join(FLAGS.TFRecordPath, DirName)
            if os.path.isdir(DirPath):
                start = time.time()
                TargetName = DirPath.split('/')[-1]

                # get the width and height of the low resolution image
                TifName = os.path.join(FLAGS.TifPath, TargetName+'.tif')
                log.writelines("processing:"+TifName+'\n')
                slide = openslide.open_slide(TifName)
                low_dim_level = slide.get_best_level_for_downsample(FLAGS.down_sample_rate)
                assert low_dim_level == math.log(FLAGS.down_sample_rate, 2)
                low_dim_size = slide.level_dimensions[low_dim_level]
                img_width = low_dim_size[0]
                img_height = low_dim_size[1]

                # get the number of patches
                PatchName = os.path.join(FLAGS.PatchPath, TargetName)
                num_examples = len([name for name in os.listdir(PatchName) if os.path.isfile(os.path.join(PatchName, name))])
                log.writelines("PatchesNum:"+str(num_examples)+'\n')
                pathology_test_eval.evaluate(DirPath, num_examples, img_height, img_width)

                stop = time.time()
                print "processing time:", stop - start
                log.writelines("processing time:" + str(stop - start) + '\n')
                log.writelines('\n')

        total_stop = time.time()
        print "total processing time:", total_stop - total_start
        log.writelines("total processing time:" + str(total_stop - total_start) + '\n')

if __name__ == '__main__':
    tf.app.run()

