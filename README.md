MVF-CasCNN
====

The Tensorflow implementation of our pricai18 paper [Multiple Visual Fields Cascaded Convolutional 
Neural Network for Breast Cancer Detection](https://link.springer.com/chapter/10.1007/978-3-319-97304-3_41).

<div align=center><img src="examples/overview.jpg" width="568px" height="324px"/></div>

Dependencies
----
Python 2.7， Tensorflow 1.1.0, openslide-python 1.1.1, scikit-image 0.13.0, scipy 1.1.0, numpy 1.13.1, matplotlib 2.0.2.

Data Preparation
----
Our benchmark dataset is from [Camelyon16 Challenge](https://camelyon16.grand-challenge.org/). You can download the dataset from this [link](https://camelyon16.grand-challenge.org/Download/).

Quick Start
----
This code is mainly for testing. You can run the inference on testing dataset as follows:
1. We firstly utilize LVF-CNN (large visual field CNN) to coarsely locate the possible lesion areas. So to obtain LVF patches, please run **pathology_divide_test40X1196.py** to preprocess testing slides and divide them into patches.

2. Please run **pathology_test_tfrecords.py** to convert image patches to [tfrecord format](https://www.tensorflow.org/api_guides/python/python_io). You may find more useful information from this [link](https://www.tensorflow.org/api_guides/python/reading_data).

3. Please run **pathology_test_test.py** to create the initial LVF heatmaps. Also, don't forget to set your own paths in **pathology_test_preprocessing.py** and **pathology_test_eval.py**. 

4. To get the final LVF heatmaps, please run **UpsampleLVFHeatmap.py**.

5. Subsequently, we leverage these LVF heatmaps to generate small visual field patches. Please run **pathology_generate_tumor_candidate.py**. Then, use **pathology_test_tfrecords.py** to convert patches to tfrecord format and run **pathology_test_test.py** to produce the final MVF-CNN heatmaps. You can adopt **SaveHeatmapToFig.py** to visualize these heatmaps.

6. Finally, please utilize **Evaluating_AUC.py** and **Evaluating_FROC.py** to measure the performance. 

Citing MVF-CasCNN
----
If you find our approaches useful in your research, please consider citing:
```
@inproceedings{ni2018multiple,
  title={Multiple Visual Fields Cascaded Convolutional Neural Network for Breast Cancer Detection},
  author={Ni, Haomiao and Liu, Hong and Guo, Zichao and Wang, Xiangdong and Jiang, Taijiao and Wang, Kuansong and Qian, Yueliang},
  booktitle={Pacific Rim International Conference on Artificial Intelligence},
  pages={531--544},
  year={2018},
  organization={Springer}
}
```
For any problems with the code, please feel free to contact me: homerhm.ni@gmail.com

Acknowledgement
----
Our MVF-CasCNN borrowed some functions from [Inception in Tensorflow](https://github.com/tensorflow/models/tree/master/research/inception).
