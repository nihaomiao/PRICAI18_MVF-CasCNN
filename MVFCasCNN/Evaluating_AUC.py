# Evaluating AUC
# Author: Haomiao Ni
from scipy.sparse import load_npz
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d
import os

def MedianTest(MatPath, MaskPath, FileNum, MedPara):
    FileLabel = np.zeros(FileNum)
    FileProb = np.zeros(FileNum)
    FileList = os.listdir(MatPath)
    cnt = 0
    for FileName in FileList:
        if os.path.splitext(FileName)[-1]==".npz":
            print "processing:", FileName
            MaskName = FileName.split('_')[0]+'_'+FileName.split('_')[1]+'_Mask.tif'
            Mask = os.path.join(MaskPath, MaskName)
            file = os.path.join(MatPath, FileName)
            heatmap = load_npz(file)
            heatmap = heatmap.todense()
            if MedPara:
                MedianMap = medfilt2d(heatmap, (MedPara, MedPara))
            else:
                MedianMap = np.array(heatmap)

            max_val = np.max(MedianMap)
            FileProb[cnt] = max_val
            FileLabel[cnt] = int(os.path.isfile(Mask))
            cnt = cnt + 1

    # draw auc curve
    fpr, tpr, _ = roc_curve(FileLabel, FileProb, pos_label=1)
    roc_auc =  auc(fpr, tpr)
    print roc_auc_score(FileLabel, FileProb)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b',
             label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


if __name__ =="__main__":
    MatPath = '' # the path of heatmaps sparse matrices
    MaskPath = ''
    FileNum = 128
    MedPara = 0 # median filter parameter
    MedianTest(MatPath, MaskPath, FileNum, MedPara)

