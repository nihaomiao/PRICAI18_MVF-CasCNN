# Building CSV from heatmap and calculating FROC
# Author: Haomiao Ni
from scipy.sparse import load_npz, coo_matrix
import numpy as np
import os
from scipy.signal import medfilt2d
import csv
from MVFCasCNN import Calculating_FROC

# Building CSV files by Non-Maximum Suppression
def BuildCSVUsingNMS(MatPath, CSVPath, down_sample_rate, MedPara, t, r):
    FileList = os.listdir(MatPath)
    for FileName in FileList:
        if os.path.splitext(FileName)[-1] == ".npz":
            file = os.path.join(MatPath, FileName)
            heatmap = load_npz(file)
            heatmap = heatmap.todense()
            if MedPara:
                heatmap = medfilt2d(heatmap, (MedPara, MedPara))
            heatmap = coo_matrix(heatmap)
            data = heatmap.data
            row = heatmap.row
            col = heatmap.col
            databin = data > t
            data = data[databin]
            prefix = FileName.split('_')[0]+'_'+FileName.split('_')[1]
            CSVName = prefix + '.csv'
            CSVFile = os.path.join(CSVPath, CSVName)
            if not len(data):
                with open(CSVFile, "wb") as f:
                    print "generate Empty File:" + CSVFile
                    writer = csv.writer(f)
                    writer.writerow([0, down_sample_rate, down_sample_rate])
                continue
            row = row[databin]
            assert not np.any(np.diff(row)<0)
            col = col[databin]

            num = len(data)
            radius = r
            for ind in range(num):
                center_row = row[ind]
                center_col = col[ind]
                center_val = data[ind]
                if center_val == 0:
                    continue
                for temp_ind in range(ind+1, num):
                    temp_row = row[temp_ind]
                    temp_col = col[temp_ind]
                    temp_val = data[temp_ind]
                    if temp_val == 0:
                        continue
                    if np.abs(temp_row - center_row)>radius:
                        break
                    if np.abs(temp_col - center_col)>radius:
                        continue
                    if np.square(temp_row - center_row) + np.square(temp_col - center_col) > radius*radius:
                        continue
                    if temp_val>center_val:
                        data[ind] = 0
                        break
                    if temp_val<=center_val:
                        data[temp_ind] = 0

            zerobin = data!=0
            data = data[zerobin]
            row = row[zerobin]
            col = col[zerobin]

            CSVMat = np.column_stack((data, col*down_sample_rate, row*down_sample_rate))
            with open(CSVFile, "wb") as f:
                print "generate:" + CSVFile
                writer = csv.writer(f)
                writer.writerows(CSVMat)

def GetParameter(CSVPath):
    ParaList = CSVPath.split('-')
    MedPara = int(ParaList[-6])
    thre = float(ParaList[-2])
    radius = int(ParaList[-4])
    return MedPara, thre, radius

if __name__=="__main__":
    MatPath = ''
    mask_folder = ''
    print MatPath

    down_sample_rate = 128

    CSVPath = ''
    if not os.path.exists(CSVPath):
        os.mkdir(CSVPath)

    MedPara, thre, radius = GetParameter(CSVPath)
    BuildCSVUsingNMS(MatPath, CSVPath, down_sample_rate, MedPara, thre, radius)

    # Calculate FROC
    print "Begin calculating FROC."
    EVALUATION_MASK_LEVEL = 5  # Image level at which the evaluation is done
    L0_RESOLUTION = 0.243  # pixel resolution at level 0
    Calculating_FROC.CalcuateFROC(mask_folder, CSVPath,
                                  EVALUATION_MASK_LEVEL, L0_RESOLUTION)

