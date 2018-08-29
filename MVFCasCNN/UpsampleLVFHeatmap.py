# Upsampling the heatmaps of LVF
# Author: Haomiao Ni
# We first "squeeze" the LVF heatmaps and then upsample them.
import os
from scipy.sparse import load_npz, save_npz, csr_matrix
import numpy as np
import time

def DownSample(MapPath, ThumbPath):
    MapList = os.listdir(MapPath)
    for MapName in MapList:
        print MapName
        MapFile = os.path.join(MapPath, MapName)
        heatmap = load_npz(MapFile)
        height = heatmap.shape[0]
        width = heatmap.shape[1]
        thumbnail = heatmap[0:height:4, 0:width:4]
        save_npz(os.path.join(ThumbPath, MapName.replace('MAP','THUMBNAIL')),
                 thumbnail)

def UpSample(MapPath, UpsamplePath):
    MapList = os.listdir(MapPath)
    for MapName in MapList:
        print MapName
        MapFile = os.path.join(MapPath, MapName)
        heatmap = load_npz(MapFile)
        height = heatmap.shape[0]
        width = heatmap.shape[1]
        UpsampleMap = np.zeros((height*4, width*4))
        for y in range(height):
            for x in range(width):
                UpsampleMap[4*y:4*(y+1), 4*x:4*(x+1)] = heatmap[y, x]
        save_npz(os.path.join(UpsamplePath, MapName.replace('THUMBNAIL','UPSAMPLE')),
                 csr_matrix(UpsampleMap))


if __name__=='__main__':
    start = time.time()
    MapPath = ''
    ThumbPath = ''
    UpsamplePath = ''
    DownSample(MapPath, ThumbPath)
    UpSample(ThumbPath, UpsamplePath)
    print "procesing time: ", time.time()-start
