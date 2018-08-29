# Transforming heatmap Matrix to Figure and save them
# Author: Haomiao Ni
import os
import matplotlib.pyplot as plt
from scipy.sparse import load_npz, lil_matrix
from scipy.signal import medfilt2d
import numpy as np

def run(MatPath, FigPath, heatthre, medflag):
    dpi = 1000.0
    FileList = os.listdir(MatPath)
    FileList.sort()
    plt.ioff()
    fig = plt.figure(frameon=False)
    for FileName in FileList:
        print FileName
        if os.path.splitext(FileName)[1] == '.npz':
            file = os.path.join(MatPath, FileName)
            heatmap = load_npz(file)
            heatmap = lil_matrix(heatmap)
            heatmap = np.array(heatmap.todense())

            if heatthre:
                # threshold 0.5
                heatmap[np.logical_and(heatmap<0.5, heatmap>0)] = 0.1
            if medflag:
                # post processing
                heatmap = medfilt2d(heatmap, (3, 3))

            heatmap[0, 0] = 1.0
            fig.clf()
            fig.set_size_inches(heatmap.shape[1]/dpi, heatmap.shape[0]/dpi)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            cm = plt.cm.get_cmap('jet')
            ax.imshow(heatmap, cmap=cm, aspect='auto')

            postfix = FileName.split('_')[-1]
            FigName = FileName.replace(postfix,"FIG.jpg")
            fig.savefig(os.path.join(FigPath, FigName), dpi=int(dpi))

if __name__ == "__main__":
    heatthre = False # choose False to show those pixels whose predictions are less than 0.5
    medflag = False # choose True to median filter heatmaps
    MatPath = ''
    FigPath = ""
    if not os.path.exists(FigPath):
        os.makedirs(FigPath)
    run(MatPath, FigPath, heatthre, medflag)