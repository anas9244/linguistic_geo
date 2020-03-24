
import matplotlib.pyplot as plt

import pickle
import seaborn as sns
# sns.set(font="monospace")
import scipy.spatial as sp
import scipy.cluster.hierarchy as hc
from scipy.cluster.hierarchy import ward, dendrogram
import numpy as np

import plotly.graph_objects as go


def show_mat(mat, names, target_names):

    fig = plt.figure()
    ax = fig.add_subplot()
    cax = ax.matshow(mat, cmap='PuRd')
    fig.colorbar(cax)
    ticks = np.arange(0, len(names), 1)

    ax.set_xticks(ticks,)
    ax.set_yticks(ticks)
    highlight_names = []
    for name in names:
        if name in target_names:
            highlight_names.append(name)
        else:
            highlight_names.append(None)

    ax.set_xticklabels(highlight_names, size=5, rotation=45)
    ax.set_yticklabels(highlight_names, size=5)

    plt.axis('off')
    # ax.xaxis.set

    #plt.savefig('highlight_mat.png', dpi=1000)
    plt.show()


def show_plt(gran, method, geo_sort):

    noremd_mat_file = open("noremd_mat_" + gran + ".pickle", "rb")
    noremd_mat = pickle.load(noremd_mat_file)
    noremd_mat_file.close()

    names_file = open("names_" + gran + ".pickle", "rb")
    names = pickle.load(names_file)
    names_file.close()

    print(len(noremd_mat))

    if geo_sort:
        geo_mat_file = open("geo_mat_" + gran + ".pickle", "rb")
        geo_mat = pickle.load(geo_mat_file)
        geo_mat_file.close()

        linkage = hc.linkage(sp.distance.squareform(geo_mat), method=method)
    else:
        linkage = hc.linkage(sp.distance.squareform(
            noremd_mat), method=method)

    dendo = dendrogram(linkage)

    leaves = dendo['leaves']

    cluster_names = []
    for i in leaves:
        cluster_names.append(names[i])

    highlight_mat = np.empty(noremd_mat.shape)
    #taget_names = ['OH', 'UT', 'IN', 'OK', 'KY', 'ND', 'WV']
    taget_names = ['Lewisville, TX', 'Santa Clarita, CA',
                   'Spokane, WA', 'Riverview, FL']

    target_ind = [i for i in range(len(cluster_names))
                  if cluster_names[i] in taget_names]

    print(len(target_ind))

    for i in range(len(leaves)):
        for j in range(len(leaves)):

            # if i in target_ind:
            #     highlight_mat[i][j] = 1.8
            # elif j in target_ind:
            #     highlight_mat[i][j] = 1.8
            # else:
            highlight_mat[i][j] = noremd_mat[leaves[i]][leaves[j]]

    show_mat(highlight_mat, cluster_names, taget_names)


show_plt('cities', 'average', False)

# positions = [i + 0.5 for i in range(len(names))]
# dendo.ax_heatmap.set_xticklabels(cluster_names)

# dendo.ax_heatmap.xaxis.set_ticks(positions)
# dendo.ax_heatmap.xaxis.set_ticklabels(
#     cluster_names, rotation=45, fontsize=10)

# dendo.ax_heatmap.yaxis.set_ticks(positions)
# dendo.ax_heatmap.yaxis.set_ticklabels(
#     cluster_names, rotation=45, fontsize=10)

# dendo.ax_heatmap.xaxis.set_ticks_position('top')
# dendo.ax_heatmap.yaxis.set_ticks_position('left')
# plt.xticks(rotation=0)
# # hide lables

# # dendo.ax_heatmap.axis('off')

# dendo.ax_row_dendrogram.set_visible(False)
# # dendo.ax_col_dendrogram.set_visible(False)

# ll, bb, ww, hh = dendo.ax_heatmap.get_position().bounds

# print(ll, bb, ww, hh)
# dendo.ax_heatmap.set_position([ll, bb - 0.04 * hh, ww, hh])
# dendo.ax_heatmap.set_title(gran + ", method: " + method +
#                            ", sorted by geo-dist: " + str(geo_sort))
# # plt.title(gran + "Method: " + method +
# #           " Sorjetted by geo-dist: " + str(geo_sort), fontsize=7)

# plt.show()

# return noremd_mat, leafs, cluster_names
