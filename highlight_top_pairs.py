
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
    cax = ax.matshow(mat, cmap='Reds')
    fig.colorbar(cax)
    ticks = np.arange(0, len(names), 1)

    ax.set_xticks(ticks,)
    ax.set_yticks(ticks)
    highlight_names = []
    # for name in names:
    #     if name in target_names:
    #         highlight_names.append(name)
    #     else:
    #         highlight_names.append(None)

    ax.set_xticklabels(names, size=4, rotation=45)
    ax.set_yticklabels(names, size=4)

    # plt.axis('off')
    # ax.xaxis.set

    #plt.savefig('archive/jsd.png', dpi=500)
    plt.show()


def show_plt(gran, method, geo_sort):

    #noremd_mat_file = open("iter_results_jsd_states.pickle", "rb")

    noremd_mat_file = open("noremd_mat_" + gran + ".pickle", "rb")
    noremd_mat = pickle.load(noremd_mat_file)
    #noremd_mat = sum(noremd_mat) / len(noremd_mat)
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

    state_deltas = []

    for index, x in enumerate(noremd_mat):
        values = [val for val in x if val > 0]
        min_dist = min(values)
        min_dist_i = np.argmin(values)
        state01 = index
        state02 = min_dist_i

        state_deltas.append((state01, state02))

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
            if j == state_deltas[i][1]:
                highlight_mat[i][j] = -1
            else:
                highlight_mat[i][j] = noremd_mat[leaves[i]][leaves[j]]

    show_mat(highlight_mat, cluster_names, taget_names)


show_plt('cities', 'average', False)
