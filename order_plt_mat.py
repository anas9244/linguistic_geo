
import matplotlib.pyplot as plt

import pickle
import seaborn as sns
# sns.set(font="monospace")
import scipy.spatial as sp
import scipy.cluster.hierarchy as hc

import numpy as np


# def translate(value, leftMin, leftMax):
#     # Figure out how 'wide' each range is
#     leftSpan = leftMax - leftMin
#     rightSpan = 1 - 0

#     # Convert the left range into a 0-1 range (float)
#     valueScaled = float(value - leftMin) / float(leftSpan)

#     # Convert the 0-1 range into a value in the right range.
#     return 0 + (valueScaled * rightSpan)


# iter_results_file = open("iter_results_z_states.pickle", "rb")
# iter_results_Z = pickle.load(iter_results_file)
# iter_results_file.close()
# D_Z = sum(iter_results_Z) / len(iter_results_Z)
# print(len(iter_results_Z))

# iter_results_file = open("iter_results_tfidf_states.pickle", "rb")
# iter_results_tfidf = pickle.load(iter_results_file)
# iter_results_file.close()
# D_tfidf = sum(iter_results_tfidf) / len(iter_results_tfidf)
# print(len(iter_results_tfidf))

# iter_results_file = open("iter_results_jsd_states.pickle", "rb")
# iter_results_jsd = pickle.load(iter_results_file)
# iter_results_file.close()
# D_jsd = sum(iter_results_jsd) / len(iter_results_jsd)
# print(len(iter_results_jsd))


# noremd_mat = np.zeros((len(D_jsd), len(D_jsd)))


# for i in range(len(D_jsd)):
#     for j in range(len(D_jsd)):
#         D_jsd_max = D_jsd.max()
#         D_Z_max = D_Z.max()
#         D_tfidf_max = D_tfidf.max()

#         D_jsd_min = D_jsd.min()
#         D_Z_min = D_Z.min()
#         D_tfidf_min = D_tfidf.min()

#         D_Z_norm = translate(D_Z[i, j], D_Z_min, D_Z_max)
#         D_tfidf_norm = translate(D_tfidf[i, j], D_tfidf_min, D_tfidf_max)
#         D_jsd_norm = translate(D_jsd[i, j], D_jsd_min, D_jsd_max)

#         x = np.array([D_Z_norm, D_tfidf_norm, D_jsd_norm])
#         noremd_mat[i, j] = np.linalg.norm(x)


# cities_coor_file = open("cities_coor.pickle", "rb")
# cities_coor = pickle.load(cities_coor_file)
# cities_coor_file.close()


# top_cities_file = open("top_cities.pickle", "rb")
# top_cities = pickle.load(top_cities_file)
# top_cities_file.close()


# names = [cities_coor[i]['name'] for i in top_cities]

# save_names = open(
#     "names_cities.pickle", "wb")
# pickle.dump(names, save_names, -1)
# save_names.close()


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

    plt.xticks(rotation=0)

    dendo = sns.clustermap(noremd_mat, row_linkage=linkage,
                           col_linkage=linkage, cmap="Reds", cbar_pos=(0.02, 0.01, .03, 0.7480577777777779))

    leafs = dendo.dendrogram_col.reordered_ind

    print(type(dendo.ax_heatmap))
    # print(leafs)
    cluster_names = []
    for i in leafs:
        cluster_names.append(names[i])

    # print(cluster_names)

    positions = [i + 0.5 for i in range(len(names))]
    dendo.ax_heatmap.set_xticklabels(cluster_names)

    dendo.ax_heatmap.xaxis.set_ticks(positions)
    dendo.ax_heatmap.xaxis.set_ticklabels(
        cluster_names, rotation=45, fontsize=10)

    dendo.ax_heatmap.yaxis.set_ticks(positions)
    dendo.ax_heatmap.yaxis.set_ticklabels(
        cluster_names, rotation=45, fontsize=10)

    dendo.ax_heatmap.xaxis.set_ticks_position('top')
    dendo.ax_heatmap.yaxis.set_ticks_position('left')

    # hide lables

    # dendo.ax_heatmap.axis('off')

    dendo.ax_row_dendrogram.set_visible(False)
    # dendo.ax_col_dendrogram.set_visible(False)

    ll, bb, ww, hh = dendo.ax_heatmap.get_position().bounds

    print(ll, bb, ww, hh)
    dendo.ax_heatmap.set_position([ll, bb - 0.04 * hh, ww, hh])
    dendo.ax_heatmap.set_title(gran + ", method: " + method +
                               ", sorted by geo-dist: " + str(geo_sort))
    # plt.title(gran + "Method: " + method +
    #           " Sorted by geo-dist: " + str(geo_sort), fontsize=7)

    plt.show()

    return noremd_mat, leafs, cluster_names


noremd_mat, leafs, cluster_names = show_plt(
    'cities', 'average', geo_sort=False)



################################
##########################

# def show_mat(mat, names):

#     fig = plt.figure()
#     ax = fig.add_subplot()
#     cax = ax.matshow(mat, cmap='Reds')
#     fig.colorbar(cax)
#     ticks = np.arange(0, len(names), 1)

#     ax.set_xticks(ticks,)
#     ax.set_yticks(ticks)
#     ax.set_xticklabels(names, size=7)
#     ax.set_yticklabels(names, size=7)
#     # plt.axis('off')

#     plt.show()


# highlight_mat = np.empty(noremd_mat.shape)
# # taget_names = ['OH', 'UT', 'IN', 'OK', 'KY', 'ND', 'WV']
# taget_names=['Lewisville, TX','Santa Clarita, CA','Spokane, WA','Riverview, FL']
# target_ind = [i for i in range(len(cluster_names))
#                if cluster_names[i] in taget_names]


# for i in range(len(leafs)):
#     for j in range(len(leafs)):

#         if i in target_ind:
#             highlight_mat[i][j] = 0
#         elif j in target_ind:
#             highlight_mat[i][j] = 0
#         else:
#             highlight_mat[i][j] = noremd_mat[leafs[i]][leafs[j]]


# show_mat(highlight_mat, cluster_names)
# print(len(taget_names))
# print(target_ind)
# print(len(target_ind))
