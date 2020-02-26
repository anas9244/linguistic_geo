
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
sns.set(font="monospace")
import scipy.spatial as sp
import scipy.cluster.hierarchy as hc

import numpy as np


new_tweets_dict_file = open("normed_tweets.pickle", "rb")
new_tweets_dict = pickle.load(new_tweets_dict_file)
new_tweets_dict_file.close()

names = [i for i in new_tweets_dict]


iter_results_file = open("iter_results_merged_new.pickle", "rb")
iter_results_Z = pickle.load(iter_results_file)
iter_results_file.close()
D_Z = sum(iter_results_Z) / len(iter_results_Z)
print(len(iter_results_Z))
iter_results_file = open("iter_results_tfidf_new.pickle", "rb")
iter_results_tfidf = pickle.load(iter_results_file)
iter_results_file.close()
D_tfidf = sum(iter_results_tfidf) / len(iter_results_tfidf)
print(len(iter_results_tfidf))

iter_results_file = open("iter_results_jsd_merged_new.pickle", "rb")
iter_results_jsd = pickle.load(iter_results_file)
iter_results_file.close()
D_jsd = sum(iter_results_jsd) / len(iter_results_jsd)
print(len(iter_results_jsd))

noremd_mat = np.zeros((len(D_jsd), len(D_jsd)))


for i in range(len(D_jsd)):
    for j in range(len(D_jsd)):

        x = np.array([D_Z[i, j], D_tfidf[i, j], D_jsd[i, j]])
        noremd_mat[i, j] = np.linalg.norm(x)


print(len(noremd_mat))

linkage = hc.linkage(sp.distance.squareform(noremd_mat), method='complete')

plt.xticks(rotation=0)

dendo = sns.clustermap(noremd_mat, row_linkage=linkage,
                       col_linkage=linkage, cmap="jet", cbar_pos=(0.09, 0.01, .03, 0.7480577777777779))

leafs = dendo.dendrogram_col.reordered_ind
cluster_names = []
for i in leafs:
    cluster_names.append(names[i])

positions = [i + 0.5 for i in range(len(names))]
dendo.ax_heatmap.set_xticklabels(cluster_names)

dendo.ax_heatmap.xaxis.set_ticks(positions)
dendo.ax_heatmap.xaxis.set_ticklabels(cluster_names)

dendo.ax_heatmap.yaxis.set_ticks(positions)
dendo.ax_heatmap.yaxis.set_ticklabels(cluster_names)


dendo.ax_heatmap.xaxis.set_ticks_position('top')
dendo.ax_heatmap.yaxis.set_ticks_position('left')

# dendo.ax_heatmap.axis('off')


dendo.ax_row_dendrogram.set_visible(False)

ll, bb, ww, hh = dendo.ax_heatmap.get_position().bounds

print(ll, bb, ww, hh)
dendo.ax_heatmap.set_position([ll, bb - 0.04 * hh, ww, hh])


plt.show()
