import pickle
from sklearn_extra.cluster import KMedoids
import json
import scipy.spatial as sp
from scipy.cluster.hierarchy import fcluster, linkage
import matplotlib.pyplot as plt
import numpy as np
import os


def _plot_clusters_freq(clustering):

    corr = 0 if 0 in list(clustering) else 1

    for i in range(corr, len(set(clustering)) + corr):
        print(i, list(clustering).count(i))

    x = np.arange(corr, len(set(clustering)) + corr)
    values = []

    for i in range(corr, len(set(clustering)) + corr):
        values.append(list(clustering).count(i))

    plt.bar(x, values)
    # plt.title(method)
    plt.xlabel("cluster_lables")
    plt.ylabel("cities")
    plt.show()


#######################################
def clustering(gran, metric, n_clusters, algo, method):
    gran_path = "data/" + gran
    dist_mat_file = open(gran_path + "/dist_mats/" +
                         metric + "_dist_mat.pickle", "rb")
    dist_mat = pickle.load(dist_mat_file)

    subset_names_file = open(gran_path + "/labels.pickle", "rb")
    subset_names = pickle.load(subset_names_file)

    if algo == 'hrchy':
        linkage_result = linkage(
            sp.distance.squareform(dist_mat), method=method)
        clustering = fcluster(
            linkage_result, t=n_clusters, criterion='maxclust')

    elif algo == 'kmed':
        clustering = KMedoids(
            n_clusters=n_clusters, metric='precomputed').fit_predict(dist_mat)

    _plot_clusters_freq(clustering)

    clust_path = gran_path + "/clustering_results/"
    if not os.path.exists(clust_path):
        os.mkdir(clust_path)

    result_folder = "KMedoids" if algo == 'kmed' else method if algo == 'hrchy' else None
    result_path = clust_path + result_folder + "/"
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    file_path = result_path + str(n_clusters) + "_cluster_" + gran + ".json"

    file_out = open(file_path, 'w', encoding="utf-8")

    if gran == 'cities':
        cities_ids_file = open(gran_path + "/city_ids.pickle", "rb")
        cities_ids = pickle.load(cities_ids_file)
        for index, c in enumerate(clustering):
            record = {'city_id': cities_ids[index],
                      "city_name": subset_names[index], 'cluster': int(c)}
            json.dump(record, file_out, ensure_ascii=False)
            file_out.write("\n")

        #file_path = os.path.abspath(output_path)

    elif gran == 'states':
        for index, c in enumerate(clustering):
            record = {"state_code": subset_names[index], 'cluster': int(c)}
            json.dump(record, file_out, ensure_ascii=False)
            file_out.write("\n")

    print("Clustering stored in ", os.path.abspath(file_path))


clustering(gran='cities', metric='norm',
           n_clusters=6, algo='kmed', method="median")
