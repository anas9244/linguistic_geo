from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import numpy as np
import pickle
import plotly.graph_objects as go

from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as sch
import os
from scipy.cluster.hierarchy import ward, dendrogram


# def prepend(list, str):
#     # Using format()
#     str += '{0}'
#     list = [str.format(i) for i in list]
#     return(list)


# def get_files():
#     fileDir = os.path.dirname(os.path.abspath(__file__))
#     path_dir = fileDir + "/merged_states/"
#     files = os.listdir(path=path_dir)
#     files_paths = prepend(files, path_dir)
#     files_paths.sort(key=os.path.basename)
#     return(files_paths)


# names = [os.path.basename(
#     file).replace(".json", "") for file in get_files()]


clusters_names = {'AK': 0, 'AL': 1, 'AR': 6, 'AZ': 3, 'CA': 3, 'CO': 2, 'CT': 4,
                  'DC': 4, 'DE': 4, 'FL': 1, 'GA': 1, 'IA': 5, 'ID': 0, 'IL': 5, 'IN': 4, 'KS': 2, 'KY': 1, 'LA': 1,
                  'MA': 4, 'MD': 4, 'ME': 4, 'MI': 4, 'MN': 5, 'MO': 2, 'MS': 1, 'MT': 0, 'NC': 1, 'ND': 5, 'NE': 5,
                  'NH': 4, 'NJ': 4, 'NM': 6, 'NV': 3, 'NY': 4, 'OH': 4, 'OK': 6, 'OR': 0, 'PA': 4, 'RI': 4, 'SC': 1,
                  'SD': 5, 'TN': 1, 'TX': 6, 'UT': 2, 'VA': 1, 'VT': 4, 'WA': 0, 'WI': 5, 'WV': 4, 'WY': 2}

names = [n for n in clusters_names.keys()]


def cluster(n, measure):

    mat = ""

    if measure == "z":
        mat = "iter_results_merged_new"
    if measure == 'tfidf':
        mat = "iter_results_tfidf_new"
    if measure == 'jsd':
        mat = "iter_results_jsd_merged_new"

    result_mat_file = open(mat + '.pickle', "rb")
    result_mat = pickle.load(result_mat_file)
    result_mat_file.close()

    average_mat = sum(result_mat) / len(result_mat)

    clustering = KMedoids(
        n_clusters=n, metric='precomputed').fit_predict(average_mat)
    print(clustering)

    values = list(clustering)

    fig = go.Figure(data=go.Choropleth(
        locations=list(names),  # Spatial coordinates
        z=list(values),  # Data to be color-coded
        locationmode='USA-states',  # set of locations match entries in `locations`
        colorscale='matter',
        colorbar_title="Cluster",
        reversescale=True,

    ))

    fig.update_layout(
        title_text='KMedoids-CLustering, ' +
        measure + ' , clusters: ' + str(n),
        geo_scope='usa',  # limite map scope to USA
    )

    fig.show()


def get_dendo(measure, method):

    mat = ""

    if measure == "z":
        mat = "iter_results_merged"
    if measure == 'tfidf':
        mat = "iter_results_tfidf_merged_manhat"
    if measure == 'jsd':
        mat = "merged_data_pickles/iter_results_kld_merged_rand"

    result_mat_file = open(mat + '.pickle', "rb")
    result_mat = pickle.load(result_mat_file)
    result_mat_file.close()

    average_mat = sum(result_mat) / len(result_mat)
    # else:
    #     mat = tf_idf_dist

    #keys = list(states_zscores.keys())

    dendrogram(sch.linkage(
        average_mat, method=method), labels=names, leaf_font_size=10)

    plt.xticks(rotation=0)
    plt.title('Hierarchical Clustering Dendrogram based on ' +
              measure + '.   linkage: ' + method + ')')

    plt.show()


# z
# tfidf
# jsd
cluster(7, 'z')

#get_dendo('z', 'average')


# dendrogram=sch.dendrogram(sch.linkage(average_mat,method='complete',optimal_ordering=False),labels=keys,leaf_font_size=10)


# print (dendrogram)


# #print (clustering.shape)
# #print (clustering)


# plt.xticks(rotation=0)
# plt.title('Hierarchical Clustering Dendrogram')

# plt.show()
