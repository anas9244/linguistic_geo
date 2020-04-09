import pickle
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
import json

import scipy.spatial as sp

from scipy.cluster.hierarchy import fcluster, linkage, correspond
import matplotlib.pyplot as plt

import numpy as np

# city_tweets_file = open("city_tweets_dict.pickle", "rb")
# city_tweets = pickle.load(city_tweets_file)
# city_tweets_file.close()


top_cities_file = open("top_cities.pickle", "rb")
top_cities = pickle.load(top_cities_file)
top_cities_file.close()


names_cities_file = open("names_cities.pickle", "rb")
names_citie = pickle.load(names_cities_file)
names_cities_file.close()


dist_cities_file = open("noremd_mat_cities.pickle", "rb")
dist_cities_mat = pickle.load(dist_cities_file)
dist_cities_file.close()

taget_names = ['Lewisville, TX', 'Santa Clarita, CA',
               'Spokane, WA', 'Riverview, FL']

# city_size = {}
# for city in taget_names:
#     cityname_id = names_citie.index(city)
#     city_id = top_cities[cityname_id]
#     city_size[city] = len(city_tweets[city_id])


# print(city_size)

# print(len(names_citie))

#######################################
def clustering(n_clusters, method):

    global linkage
    linkage = linkage(sp.distance.squareform(
        dist_cities_mat), method=method)

    clustering = fcluster(linkage, t=n_clusters, criterion='maxclust')

    #######################################

    # clustering = KMedoids(
    #     n_clusters=7, metric='precomputed').fit_predict(dist_cities_mat)

    # clustering = AgglomerativeClustering(
    #     n_clusters, affinity='precomputed', linkage=method).fit(dist_cities_mat)
    #print(clustering)
    print(len(set(clustering)))
    for i in range(1, len(set(clustering)) + 1):
        print(i, list(clustering).count(i))

    x = np.arange(1, len(set(clustering)) + 1)
    print(x)
    values = []

    for i in range(1, len(set(clustering)) + 1):
        values.append(list(clustering).count(i))

    plt.bar(x, values)
    plt.title(method)
    plt.xlabel("cluster_lables")
    plt.ylabel("cities")
    plt.show()

    file_out = open(method + "/" + str(n_clusters) +
                    "_cluster_cities.json", 'w', encoding="utf-8")

    for index, c in enumerate(clustering):
        record = {'city_id': top_cities[index],
                  "city_name": names_citie[index], 'cluster': int(c)}
        json.dump(record, file_out, ensure_ascii=False)
        file_out.write("\n")


clustering(7, "complete")
