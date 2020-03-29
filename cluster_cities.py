import pickle
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
import json

import scipy.spatial as sp

from scipy.cluster.hierarchy import fcluster, linkage

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

linkage = linkage(sp.distance.squareform(
    dist_cities_mat), method='ward')

clustering = fcluster(linkage, 7, criterion='maxclust')


# clustering = KMedoids(
#     n_clusters=6, metric='precomputed').fit_predict(dist_cities_mat)

# print(list(clustering.labels_))

for i in range(7):
    print(i, list(clustering).count(i))


file_out = open("aglo_ward_7_cluster_cities.json", 'w', encoding="utf-8")


for index, c in enumerate(clustering):
    record = {  # 'city_id': top_cities[index],
        "city_name": names_citie[index], 'cluster': int(c)}
    json.dump(record, file_out, ensure_ascii=False)
    file_out.write("\n")
