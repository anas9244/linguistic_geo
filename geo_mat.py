import numpy as np
import matplotlib.pyplot as plt
import pickle
from operator import itemgetter
import math
import tweepy
# from shapely.geometry import MultiPoint, Point, Polygon
# import shapefile
# return a polygon for each state in a dictionary

from geopy.distance import geodesic


# def get_center(result):
#     south = result[0][1]
#     north = result[1][1]
#     west = result[0][0]
#     east = result[2][0]

#     location = [south,north,west,east]
#     centerx, centery = (np.average(location[:2]), np.average(location[2:]))
#     coords = [centerx, centery]
#     return coords


# tweets_dict_file = open("city_tweets_dict.pickle", "rb")
# tweets_dict = pickle.load(tweets_dict_file)
# tweets_dict_file.close()



city_coors_file = open("places_coor.pickle", "rb")
city_coors = pickle.load(city_coors_file)
city_coors_file.close()


# # # def get_dist(place01,place02):
# # #     coor1=API.geo_id(place01).centroid
# # #     coor2=API.geo_id(place02).centroid




# # #     return geodesic(coor1, coor2).kilometers



def get_geo_delta(target):

    deltas = {}
    for p in city_coors:

        deltas[p] = geodesic(target, city_coors[p]).kilometers

    return deltas



result_mat = np.zeros((len(city_coors), len(city_coors)))

for index, p in enumerate(city_coors):

    delats = get_geo_delta(city_coors[p])
    values = [value for value in delats.values()]
    result_mat[index] = values



save_geo_mat = open("geo_mat.pickle", "wb")
pickle.dump(result_mat, save_geo_mat, -1)
save_geo_mat.close()