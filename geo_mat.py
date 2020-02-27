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

from shapely.geometry import MultiPoint, Point, Polygon
import shapefile


# def get_center(result):
#     south = result[0][1]
#     north = result[1][1]
#     west = result[0][0]
#     east = result[2][0]

#     location = [south,north,west,east]
#     centerx, centery = (np.average(location[:2]), np.average(location[2:]))
#     coords = [centerx, centery]
#     return coords


names_states_file = open("names_states.pickle", "rb")
names_states = pickle.load(names_states_file)
names_states_file.close()



# city_coors_file = open("places_coor.pickle", "rb")
# city_coors = pickle.load(city_coors_file)
# city_coors_file.close()


# # # def get_dist(place01,place02):
# # #     coor1=API.geo_id(place01).centroid
# # #     coor2=API.geo_id(place02).centroid




# # #     return geodesic(coor1, coor2).kilometers


def get_us_border_polygon():

    sf = shapefile.Reader("data/cb_2015_us_state_20m.shp")
    shapes = sf.shapes()
    # shapes[i].points

    records = sf.records()
    state_polygons = {}
    for i, record in enumerate(records):
        state = record[4]
        points = shapes[i].points
        poly = Polygon(points)
        state_polygons[state] = poly

    return state_polygons


state_polygons = get_us_border_polygon()


def get_dis_deltas(target):
    deltas = {}

    for s in names_states:

        point01 = state_polygons[s].centroid.coords[0]
        point02 = state_polygons[target].centroid.coords[0]

        point01 = (point01[1], point01[0])
        point02 = (point02[1], point02[0])

        deltas[s] = geodesic(point01, point02).kilometers
    return deltas

result_mat = np.zeros((len(names_states), len(names_states)))



for index, s in enumerate(names_states):
    delats = get_dis_deltas(s)
    values = [value for value in delats.values()]
    result_mat[index] = values


# def get_geo_delta(target):

#     deltas = {}
#     for p in city_coors:

#         deltas[p] = geodesic(target, city_coors[p]).kilometers

#     return deltas



# result_mat = np.zeros((len(city_coors), len(city_coors)))

# for index, p in enumerate(city_coors):

#     delats = get_geo_delta(city_coors[p])
#     values = [value for value in delats.values()]
#     result_mat[index] = values



save_geo_mat = open("geo_mat_states.pickle", "wb")
pickle.dump(result_mat, save_geo_mat, -1)
save_geo_mat.close()