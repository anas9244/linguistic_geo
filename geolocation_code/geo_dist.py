import numpy as np
import matplotlib.pyplot as plt
import pickle
from operator import itemgetter
import math
import tweepy

from geopy.distance import geodesic

from shapely.geometry import MultiPoint, Point, Polygon
import shapefile
import os


def _get_mat_size(gran):

    dataset_file = open("data/" + gran + "/dataset.pickle", "rb")
    dataset = pickle.load(dataset_file)
    dataset_file.close()

    return len(dataset)


def _get_labels(gran):
    file_path = "data/" + gran + "labels.pickle"

    labels_file = open("data/" + gran + "/dataset.pickle", "rb")
    labels = pickle.load(labels_file)
    labels_file.close()

    return labels


def _get_state_border_polygon():

    sf = shapefile.Reader("data/states/geo-data/cb_2015_us_state_20m.shp")
    shapes = sf.shapes()

    records = sf.records()
    state_polygons = {}
    for i, record in enumerate(records):
        state = record[4]
        points = shapes[i].points
        poly = Polygon(points)
        state_polygons[state] = poly

    return state_polygons


def _get_state_deltas(target, labels):
    state_polygons = _get_state_border_polygon()
    deltas = {}

    for s in labels:

        point01 = state_polygons[s].centroid.coords[0]
        point02 = state_polygons[target].centroid.coords[0]

        point01 = (point01[1], point01[0])
        point02 = (point02[1], point02[0])

        deltas[s] = geodesic(point01, point02).kilometers
    return deltas


def _get_city_delta(target):
    # city coors is the center coordinates of cites
    deltas = {}
    for p in city_coors:

        deltas[p] = geodesic(target, city_coors[p]).kilometers

    return deltas


def get_geo_mat(gran):
    if gran in {"states", "cities"}:
        if not os.path.exists('data/' + gran):
            print(
                "No generated dataset for " + gran + " found. Please generate dataset with build_data.py")
        elif len(os.listdir('data/' + gran)) == 0:
            print(
                "No generated dataset for " + gran + " found. Please generate dataset with build_data.py")
        else:
            mat_size = _get_mat_size(gran)
            result_mat = np.zeros((len(mat_size), len(mat_size)))

            if gran == 'states':

                for index, s in enumerate(mat_size):
                    delats = _get_state_deltas(s, mat_size)
                    values = [value for value in delats.values()]
                    result_mat[index] = values

            save_geo_mat = open(
                "data/" + gran + "/geo_mat_" + gran + ".pickle", "wb")
            pickle.dump(result_mat, save_geo_mat, -1)

    else:
        print("'" + gran + "'" +
              " is invalid. Possible values are 'states' or 'cities'")
