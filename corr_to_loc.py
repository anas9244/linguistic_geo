import numpy as np
import matplotlib.pyplot as plt
import pickle
from operator import itemgetter
import math

from shapely.geometry import MultiPoint, Point, Polygon
import shapefile
# return a polygon for each state in a dictionary

from geopy.distance import geodesic

import seaborn as sns
from scipy.stats import pearsonr
sns.set(color_codes=True)


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


def get_dis(state01, state02):
    point01 = state_polygons[state01].centroid.coords[0]
    point02 = state_polygons[state02].centroid.coords[0]

    point01 = (point01[1], point01[0])
    point02 = (point02[1], point02[0])
    return geodesic(point01, point02).kilometers


iter_results_file = open(
    "noremd_mat_states.pickle", "rb")
average_mat = pickle.load(iter_results_file)
iter_results_file.close()


noremd_mat_file = open("names_states.pickle", "rb")
names = pickle.load(noremd_mat_file)
noremd_mat_file.close()


state_deltas = {}
state_deltas['dis'] = []
state_deltas['km'] = []
state_deltas['names'] = []

for index, x in enumerate(average_mat):

    values = [val for val in x if val > 0]
    min_dist = min(values)
    print(min_dist)
    min_dist_i = np.argmin(values)
    state01 = index
    state02 = min_dist_i

    state_deltas['dis'].append(min_dist)
    state_deltas['km'].append(get_dis(names[state01], names[state02]))
    state_deltas['names'].append(names[state01] + '_' + names[state02])

    # for v_index, v in enumerate(values):
    #     state_deltas['dis'].append(v)
    #     state_deltas['km'].append(get_dis(names[state01], names[v_index]))
    #     state_deltas['names'].append(names[state01] + '_' + names[v_index])

dis_values01 = [value for value in state_deltas['dis']]

km_values02 = [value for value in state_deltas['km']]


corr, _ = pearsonr(dis_values01, km_values02)
print(corr)
plt.title("Correlation between Z score distance and georaphic distance of the 51 state pairs with the lowest Z distance. Pearson coefficient= " + str(corr))
plt.xlabel("Z distance")
plt.ylabel("KM")
# for i, txt in enumerate(state_deltas['names']):
#     plt.annotate(txt, (dis_values01[i], km_values02[i]))
sns.regplot(x=dis_values01, y=km_values02)
plt.axis('off')
plt.show()
