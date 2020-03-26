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


# def get_us_border_polygon():

#     sf = shapefile.Reader("data/cb_2015_us_state_20m.shp")
#     shapes = sf.shapes()
#     # shapes[i].points

#     records = sf.records()
#     state_polygons = {}
#     for i, record in enumerate(records):
#         state = record[4]
#         points = shapes[i].points
#         poly = Polygon(points)
#         state_polygons[state] = poly

#     return state_polygons


# state_polygons = get_us_border_polygon()


# def get_dis(state01, state02):
#     point01 = state_polygons[state01].centroid.coords[0]
#     point02 = state_polygons[state02].centroid.coords[0]

#     point01 = (point01[1], point01[0])
#     point02 = (point02[1], point02[0])
#     return geodesic(point01, point02).kilometers


geo_mat_file = open("geo_mat_cities.pickle", "rb")
geo_mat = pickle.load(geo_mat_file)
geo_mat_file.close()


noremd_mat_file = open("noremd_mat_cities.pickle", "rb")
noremd_mat = pickle.load(noremd_mat_file)
noremd_mat_file.close()


names_file = open("names_cities.pickle", "rb")
names = pickle.load(names_file)
names_file.close()


state_deltas = {}
state_deltas['dis'] = []
state_deltas['km'] = []
state_deltas['dis_all'] = []
state_deltas['km_all'] = []

state_deltas['names'] = []

for index, x in enumerate(noremd_mat):

    values = [val for val in x if val > 0]
    min_dist = min(values)
    # print(min_dist)
    min_dist_i = np.argmin(values)
    state01 = index
    state02 = min_dist_i

    state_deltas['dis'].append(min_dist)
    state_deltas['km'].append(geo_mat[state01, state02])

    state_deltas['dis_all'] += values
    state_deltas['km_all'] += [value for value in geo_mat[index] if value > 0]

    #state_deltas['names'].append(names[state01] + '_' + names[state02])

    #state_deltas['names'].append(names[state01] + '_' + names[state02])
    # for v_index, v in enumerate(values):
    #     state_deltas['dis'].append(v)
    #     #state_deltas['km'].append(get_dis(names[state01], names[v_index]))
    #     state_deltas['names'].append(names[state01] + '_' + names[v_index])

dis_values01 = [value for value in state_deltas['dis']]

km_values02 = [value for value in state_deltas['km']]


dis_values01_all = [value for value in state_deltas['dis_all']]

km_values02_all = [value for value in state_deltas['km_all']]

# plt.scatter(averages_values01, averages_values02)

# plt.title("Correlation between KLD and TF-IDF distance averages")
# plt.xlabel("KLD")
# plt.ylabel("TF-IDF")


# plt.show()
print(len(dis_values01))
corr, _ = pearsonr(dis_values01, km_values02)
corr_all, _ = pearsonr(dis_values01_all, km_values02_all)
print(corr)
plt.title("Correlation between language distance and georaphic distance of the 457 US cities pairs with the lowest language distance \n with Pearson coefficient= " +
          str(corr) + " , and Pearson coefficient of all pairs= " + str(corr_all))
plt.xlabel("Language distance")
plt.ylabel("KM")
# for i, txt in enumerate(state_deltas['names']):
#     plt.annotate(txt, (dis_values01[i], km_values02[i]))
sns.regplot(x=dis_values01, y=km_values02)
# plt.axis('off')
plt.show()
