
import matplotlib.pyplot as plt
import pickle
from operator import itemgetter
import math

import numpy as np
import time
# new_tweets_dict_file = open("normed_tweets.pickle", "rb")
# new_tweets_dict = pickle.load(new_tweets_dict_file)
# new_tweets_dict_file.close()


def translate(value, leftMin, leftMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = 1 - 0

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return 0 + (valueScaled * rightSpan)


# names_states_file = open("names_states.pickle", "rb")
# names = pickle.load(names_states_file)
# names_states_file.close()


# noremd_mat_file = open("noremd_mat_states.pickle", "rb")
# noremd_mat = pickle.load(noremd_mat_file)
# noremd_mat_file.close()


# iter_results_file = open("iter_results_z_states.pickle", "rb")
# iter_results_Z = pickle.load(iter_results_file)
# iter_results_file.close()
# D_Z = sum(iter_results_Z) / len(iter_results_Z)
# print(len(iter_results_Z))

# iter_results_file = open("iter_results_tfidf_states.pickle", "rb")
# iter_results_tfidf = pickle.load(iter_results_file)
# iter_results_file.close()
# D_tfidf = sum(iter_results_tfidf) / len(iter_results_tfidf)
# print(len(iter_results_tfidf))

# iter_results_file = open("iter_results_jsd_states.pickle", "rb")
# iter_results_jsd = pickle.load(iter_results_file)
# iter_results_file.close()
# D_jsd = sum(iter_results_jsd) / len(iter_results_jsd)
# print(len(iter_results_jsd))


# noremd_mat = np.zeros((len(D_jsd), len(D_jsd)))

# D_jsd_max = D_jsd.max()
# D_Z_max = D_Z.max()
# D_tfidf_max = D_tfidf.max()

# D_jsd_min = D_jsd.min()
# D_Z_min = D_Z.min()
# D_tfidf_min = D_tfidf.min()
# for i in range(len(D_jsd)):
#     for j in range(len(D_jsd)):

#         D_Z_norm = translate(D_Z[i, j], D_Z_min, D_Z_max)
#         D_tfidf_norm = translate(D_tfidf[i, j], D_tfidf_min, D_tfidf_max)
#         D_jsd_norm = translate(D_jsd[i, j], D_jsd_min, D_jsd_max)

#         print(D_Z_norm, D_tfidf_norm, D_jsd_norm)

#         x = np.array([D_Z_norm, D_tfidf_norm, D_jsd_norm])
#         print(np.linalg.norm(x))

#         #noremd_mat[i, j] = np.linalg.norm(x)


########################################################

def show_mat(gran, geo=False):

    names_file = open("names_" + gran + ".pickle", "rb")
    names = pickle.load(names_file)
    names_file.close()

    if geo:
        noremd_mat_file = open("geo_mat_" + gran + ".pickle", "rb")
        noremd_mat = pickle.load(noremd_mat_file)
        noremd_mat_file.close()

    else:
        #noremd_mat_file = open(
            #"geolocation_code/dist_mats/tfidf_dist_mat.pickle", "rb")
        noremd_mat_file = open('iter_results_tfidf_cities.pickle', "rb")
        #noremd_mat_file = open("noremd_mat_" + gran + ".pickle", "rb")
        noremd_mat = pickle.load(noremd_mat_file)
        noremd_mat = sum(noremd_mat) / len(noremd_mat)
        noremd_mat_file.close()

    fig = plt.figure()
    ax = fig.add_subplot()
    cax = ax.matshow(noremd_mat, cmap='jet')
    fig.colorbar(cax)
    ticks = np.arange(0, len(names), 1)
    if geo:
        plt.title(gran + ", Geographic distance, " +
                  "num of " + gran + ": " + str(len(names)))
    else:
        plt.title(gran + ", Language distance, " +
                  "num of " + gran + ": " + str(len(names)))

    ax.set_xticks(ticks,)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names, size=7)
    ax.set_yticklabels(names, size=7)
    # plt.axis('off')

    plt.show()


show_mat('cities', False)
#>5000 : wordset 717, 63 iters, max: 317697


# tfidf >2000: 159 iters,max :
