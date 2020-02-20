
import matplotlib.pyplot as plt
import pickle
from operator import itemgetter
import math

import numpy as np

new_tweets_dict_file = open("normed_tweets.pickle", "rb")
new_tweets_dict = pickle.load(new_tweets_dict_file)
new_tweets_dict_file.close()


iter_results_file = open("iter_results_merged_new.pickle", "rb")
iter_results_Z = pickle.load(iter_results_file)
iter_results_file.close()
D_Z = sum(iter_results_Z) / len(iter_results_Z)
print(len(iter_results_Z))
iter_results_file = open("iter_results_tfidf_new.pickle", "rb")
iter_results_tfidf = pickle.load(iter_results_file)
iter_results_file.close()
D_tfidf = sum(iter_results_tfidf) / len(iter_results_tfidf)
print(len(iter_results_tfidf))

iter_results_file = open("iter_results_jsd_merged_new.pickle", "rb")
iter_results_jsd = pickle.load(iter_results_file)
iter_results_file.close()
D_jsd = sum(iter_results_jsd) / len(iter_results_jsd)
print(len(iter_results_jsd))

noremd_mat = np.zeros((len(D_jsd), len(D_jsd)))


for i in range(len(D_jsd)):
    for j in range(len(D_jsd)):

        x = np.array([D_Z[i, j], D_tfidf[i, j], D_jsd[i, j]])
        noremd_mat[i, j] = np.linalg.norm(x)


names = [state for state in new_tweets_dict]


print(len(noremd_mat))

fig = plt.figure()
ax = fig.add_subplot()
cax = ax.matshow(noremd_mat, cmap='jet')
fig.colorbar(cax)
ticks = np.arange(0, len(new_tweets_dict), 1)
plt.title("The norm of the distance matrix of Z-socres, TF-IDF and JSD, num of states: " +
          str(len(new_tweets_dict)))
ax.set_xticks(ticks,)
ax.set_yticks(ticks)
ax.set_xticklabels(names, size=7)
ax.set_yticklabels(names, size=7)


plt.show()

# import numpy as np
# x = np.array([0, 4, -1])
# print(np.linalg.norm(x))
