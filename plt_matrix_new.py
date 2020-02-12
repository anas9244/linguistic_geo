
import matplotlib.pyplot as plt
import pickle
from operator import itemgetter
import math

import numpy

new_tweets_dict_file = open("normed_tweets.pickle", "rb")
new_tweets_dict = pickle.load(new_tweets_dict_file)
new_tweets_dict_file.close()


iter_results_file = open(
    "iter_results_tfidf_new.pickle", "rb")
iter_results = pickle.load(iter_results_file)
iter_results_file.close()
print(len(iter_results))

# plot dist matrix

names = [state for state in new_tweets_dict]
average_mat = sum(iter_results) / len(iter_results)

print(len(average_mat))

fig = plt.figure()
ax = fig.add_subplot()
cax = ax.matshow(average_mat, cmap='jet')
fig.colorbar(cax)
ticks = numpy.arange(0, len(new_tweets_dict), 1)
plt.title("Meetup Z scores, num of states: " + str(len(new_tweets_dict)))
ax.set_xticks(ticks,)
ax.set_yticks(ticks)
ax.set_xticklabels(names, size=7)
ax.set_yticklabels(names, size=7)


plt.show()
