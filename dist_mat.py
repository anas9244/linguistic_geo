import pickle
import os
import json
from operator import itemgetter
from collections import OrderedDict
import string
from nltk.stem import PorterStemmer
import re
import math
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
import random
ps = PorterStemmer()
punc = set(string.punctuation)


tweets_dict_file = open("normed_tweets.pickle", "rb")
tweets_dict = pickle.load(tweets_dict_file)
tweets_dict_file.close()


min_state = min([len(tweets_dict[state])
                 for state in tweets_dict])
max_state = max([len(tweets_dict[state])
                 for state in tweets_dict])
iters = int(round(max_state / min_state, 0))
print(min_state)
print(max_state)
print(iters)


def get_word_vec(tweets_list):
    word_vec = {}
    for tweet in tweets_list:
        tokens = tweet.split()

        for word in tokens:

            if word not in word_vec:
                word_vec[word] = 1
            else:
                word_vec[word] += 1
    return word_vec


def get_delta(target, states_zscores, word_list):
    deltas = {}
    for state in states_zscores:
        delta = 0
        for word in word_list:
            try:
                delta += math.fabs((states_zscores[target]
                                    [word] - states_zscores[state][word]))
            except:
                pass
                #print(state, word)
        # print(len(word_list))
        delta /= len(word_list)
        deltas[state] = delta

    return deltas


def get_average(deltas):

    values = [value for value in deltas.values()]
    average = sum(values) / len(values)

    return average


iter_results = []


for i in range(iters):
    start_time = time.time()

    states_words = {}
    states_features = {}
    word_set = set()
    for state in tweets_dict:

        start_index = random.randint(
            0, len(tweets_dict[state]) - min_state)
        end_index = start_index + min_state
        sample = tweets_dict[state][start_index:end_index]

        word_vec = get_word_vec(sample)
        states_words[state] = word_vec

    print("word_vec")
    print("--- %s seconds ---" % (time.time() - start_time))
    for index, state in enumerate(states_words):

        if index == 0:
            for word in states_words[state]:
                word_set.add(word)
        else:
            set2 = set()
            for word in states_words[state]:
                set2.add(word)
            word_set = word_set.intersection(set2)
    print("word_set: ", len(word_set))



    states_features = {}
    for state in states_words:
        overall = sum(states_words[state].values())
        for word in states_words[state]:
            states_words[state][word] /= overall

    for word in list(word_set):
        states_features[word] = {}
        word_mean = 0
        for state in states_words:

            word_mean += states_words[state][word]

        word_mean /= len(states_words)
        states_features[word]["mean"] = word_mean

        word_stdev = 0
        for state in states_words:

            diff = states_words[state][word] - \
                states_features[word]["mean"]

            word_stdev += diff * diff

        word_stdev /= (len(states_words) - 1)
        word_stdev = math.sqrt(word_stdev)
        states_features[word]["stdev"] = word_stdev

    states_zscores = {}
    for state in states_words:
        states_zscores[state] = {}

        for word in list(word_set)[:]:
            if word in states_words[state]:
                word_state_freq = states_words[state][word]
                word_mean = states_features[word]["mean"]
                word_stdev = states_features[word]["stdev"]
                states_zscores[state][word] = (
                    word_state_freq - word_mean) / word_stdev

    result_mat = np.zeros((len(states_zscores), len(states_zscores)))

    for index, state in enumerate(states_zscores):
        delats = get_delta(state, states_zscores, list(word_set))
        values = [value for value in delats.values()]

        result_mat[index] = values

    iter_results.append(result_mat)
    print(i)


save_iter_results = open("iter_results_merged_new.pickle", "wb")
pickle.dump(iter_results, save_iter_results, -1)
save_iter_results.close()


# save_state_words = open(
#     "merged_data_pickles/state_words_iters_merged_rand.pickle", "wb")
# pickle.dump(state_words_iters, save_state_words, -1)
# save_state_words.close()


# save_word_set_iters = open(
#     "merged_data_pickles/word_set_iters_merged_rand.pickle", "wb")
# pickle.dump(word_set_iters, save_word_set_iters, -1)
# save_word_set_iters.close()
