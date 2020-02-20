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
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances

import smtplib
from scipy.spatial import distance
from scipy.stats import entropy
import random
import smtplib

from sklearn.feature_extraction.text import TfidfVectorizer

tweets_dict_file = open("city_tweets_dict.pickle", "rb")
tweets_dict = pickle.load(tweets_dict_file)
tweets_dict_file.close()

tweets_dict_top = {}

for city in tweets_dict:
    if len(tweets_dict[city]) > 5000:
        tweets_dict_top[city] = tweets_dict[city]

min_state = min([len(tweets_dict_top[state])
                 for state in tweets_dict_top])
max_state = max([len(tweets_dict_top[state])
                 for state in tweets_dict_top])

tweets_dict = None
iters = int(round(max_state / min_state, 0))
print(min_state)
print(max_state)
print(iters)


def get_word_vec(tweets_list):
    word_vec = {}
    for tweet in tweets_list:
        tokens = tweet.split()
        # try:
        #     vec = TfidfVectorizer(ngram_range=(1, 3))
        #     vec.fit([tweet])
        #     tokens = vec.get_feature_names()
        # except:
        #     print(tweet)

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
                # print(state, word)
        # print(len(word_list))
        delta /= len(word_list)
        deltas[state] = delta

    return deltas


iter_results = []


for i in range(50):
    print(i)
    start_time = time.time()

    states_words = {}
    states_features = {}

    word_set = set()
    for state in tweets_dict_top:

        start_index = random.randint(
            0, len(tweets_dict_top[state]) - min_state)
        end_index = start_index + min_state
        sample = tweets_dict_top[state][start_index:end_index]

        word_vec = get_word_vec(sample)
        states_words[state] = word_vec

    print("word_vec")

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

    for state in states_words:
        overall = sum(states_words[state].values())
        for word in states_words[state]:
            states_words[state][word] /= overall

    state_dist = {}

    for state in states_words:
        state_dist[state] = []

    for word in word_set:
        for state in states_words:
            state_dist[state].append(states_words[state][word])

    result_mat = np.zeros(
        (len(states_words), len(states_words)))
    state_entropys = {}
    for index, state in enumerate(state_dist):
        state_klds = []
        state_entropys[state] = entropy(state_dist[state], base=2)
        for other_state in state_dist:

            state_klds.append(
                distance.jensenshannon(state_dist[state], state_dist[other_state], 2.0))

        result_mat[index] = state_klds

    iter_results.append(result_mat)

    print("--- %s seconds ---" % (time.time() - start_time))


save_iter_results = open(
    "iter_results_jsd_cities.pickle", "wb")
pickle.dump(iter_results, save_iter_results, -1)
save_iter_results.close()


def sendemail(from_addr, to_addr_list,
              subject, message,
              login, password,
              smtpserver='smtp.gmail.com:587'):
    header = 'From: %s\n' % from_addr
    header += 'To: %s\n' % ','.join(to_addr_list)
    header += 'Subject: %s\n\n' % subject
    message = header + message

    server = smtplib.SMTP(smtpserver)
    server.starttls()
    server.login(login, password)
    problems = server.sendmail(from_addr, to_addr_list, message)
    server.quit()
    return problems


sendemail(from_addr='anasnayef1@gmail.com',
          to_addr_list=['anas.alnayef@uni-weimar.de'],
          subject='tfidf_dist done',
          message='tfidf_dist done',
          login='anasnayef1@gmail.com',
          password='Yeje_9244')
