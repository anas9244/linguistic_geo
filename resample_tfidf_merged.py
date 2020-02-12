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
from sklearn.feature_extraction.text import TfidfVectorizer
import random

tweets_dict_file = open("normed_tweets.pickle", "rb")
tweets_dict = pickle.load(tweets_dict_file)
tweets_dict_file.close()


min_state = min([len(tweets_dict[state]) for state in tweets_dict])
max_state = max([len(tweets_dict[state]) for state in tweets_dict])
iters = int(round(max_state / min_state, 0))
print(min_state)
print(max_state)
print(iters)


iter_results = []

for i in range(iters):
    start_time = time.time()

    states_words = {}
    states_features = {}
    word_set = set()

    corpus = []
    for state in tweets_dict:
        start_index = random.randint(0, len(tweets_dict[state]) - min_state)
        end_index = start_index + min_state

        sample = tweets_dict[state][start_index:end_index]
        sub_corpus = ""
        for tweet in sample:
            sub_corpus += " " + tweet
        corpus.append(sub_corpus)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    tf_idf_dist = manhattan_distances(X)
    print(tf_idf_dist.shape)

    iter_results.append(tf_idf_dist)

    print("--- %s seconds ---" % (time.time() - start_time))
    print(i)


save_iter_results = open("iter_results_tfidf_new.pickle", "wb")
pickle.dump(iter_results, save_iter_results, -1)
save_iter_results.close()
