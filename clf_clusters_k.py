import pandas as pd
import numpy as np


from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pickle
from sklearn_extra.cluster import KMedoids
from operator import itemgetter
from collections import OrderedDict
#!/usr/bin/python3
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
from statistics import mean

import string
from nltk.stem import PorterStemmer
import re
import json
from time import time

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.combine import SMOTEENN
from gensim.sklearn_api import TfIdfTransformer
from collections import Counter


ps = PorterStemmer()
punc = set(string.punctuation)


result_mat_file = open(
    "iter_results_merged_new.pickle", "rb")
dist_mat = pickle.load(result_mat_file)
result_mat_file.close()

average_mat = sum(dist_mat) / len(dist_mat)


tweets_dict_file = open("normed_tweets.pickle", "rb")
tweets_dict = pickle.load(tweets_dict_file)
tweets_dict_file.close()

names = [state for state in tweets_dict]
tweets = []

# for state in

# normed_tweets = {}
# for state in tweets_dict:
#     print(state)
#     normed_tweets[state] = []
#     for tweet in tweets_dict[state]:
#         tweet_json = json.loads(tweet)
#         if type(tweet_json) is not dict:

#             tokens = tweet_json.split()
#             normed = []
#             for t in tokens:
#                 normed.append(norm(t))
#             normed_text = " ".join(normed)
#             normed_tweets[state].append(normed_text)


# save_normed_tweets = open(
#     "normed_tweets.pickle", "wb")
# pickle.dump(normed_tweets, save_normed_tweets, -1)
# save_normed_tweets.close()


Train_X_Tfidf_file = open("Train_X_Tfidf.pickle", "rb")
Train_X_Tfidf = pickle.load(Train_X_Tfidf_file)
Train_X_Tfidf_file.close()


def clusters_tweets(n):
    cluster_labels = []

    clustering = KMedoids(
        n_clusters=n, metric='precomputed').fit_predict(average_mat)

    clusters_names = {}
    for index, state in enumerate(names):
        clusters_names[state] = clustering[index]

    for state in tweets_dict:
        for tweet in tweets_dict[state]:
            # if not tweet.startswith("[mention]"):

            cluster_labels.append(clusters_names[state])

    # cluster_sizes = {}
    # clutser_counts = {}
    # for c in cluster_labels:
    #     clutser_counts[c] = 0
    #     if c not in cluster_sizes:
    #         cluster_sizes[c] = 1
    #     else:
    #         cluster_sizes[c] += 1
    # min_cluster = min([v for v in cluster_sizes.values()])
    # max_cluster = min([v for v in cluster_sizes.values()])

    min_cluster = min(Counter(cluster_labels).values())
    max_cluster = max(Counter(cluster_labels).values())

    # sampled_cluster_labels = []
    # sampled_tweets = []

    # for index, c in enumerate(cluster_labels):
    #     if clutser_counts[c] < min_cluster:
    #         clutser_counts[c] += 1
    #         sampled_cluster_labels.append(c)
    #         sampled_tweets.append(tweets[index])

    return cluster_labels, min_cluster, max_cluster


#file = open("clf_stats_Z_KMedoids.json", "a")


#for i in range(7, 8):
file = open("clf_stats_Z_KMedoids.json", "a")
start = time()

cc = RandomOverSampler()
cluster_labels, min_cluster, max_cluster = clusters_tweets(7)
#Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(tweets,Corpus['label'],test_size=0.3)

print(min_cluster)
print(Counter(cluster_labels))

X_res, y_res = cc.fit_resample(Train_X_Tfidf, cluster_labels)

print(Counter(y_res))
print(time() - start, "finished resampling")

MNB = naive_bayes.MultinomialNB()
acc_NB = mean(cross_val_score(MNB, X_res, y_res, cv=5))
print(time() - start, "Naive Bayes Accuracy Score -> ",
      acc_NB)

# SVM = svm.LinearSVC()
# acc_SVM = mean(cross_val_score(SVM, X_res, y_res, cv=5))
# print(time() - start, "LinearSVC Accuracy Score -> ",
#       acc_SVM)

# ridge_model = RidgeClassifier()
# acc_ridge = mean(cross_val_score(
#     ridge_model, X_res, y_res, cv=5))
# print(time() - start, "RidgeClassifier Accuracy Score -> ",
#       acc_ridge)

# rfc_model = RandomForestClassifier()
# acc_rfc = mean(cross_val_score(
#     rfc_model, X_res, y_res, cv=5))
# print(time() - start, "RandomForestClassifier Accuracy Score -> ",
#       acc_rfc)

record = {"Random_oversampling_n_clusters": 7, "cluster_tweets_num": min_cluster, 'max_cluster': max_cluster,
          "acc_NB": acc_NB}  # , 'acc_SVM': acc_SVM, 'acc_ridge': acc_ridge, 'acc_rfc': acc_rfc}
json.dump(record, file)

file.write("\n")
file.close()
