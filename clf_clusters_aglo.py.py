import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pickle
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
from operator import itemgetter
from collections import OrderedDict
import string
from nltk.stem import PorterStemmer
import re
import json
from time import time
from statistics import mean

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

def clusters_tweets(tweets_dict, dist_mat, n):
    cluster_labels = []
    tweets = []

    clustering = AgglomerativeClustering(
        n_clusters=n, affinity='precomputed', linkage='average').fit_predict(dist_mat)

    clusters_names = {}
    for index, state in enumerate(names):
        clusters_names[state] = clustering[index]

    for state in tweets_dict:
        for tweet in tweets_dict[state]:
            cluster_labels.append(clusters_names[state])
            tweets.append(tweet)

    cluster_sizes = {}
    clutser_counts = {}
    for c in cluster_labels:
        clutser_counts[c] = 0
        if c not in cluster_sizes:
            cluster_sizes[c] = 1
        else:
            cluster_sizes[c] += 1
    min_cluster = min([v for v in cluster_sizes.values()])
    max_cluster = min([v for v in cluster_sizes.values()])

    sampled_cluster_labels = []
    sampled_tweets = []

    for index, c in enumerate(cluster_labels):
        if clutser_counts[c] < min_cluster:
            clutser_counts[c] += 1
            sampled_cluster_labels.append(c)
            sampled_tweets.append(tweets[index])

    return sampled_cluster_labels, sampled_tweets, min_cluster, max_cluster


file = open("clf_stats_Z_aglo_average.json", "a")

for i in range(3, 15):
    file = open("clf_stats_Z_aglo_average.json", "a")
    start = time()
    print(i)

    cluster_labels, tweets, min_cluster, max_cluster = clusters_tweets(
        tweets_dict, average_mat, i)
    print(min_cluster)
    print(len(tweets))

    Tfidf_vect = TfidfVectorizer()
    Train_X_Tfidf = Tfidf_vect.fit_transform(tweets)
    print(time() - start, "finished tfidf transform")

    MNB = naive_bayes.MultinomialNB()
    acc_NB = mean(cross_val_score(MNB, Train_X_Tfidf, cluster_labels, cv=5))
    print(time() - start, "Naive Bayes Accuracy Score -> ",
          acc_NB)

    SVM = svm.LinearSVC()
    acc_SVM = mean(cross_val_score(SVM, Train_X_Tfidf, cluster_labels, cv=5))
    print(time() - start, "LinearSVC Accuracy Score -> ",
          acc_SVM)

    ridge_model = RidgeClassifier()
    acc_ridge = mean(cross_val_score(
        ridge_model, Train_X_Tfidf, cluster_labels, cv=5))
    print(time() - start, "RidgeClassifier Accuracy Score -> ",
          acc_ridge)

    rfc_model = RandomForestClassifier()
    acc_rfc = mean(cross_val_score(
        rfc_model, Train_X_Tfidf, cluster_labels, cv=5))
    print(time() - start, "RandomForestClassifier Accuracy Score -> ",
          acc_rfc)

    record = {"n_clusters": i, "min_cluster": min_cluster, 'max_cluster': max_cluster,
              "acc_NB": acc_NB, 'acc_SVM': acc_SVM, 'acc_ridge': acc_ridge, 'acc_rfc': acc_rfc}
    json.dump(record, file)

    file.write("\n")
    file.close()
