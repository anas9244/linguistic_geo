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
import smtplib


tweets_dict_file = open("normed_tweets.pickle", "rb")
tweets_dict = pickle.load(tweets_dict_file)
tweets_dict_file.close()

tweets_dict_top = {}

for city in tweets_dict:
    if len(tweets_dict[city]) > 5000:
        tweets_dict_top[city] = tweets_dict[city]


tweets_dict = None
min_state = min([len(tweets_dict_top[state])
                 for state in tweets_dict_top])
max_state = max([len(tweets_dict_top[state])
                 for state in tweets_dict_top])
iters = int(round(max_state / min_state, 0))
print(min_state)
print(max_state)
print(iters)


iter_results = []


for i in range(iters):
    print(i)
   # if i > 20:
      #  break
    start_time = time.time()

    corpus = []
    for state in tweets_dict_top:
        start_index = random.randint(
            0, len(tweets_dict_top[state]) - min_state)
        end_index = start_index + min_state

        sample = tweets_dict_top[state][start_index:end_index]
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


save_iter_results = open("iter_results_tfidf_states02.pickle", "wb")
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
#