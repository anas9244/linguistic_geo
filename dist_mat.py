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
from sklearn.feature_extraction.text import TfidfVectorizer

import smtplib
ps = PorterStemmer()


punc = set(string.punctuation)


# test = 'hi how are you man'
# vec = TfidfVectorizer(ngram_range=(1, 3))
# vec.fit_transform([test])
# tokens = vec.get_feature_names()

# for t in tokens:
#     print(t)


tweets_dict_file = open("city_tweets_dict.pickle", "rb")
tweets_dict = pickle.load(tweets_dict_file)
tweets_dict_file.close()
print(len(tweets_dict))


# tweets_dict_top = {}

# for city in tweets_dict:
#     if len(tweets_dict[city]) > 5000:
#         tweets_dict_top[city] = tweets_dict[city]

# tweets_dict = None

# min_state = min([len(tweets_dict_top[state])
#                  for state in tweets_dict_top])
# max_state = max([len(tweets_dict_top[state])
#                  for state in tweets_dict_top])
# iters = int(round(max_state / min_state, 0))
# print(min_state)
# print(max_state)
# print(iters)


# def get_word_vec(tweets_list):
#     word_vec = {}
#     for tweet in tweets_list:
#         tokens = tweet.split()
#         # try:
#         #     vec = TfidfVectorizer(ngram_range=(1, 3))
#         #     vec.fit([tweet])
#         #     tokens = vec.get_feature_names()
#         # except:
#         #     print(tweet)

#         for word in tokens:

#             if word not in word_vec:
#                 word_vec[word] = 1
#             else:
#                 word_vec[word] += 1
#     return word_vec


# def get_delta(target, states_zscores, word_list):
#     deltas = {}
#     for state in states_zscores:
#         delta = 0
#         for word in word_list:
#             try:
#                 delta += math.fabs((states_zscores[target]
#                                     [word] - states_zscores[state][word]))
#             except:
#                 pass
#                 #print(state, word)
#         # print(len(word_list))
#         delta /= len(word_list)
#         deltas[state] = delta

#     return deltas


# def get_average(deltas):

#     values = [value for value in deltas.values()]
#     average = sum(values) / len(values)

#     return average


# iter_results = []

# print("Begin sampling ############################")
# for i in range(iters):
#     print(i)
#     start_time = time.time()

#     states_words = {}
#     states_features = {}
#     word_set = set()
#     for state in tweets_dict_top:

#         start_index = random.randint(
#             0, len(tweets_dict_top[state]) - min_state)
#         end_index = start_index + min_state
#         sample = tweets_dict_top[state][start_index:end_index]

#         word_vec = get_word_vec(sample)
#         states_words[state] = word_vec

#     print("word_vec")

#     for index, state in enumerate(states_words):

#         if index == 0:
#             for word in states_words[state]:
#                 word_set.add(word)
#         else:
#             set2 = set()
#             for word in states_words[state]:
#                 set2.add(word)
#             word_set = word_set.intersection(set2)
#     print("word_set: ", len(word_set))

#     states_features = {}
#     for state in states_words:
#         overall = sum(states_words[state].values())
#         for word in states_words[state]:
#             states_words[state][word] /= overall

#     for word in list(word_set):
#         states_features[word] = {}
#         word_mean = 0
#         for state in states_words:

#             word_mean += states_words[state][word]

#         word_mean /= len(states_words)
#         states_features[word]["mean"] = word_mean

#         word_stdev = 0
#         for state in states_words:

#             diff = states_words[state][word] - \
#                 states_features[word]["mean"]

#             word_stdev += diff * diff

#         word_stdev /= (len(states_words) - 1)
#         word_stdev = math.sqrt(word_stdev)
#         states_features[word]["stdev"] = word_stdev

#     states_zscores = {}
#     for state in states_words:
#         states_zscores[state] = {}

#         for word in list(word_set)[:]:
#             if word in states_words[state]:
#                 word_state_freq = states_words[state][word]
#                 word_mean = states_features[word]["mean"]
#                 word_stdev = states_features[word]["stdev"]
#                 states_zscores[state][word] = (
#                     word_state_freq - word_mean) / word_stdev
#     print("number of cities", len(states_zscores))
#     result_mat = np.zeros((len(states_zscores), len(states_zscores)))

#     for index, state in enumerate(states_zscores):

#         delats = get_delta(state, states_zscores, list(word_set))
#         values = [value for value in delats.values()]

#         result_mat[index] = values

#     iter_results.append(result_mat)

#     print("--- %s seconds ---" % (time.time() - start_time))


# save_iter_results = open("iter_results_Z_cities.pickle", "wb")
# pickle.dump(iter_results, save_iter_results, -1)
# save_iter_results.close()


# # save_state_words = open(
# #     "merged_data_pickles/state_words_iters_merged_rand.pickle", "wb")
# # pickle.dump(state_words_iters, save_state_words, -1)
# # save_state_words.close()


# # save_word_set_iters = open(
# #     "merged_data_pickles/word_set_iters_merged_rand.pickle", "wb")
# # pickle.dump(word_set_iters, save_word_set_iters, -1)
# # save_word_set_iters.close()


# def sendemail(from_addr, to_addr_list,
#               subject, message,
#               login, password,
#               smtpserver='smtp.gmail.com:587'):
#     header = 'From: %s\n' % from_addr
#     header += 'To: %s\n' % ','.join(to_addr_list)
#     header += 'Subject: %s\n\n' % subject
#     message = header + message

#     server = smtplib.SMTP(smtpserver)
#     server.starttls()
#     server.login(login, password)
#     problems = server.sendmail(from_addr, to_addr_list, message)
#     server.quit()
#     return problems


# sendemail(from_addr='anasnayef1@gmail.com',
#           to_addr_list=['anas.alnayef@uni-weimar.de'],
#           subject='Z_dist done',
#           message='Z_dist done',
#           login='anasnayef1@gmail.com',
#           password='Yeje_9244')
