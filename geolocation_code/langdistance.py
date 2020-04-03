import numpy as np
import random
import math
import pickle
import os
import time
from scipy.spatial import distance
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.feature_extraction.text import TfidfVectorizer


class LangDistance:
    def __init__(self, dataset):

        self.dataset = dataset

        self.min_subset = min([len(self.dataset[subset])
                               for subset in self.dataset])
        self.max_subset = max([len(self.dataset[subset])
                               for subset in self.dataset])

        self.iters = int(round(self.max_subset / self.min_subset, 0))

    def __get_word_vec(self, sample):
        word_vec = {}
        for tweet in sample:
            for word in tweet.split():
                if word not in word_vec:
                    word_vec[word] = 1
                else:
                    word_vec[word] += 1
        return word_vec

    def __get_delta(self, target, subsets_zscores, word_list):
        deltas = {}
        for subset in subsets_zscores:
            delta = 0
            for word in word_list:
                try:
                    delta += math.fabs((subsets_zscores[target]
                                        [word] - subsets_zscores[subset][word]))
                except:
                    pass
                    # print(subset, word)
            # print(len(word_list))
            delta /= len(word_list)
            deltas[subset] = delta

        return deltas

    def resample(self):

        print("Starting random resampling....")
        print("Number of subsets: ", len(self.dataset))
        print("Smallest subset: ", self.min_subset, " tweets")
        print("Largest subset: ", self.max_subset, " tweets")
        print("Num. of iterations: ", self.iters)

        os.mkdir('resampling')

        start_time = time.time()

        for i in range(1, 2):
            iter_sample = []

            subsets_words = {}
            word_set = set()
            corpus = []
            for subset in self.dataset:

                start_index = random.randint(
                    0, len(self.dataset[subset]) - self.min_subset)
                end_index = start_index + self.min_subset
                sample = self.dataset[subset][start_index:end_index]

                word_vec = self.__get_word_vec(sample)
                subsets_words[subset] = word_vec

                sub_corpus = ""
                for tweet in sample:
                    sub_corpus += " " + tweet
                corpus.append(sub_corpus)

            for index, subset in enumerate(subsets_words):
                if index == 0:
                    for word in subsets_words[subset]:
                        word_set.add(word)
                else:
                    set2 = set()
                    for word in subsets_words[subset]:
                        set2.add(word)
                    word_set = word_set.intersection(set2)

            for subset in subsets_words:
                overall = sum(subsets_words[subset].values())
                for word in subsets_words[subset]:
                    subsets_words[subset][word] /= overall
            if i == 1:
                print("Estimated word-types per iteration: ",
                      round(len(word_set), -(len(str(len(word_set))) - 1)))
                print("")

            iter_sample.append(subsets_words)
            iter_sample.append(word_set)
            iter_sample.append(corpus)

            time_elapsed = time.time() - start_time
            print("Finished " + str(i) + "/" + str(self.iters) +
                  " iteration ")
            print("Estimated time left: ",
                  int(time_elapsed * (self.iters - i)), " sec.")

            print("")

            save_resampling_iter = open(
                "resampling/iter_" + i + ".pickle", "wb")
            pickle.dump(iter_sample, save_resampling_iter, -1)

        # if metric != "tfidf":
        #     return subsets_words, word_set
        # else:
        #     return sample

    def __save_results(self, iter_results, metric):
        os.mkdir('dist_mats')
        avr_mat = sum(iter_results) / len(iter_results)

        save_avr_result = open("dist_mats/"+metric + "_dist_mat.pickle", "wb")
        pickle.dump(avr_mat, save_avr_result, -1)
        save_avr_result.close()

        file_patth = os.path.abspath("dist_mats/"+metric + "_dist_mat.pickle")
        print(metric + " distance matrix stored in ", file_patth)

    def Burrows_delta(self):
        iter_results = []

        print("Starting burrows-delta with random resampling....")
        print("Number of subsets: ", len(self.dataset))
        print("Smallest subset: ", self.min_subset, " tweets")
        print("Largest subset: ", self.max_subset, " tweets")
        print("Num. of iterations: ", self.iters)

        for i in range(1, self.iters + 1):  # 1,self.iters+1):

            start_time = time.time()
            subsets_words, word_set = self.__sample(self.dataset, "Z")
            if i == 1:
                print("Estimated word-types per iteration: ",
                      round(len(word_set), -(len(str(len(word_set))) - 1)))
                print("")
            subsets_features = {}
            for word in list(word_set):
                subsets_features[word] = {}
                word_mean = 0
                for subset in subsets_words:

                    word_mean += subsets_words[subset][word]

                word_mean /= len(subsets_words)
                subsets_features[word]["mean"] = word_mean

                word_stdev = 0
                for subset in subsets_words:

                    diff = subsets_words[subset][word] - \
                        subsets_features[word]["mean"]

                    word_stdev += diff * diff

                word_stdev /= (len(subsets_words) - 1)
                word_stdev = math.sqrt(word_stdev)
                subsets_features[word]["stdev"] = word_stdev

            subsets_zscores = {}
            for subset in subsets_words:
                subsets_zscores[subset] = {}

                for word in list(word_set)[:]:
                    if word in subsets_words[subset]:
                        word_subset_freq = subsets_words[subset][word]
                        word_mean = subsets_features[word]["mean"]
                        word_stdev = subsets_features[word]["stdev"]
                        subsets_zscores[subset][word] = (
                            word_subset_freq - word_mean) / word_stdev

            result_mat = np.zeros((len(subsets_zscores), len(subsets_zscores)))

            for index, subset in enumerate(subsets_zscores):
                delats = self.__get_delta(
                    subset, subsets_zscores, list(word_set))
                values = [value for value in delats.values()]
                result_mat[index] = values

            iter_results.append(result_mat)

            time_elapsed = time.time() - start_time
            print("Finished " + str(i) + "/" + str(self.iters) +
                  " iteration ")
            print("Estimated time left: ",
                  int(time_elapsed * (self.iters - i)), " sec.")

            print("")

        self.__save_results(iter_results, "burrows_delta")

    def JSD(self):
        iter_results = []
        print("Starting JSD with random resampling....")
        print("Number of subsets: ", len(self.dataset))
        print("Smallest subset: ", self.min_subset, " tweets")
        print("Largest subset: ", self.max_subset, " tweets")
        print("Num. of iterations: ", self.iters)

        for i in range(1, self.iters + 1):
            start_time = time.time()

            subsets_words, word_set = self.__sample(self.dataset, "jsd")
            if i == 1:
                print("Estimated word-types per iteration: ",
                      round(len(word_set), -(len(str(len(word_set))) - 1)))
                print("")
            subset_dist = {subset: [] for subset in subsets_words}
            for word in word_set:
                for subset in subsets_words:
                    subset_dist[subset].append(subsets_words[subset][word])

            result_mat = np.zeros((len(subsets_words), len(subsets_words)))

            for index, state in enumerate(subset_dist):
                subset_jsds = []

                for other_subset in subset_dist:

                    subset_jsds.append(distance.jensenshannon(
                        subset_dist[subset], subset_dist[other_subset], 2.0))

                result_mat[index] = subset_jsds

            iter_results.append(result_mat)

            time_elapsed = time.time() - start_time
            print("Finished " + str(i) + "/" + str(self.iters) +
                  " iteration ")
            print("Estimated time left: ",
                  int(time_elapsed * (self.iters - i)), " sec.")
            print("")
        self.__save_results(iter_results, "JSD")

    def TF_IDF(self):
        pass

        for i in range(1, self.iters + 1):
            corpus = []
            for subset in self.dataset:

                sample = self.__sample(self.dataset, "tfidf")
                sub_corpus = ""
                for tweet in sample:
                    sub_corpus += " " + tweet
                corpus.append(sub_corpus)

            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(corpus)

            tf_idf_dist = manhattan_distances(X)
            print(tf_idf_dist.shape)

            iter_results.append(tf_idf_dist)
