import numpy as np
import random
import math
import pickle
import os
import time


class Langdistance:
    def __init__(self, dataset):

        self.dataset = dataset

        self.min_subset = min([len(self.dataset[subset])
                               for subset in self.dataset])
        max_subset = max([len(self.dataset[subset])for subset in self.dataset])

        self.iters = int(round(max_subset / self.min_subset, 0))

    def __get_word_vec(self, sample):
        word_vec = {}
        for tweet in sample:

            tokens = tweet.split()

            for word in tokens:

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

    def __sample(self, dataset):

        subsets_words = {}
        word_set = set()
        for subset in dataset:

            start_index = random.randint(
                0, len(dataset[subset]) - self.min_subset)
            end_index = start_index + self.min_subset
            sample = dataset[subset][start_index:end_index]

            word_vec = self.__get_word_vec(sample)
            subsets_words[subset] = word_vec

        for index, subset in enumerate(subsets_words):

            if index == 0:
                for word in subsets_words[subset]:
                    word_set.add(word)
            else:
                set2 = set()
                for word in subsets_words[subset]:
                    set2.add(word)
                word_set = word_set.intersection(set2)
        return subsets_words, word_set

    def __save_results(self, iter_results, metric):
        D_Z = sum(iter_results) / len(iter_results)

        save_avr_result = open(metric + "_dist_mat.pickle", "wb")
        pickle.dump(D_Z, save_avr_result, -1)
        save_avr_result.close()
        file_patth = os.path.abspath(metric + "_dist_mat.pickle")
        print(metric + " distance matrix stored in ", file_patth)

    def burrows_delta(self):
        iter_results = []
        for i in range(1, self.iters + 1):  # 1,self.iters+1):
            start_time = time.time()
            subsets_words, word_set = self.__sample(self.dataset)
            subsets_features = {}
            for subset in subsets_words:
                overall = sum(subsets_words[subset].values())
                for word in subsets_words[subset]:
                    subsets_words[subset][word] /= overall

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
            print("Time left about: ",
                  time_elapsed * (self.iters - i), " sec.")

        self.__save_results(iter_results, "burrows_delta")
