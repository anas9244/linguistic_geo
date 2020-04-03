import numpy as np
import random
import math
import pickle
import os
import time
from scipy.spatial import distance
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.feature_extraction.text import TfidfVectorizer

import shutil


def prepend(list, str):

    # Using format()
    str += '{0}'
    list = [str.format(i) for i in list]
    return(list)


def get_files(dirr):

    fileDir = os.path.dirname(os.path.abspath(__file__))
    path_dir = fileDir + "/" + dirr + "/"
    files = os.listdir(path=path_dir)
    files_paths = prepend(files, path_dir)

    files_paths.sort(key=os.path.getmtime)

    return(files_paths)


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

        if not os.path.exists('resampling'):
            os.makedirs('resampling')

        start_time = time.time()

        for i in range(1, 5):
            iter_sample = []

            subsets_words = {}
            word_set = set()

            for subset in self.dataset:

                start_index = random.randint(
                    0, len(self.dataset[subset]) - self.min_subset)
                end_index = start_index + self.min_subset
                sample = self.dataset[subset][start_index:end_index]

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

            # for subset in subsets_words:
            #     overall = sum(subsets_words[subset].values())
            #     for word in subsets_words[subset]:
            #         subsets_words[subset][word] /= overall
            if i == 1:
                print("Estimated word-types per iteration: ",
                      round(len(word_set), -(len(str(len(word_set))) - 1)))
                print("")

            iter_sample.append(subsets_words)
            iter_sample.append(word_set)

            time_elapsed = time.time() - start_time
            print("Finished " + str(i) + "/" + str(self.iters) +
                  " iteration ")
            print("Estimated time left: ",
                  int(time_elapsed * (self.iters - i)), " sec.")

            print("")

            save_resampling_iter = open(
                "resampling/iter_" + str(i) + ".pickle", "wb")
            pickle.dump(iter_sample, save_resampling_iter, -1)

        # if metric != "tfidf":
        #     return subsets_words, word_set
        # else:
        #     return sample

    def __save_results(self, iter_results, metric):

        if not os.path.exists('dist_mats'):
            os.mkdir('dist_mats')

        avr_mat = sum(iter_results) / len(iter_results)

        save_avr_result = open("dist_mats/" + metric +
                               "_dist_mat.pickle", "wb")
        pickle.dump(avr_mat, save_avr_result, -1)
        save_avr_result.close()

        file_path = os.path.abspath(
            "dist_mats/" + metric + "_dist_mat.pickle")
        print(metric + " distance matrix stored in ", file_path)

    def __getResamplData(self, pickle_file):
        resample_file = open(pickle_file, "rb")
        resample_data = pickle.load(resample_file)
        resample_file.close()

        return resample_data[0], resample_data[1]

    def Burrows_delta(self):
        iter_results = []
        if len(os.listdir('resampling')) == 0 or not os.path.exists('resampling'):
            print("No resampling data found! Please run resample() first.")
        else:

            for file in get_files('resampling'):
                print(file)
                subsets_words, word_set = self.__getResamplData(file)
                #subsets_words, word_set = self.__sample(self.dataset, "Z")
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

                result_mat = np.zeros(
                    (len(subsets_zscores), len(subsets_zscores)))

                for index, subset in enumerate(subsets_zscores):
                    delats = self.__get_delta(
                        subset, subsets_zscores, list(word_set))
                    values = [value for value in delats.values()]
                    result_mat[index] = values

                iter_results.append(result_mat)

            self.__save_results(iter_results, "burrows_delta")

    def JSD(self):
        iter_results = []
        if len(os.listdir('resampling')) == 0 or not os.path.exists('resampling'):
            print("No resampling data found! Please run resample() first.")
        else:

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


# word_vec = {}
# test = ['Anas alnayef Anas is is very good guy', 'Anas is not so guy good guy']
# print(test)
# for state in test:
#     for word in state.split():
#         if word not in word_vec:
#             word_vec[word] = 1
#         else:
#             word_vec[word] += 1

# print(word_vec)


# test_out = " ".join([(word + ' ') * word_vec[word] for word in word_vec])

# print(test_out)
# word_vec = {}
# # for state in test_out:
# for word in test_out.split():
#     if word not in word_vec:
#         word_vec[word] = 1
#     else:
#         word_vec[word] += 1

# print(word_vec)
