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


#-- Helper functions --#
def _prepend(files, dir_path):
    """ Prepend the full directory path to files, so they can be used in open() """
    dir_path += '{0}'
    files = [dir_path.format(i) for i in files]
    return(files)


def _get_files(dirr):
    """ Generates a list of files with full paths inside the given folder name """
    fileDir = os.path.dirname(os.path.abspath(__file__))
    path_dir = fileDir + "/" + dirr + "/"
    files = os.listdir(path=path_dir)
    files_paths = _prepend(files, path_dir)
    files_paths.sort(key=os.path.getmtime)
    return(files_paths)


def _get_delta(index, subsets_zscores):
    """ Generates a list of burrows_deltas for a subset with the rest of the subsets """
    deltas = []
    target = list(subsets_zscores.keys())[index]
    for subset in subsets_zscores:
        delta = 0
        for i in range(len(subsets_zscores[target])):
            delta += math.fabs((subsets_zscores[target]
                                [i] - subsets_zscores[subset][i]))
        delta /= len(subsets_zscores[target])
        deltas.append(delta)
    return deltas


def _getResamplData(pickle_file):
    """ Extracts sample of the dataset given a pickle file """
    resample_file = open(pickle_file, "rb")
    resample_data = pickle.load(resample_file)
    resample_file.close()

    return resample_data


def _get_word_vec(sample):
    """ Generates a directory of word occurrences for subsets in a given sample """
    word_vec = {}
    for tweet in sample:
        for word in tweet.split():
            if word not in word_vec:
                word_vec[word] = 1
            else:
                word_vec[word] += 1
    return word_vec


def _save_results(gran, iter_results, metric):
    """ Saves a pickle file of distance matrix for given granularity and metric after averaging the samples results """

    avr_mat = sum(iter_results) / len(iter_results)
    output_path = "data/" + gran + "/dist_mats/" + metric + "_dist_mat.pickle"
    save_avr_result = open(output_path, "wb")
    pickle.dump(avr_mat, save_avr_result, -1)
    save_avr_result.close()

    file_path = os.path.abspath(output_path)
    print(metric + " distance matrix stored in ", file_path)


def _get_word_set(subsets_words):
    """ Generates a set of word types that are common across all subsets """
    word_set = set()
    for index, subset in enumerate(subsets_words):
        if index == 0:
            for word in subsets_words[subset]:
                word_set.add(word)
        else:
            set2 = set()
            for word in subsets_words[subset]:
                set2.add(word)
            word_set = word_set.intersection(set2)
    return word_set


def _translate(value, leftMin, leftMax):
    """ Translates a value in a given range into 0-1 range """
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = 1 - 0

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return 0 + (valueScaled * rightSpan)

#-- End of helper functions --#


#-- Main functions --#
def Resample(gran, dataset):
    """ Generates samples given granularity and dataset. A pickle for each iteration """
    if gran in {"states", "cities"}:
        resample_path = "data/" + gran + "/resampling"
        min_subset = min([len(dataset[subset])
                          for subset in dataset])
        max_subset = max([len(dataset[subset])
                          for subset in dataset])
        iters = int(round(max_subset / min_subset, 0))

        print("Creating random-resampling data....")
        print("Number of subsets: ", len(dataset))
        print("Smallest subset: ", min_subset, " tweets")
        print("Largest subset: ", max_subset, " tweets")
        print("Num. of iterations: ", iters)

        if os.path.exists(resample_path):
            shutil.rmtree(resample_path)
            os.makedirs(resample_path)
        else:
            os.makedirs(resample_path)

        for i in range(1, iters + 1):
            start_time = time.time()

            #iter_sample = []
            subsets_words = {}

            for subset in dataset:

                start_index = random.randint(
                    0, len(dataset[subset]) - min_subset)
                end_index = start_index + min_subset
                sample = dataset[subset][start_index:end_index]

                word_vec = _get_word_vec(sample)
                subsets_words[subset] = word_vec

            word_set = _get_word_set(subsets_words)
            if i == 1:
                print("Estimated word-types per iteration: ",
                      round(len(word_set), -(len(str(len(word_set))) - 1)))
                print("")

            # iter_sample.append(subsets_words)

            time_elapsed = time.time() - start_time
            print("Finished " + str(i) + "/" + str(iters) + " iteration ")
            print("Estimated time left: ", int(
                time_elapsed * (iters - i)), " sec.")
            print("")

            save_resampling_iter = open(
                "data/" + gran + "/resampling/iter_" + str(i) + ".pickle", "wb")
            pickle.dump(subsets_words, save_resampling_iter, -1)
    else:
        print("'" + gran + "'" +
              " is invalid. Possible values are ('states' , 'cities')")


def Burrows_delta(gran):
    """ Generates burrows_delta matrix given a granularity and store in a pickle file, This must be run after Resample() """
    if gran in {"states", "cities"}:
        resample_path = "data/" + gran + "/resampling"
        if not os.path.exists(resample_path):

            print("No resampling data found! Please run Resample() first.")
        elif len(os.listdir(resample_path)) == 0:
            print("No resampling data found! Please run Resample() first.")
        else:

            print("Starting Burrows_delta...")
            iter_results = []
            for res_index, file in enumerate(_get_files(resample_path)):
                start_time = time.time()

                subsets_words = _getResamplData(file)
                word_set = _get_word_set(subsets_words)

                if res_index == 0:
                    print("Estimated word-types per iteration: ",
                          round(len(word_set), -(len(str(len(word_set))) - 1)))
                    print("")

                for subset in subsets_words:
                    overall = sum(subsets_words[subset].values())
                    for word in subsets_words[subset]:
                        subsets_words[subset][word] /= overall

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
                    subsets_zscores[subset] = []
                    for word in list(word_set):
                        word_subset_freq = subsets_words[subset][word]
                        word_mean = subsets_features[word]["mean"]
                        word_stdev = subsets_features[word]["stdev"]

                        subsets_zscores[subset].append((
                            word_subset_freq - word_mean) / word_stdev)

                result_mat = np.zeros(
                    (len(subsets_zscores), len(subsets_zscores)))

                _get_delta_start_time = time.time()

                for i in range(len(result_mat)):
                    result_mat[i] = _get_delta(i, subsets_zscores)

                print('_get_delta_time', time.time() -
                      _get_delta_start_time)

                iter_results.append(result_mat)
                time_elapsed = time.time() - start_time
                print("Finished " + str(res_index + 1) + "/" + str(len(_get_files(resample_path))) +
                      " iteration ")
                print("Estimated time left: ",
                      int(time_elapsed * (len(_get_files(resample_path)) - (res_index + 1))), " sec.")
                print("")

            _save_results(gran, iter_results, "burrows_delta")

    else:
        print("'" + gran + "'" +
              " is invalid. Possible values are ('states' , 'cities')")


def JSD(gran):
    """ Generates JSD matrix given a granularity and store in a pickle file, This must be run after Resample() """
    resample_path = "data/" + gran + "/resampling"
    if not os.path.exists(resample_path):
        print("No resampling data found! Please run Resample() first.")
    elif len(os.listdir(resample_path)) == 0:
        print("No resampling data found! Please run Resample() first.")
    else:
        print("Starting JSD...")
        iter_results = []
        for res_index, file in enumerate(_get_files(resample_path)):
            start_time = time.time()

            subsets_words = _getResamplData(file)
            word_set = _get_word_set(subsets_words)

            if res_index == 0:
                print("Estimated word-types per iteration: ",
                      round(len(word_set), -(len(str(len(word_set))) - 1)))
                print("")

            for subset in subsets_words:
                overall = sum(subsets_words[subset].values())
                for word in subsets_words[subset]:
                    subsets_words[subset][word] /= overall

            subset_dist = {subset: [] for subset in subsets_words}

            for word in word_set:
                for subset in subsets_words:
                    subset_dist[subset].append(subsets_words[subset][word])

            result_mat = np.zeros((len(subsets_words), len(subsets_words)))

            for index, subset in enumerate(subset_dist):
                subset_jsds = []
                for other_subset in subset_dist:
                    subset_jsds.append(distance.jensenshannon(
                        subset_dist[subset], subset_dist[other_subset], 2.0))
                result_mat[index] = subset_jsds

            iter_results.append(result_mat)

            time_elapsed = time.time() - start_time
            print("Finished " + str(res_index + 1) + "/" + str(len(_get_files(resample_path))) +
                  " iteration ")
            print("Estimated time left: ",
                  int(time_elapsed * (len(_get_files(resample_path)) - (res_index + 1))), " sec.")
            print("")
        _save_results(gran, iter_results, "jsd")


def TF_IDF(gran):
    """ Generates a TF-IDF matrix given a granularity and store in a pickle file, This must be run after Resample() """
    resample_path = "data/" + gran + "/resampling"
    if not os.path.exists(resample_path):
        print("No resampling data found! Please run Resample() first.")
    elif len(os.listdir(resample_path)) == 0:
        print("No resampling data found! Please run Resample() first.")
    else:
        print("Starting TF_IDF...")
        iter_results = []
        for res_index, file in enumerate(_get_files(resample_path)):
            start_time = time.time()

            subsets_words = _getResamplData(file)

            corpus = []
            for subset in subsets_words:
                sub_corpus = " ".join(
                    [(word + ' ') * subsets_words[subset][word] for word in subsets_words[subset]])
                corpus.append(sub_corpus)

            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(corpus)
            tf_idf_dist = manhattan_distances(X)

            iter_results.append(tf_idf_dist)

            time_elapsed = time.time() - start_time
            print("Finished " + str(res_index + 1) + "/" + str(len(_get_files(resample_path))) +
                  " iteration ")
            print("Estimated time left: ",
                  int(time_elapsed * (len(_get_files(resample_path)) - (res_index + 1))), " sec.")
            print("")
        _save_results(gran, iter_results, "tfidf")


def Norm_mat(gran):
    """ Combines the matrices generated from burrows_delta, JSD and TF-IDF by calculating the norm matrix of the 3 """
    dist_path = "data/" + gran + "/dist_mats"
    if not os.path.exists(dist_path):
        print("Missing distance matrices data! Please run Burrows_delta(), JSD(), and TF_IDF() first.")
    elif len(os.listdir(dist_path)) < 3:
        print("Missing distance matrices data! Please run Burrows_delta(), JSD(), and TF_IDF() first.")

    else:
        print("Starting matrices combination...")

        Z_mat_file = open(
            "data/" + gran + "/dist_mats/burrows_delta_dist_mat.pickle", "rb")
        z_mat = pickle.load(Z_mat_file)

        jsd_mat_file = open(
            "data/" + gran + "/dist_mats/jsd_dist_mat.pickle", "rb")
        jsd_mat = pickle.load(jsd_mat_file)

        tfidf_mat_file = open(
            "data/" + gran + "/dist_mats/tfidf_dist_mat.pickle", "rb")
        tfidf_mat = pickle.load(tfidf_mat_file)

        mat_size = len(z_mat)
        norm_mat = np.zeros((mat_size, mat_size))

        z_max = z_mat.max()
        jsd_max = jsd_mat.max()
        tfidf_max = tfidf_mat.max()

        z_min = z_mat.min()
        jsd_min = jsd_mat.min()
        tfidf_min = tfidf_mat.min()

        for i in range(mat_size):
            for j in range(mat_size):

                z_norm = _translate(z_mat[i, j], z_min, z_max)
                jsd_norm = _translate(jsd_mat[i, j], jsd_min, jsd_max)
                tfidf_norm = _translate(tfidf_mat[i, j], tfidf_min, tfidf_max)

                x = np.array([z_norm, jsd_norm, tfidf_norm])
                norm_mat[i, j] = np.linalg.norm(x)

        output_path = "data/" + gran + "/dist_mats/norm_dist_mat.pickle"

        save_norm_mat = open(output_path, "wb")
        pickle.dump(norm_mat, save_norm_mat, -1)

        file_path = os.path.abspath(output_path)
        print("The combination distance matrix stored in ", file_path)
