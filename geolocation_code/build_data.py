import json
import os
import pickle
from nltk.stem import PorterStemmer
from collections import OrderedDict
import string
import re
import numpy as np
from geopy.distance import geodesic

import time

# US states names for extracting a state code from  tweet's place[full_name] dield.
# if place_type=='admin, this list will be used
states_full = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
               "Connecticut", "District of Columbia", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois",
               "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
               "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana",
               "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York",
               "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
               "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah",
               "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]
# if place_type=='city', this list will be used
states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

# List of accepted sources for tweets. This is for the cleaned tweets
whitelist = [
    "iPhone",
    "android",
    "web",
    "iPad"]
# This is for raw tweets
whitelist_full = [
    "<a href=\"http://twitter.com/download/iphone\" rel=\"nofollow\">Twitter for iPhone</a>",
    "<a href=\"http://twitter.com/download/android\" rel=\"nofollow\">Twitter for Android</a>",
    "<a href=\"http://twitter.com\" rel=\"nofollow\">Twitter Web Client</a>",
    "<a href=\"http://twitter.com/#!/download/ipad\" rel=\"nofollow\">Twitter for iPad</a>"]


# Path to a folder with json files of tweets where each line is a tweet object
raw_data_path = "/media/data/twitter_geolocation/tweets_clean/"


# Creates main data folder to hold all the clustering resources
if not os.path.exists('data'):
    os.mkdir('data')


def _prepend(files, dir_path):
    """ Prepend the full directory path to files, so they can be used in open() """
    dir_path += '{0}'
    files = [dir_path.format(i) for i in files]
    return(files)


def _get_files():
    """ Generates a list of files with full paths """
    files = os.listdir(path=raw_data_path)
    files_paths = _prepend(files, raw_data_path)
    files_paths.sort(key=os.path.basename)
    return(files_paths)


text_re = re.compile("[^a-zA-Z\s]")
url_re = re.compile("http(s)*://[\w]+\.(\w|/)*(\s|$)")
hashtag_re = re.compile("[\W]#[\w]*[\W]")
mention_re = re.compile("(^|[\W\s])@[\w]*[\W\s]")
smile_re = re.compile("(:\)|;\)|:-\)|;-\)|<3|😀|😃|😂|🤣|😊|😍|😞|😠|😩|😢|😭|😒)")
time_re = re.compile("(^|\D)[\d]+:[\d]+")
numbers_re = re.compile("(^|\D)[\d]+[.'\d]*\D")
repetition_re = re.compile("[\s]+")


def _clean_string(tweet: str):
    """ replace urls, numbers, times, emojis, mentions, split of hashtags"""
    t = tweet.lower()
    t = ":".join(t.split(":")[1:]) if t.startswith("rt") else t
    t = re.sub(url_re, " <url> ", t)
    t = t.replace("\n", "")
    t = t.replace("#", " # ")
    t = re.sub(mention_re, " <user> ", t)
    t = re.sub(smile_re, " <emoji> ", t)
    t = re.sub(time_re, " <time> ", t)
    t = re.sub(numbers_re, " <number> ", t)
    t = re.sub(repetition_re, " ", t)
    t = t.strip()
    return t


def _clean_text(tweet):
    """ Extracts tweet text and apply preproccesing. If raw tweets were used, extract full text if truncated """
    text = ""
    if 'truncated' in tweet:
        if tweet['truncated']:
            text = tweet['extended_tweet']['full_text']
        else:
            text = tweet['text']
    else:
        text = tweet['Text']
    clean_text = _clean_string(text)
    if len(clean_text) > 5:
        return clean_text
    else:
        return False


def _get_center(coords):
    """ Gets geographic center given a tweet's bounding_box """
    south = coords[0][1]
    north = coords[1][1]
    west = coords[0][0]
    east = coords[2][0]

    location = [south, north, west, east]
    centerx, centery = (np.average(location[:2]), np.average(location[2:]))
    center = [centerx, centery]
    return center


def _get_geo_delta(target, coords):
    """ Generates a list of geographic deltas for a subset with the rest of the subsets """
    deltas = {}
    for coord in coords:
        deltas[coord] = geodesic(target, coords[coord]).kilometers
    return deltas


def _get_geo_mat(coords):
    """ Generates a geographic distance matrix given a list of subset's geographic centers """
    result_mat = np.zeros((len(coords), len(coords)))
    for index, coord in enumerate(coords):
        delats = _get_geo_delta(coords[coord], coords)
        values = [value for value in delats.values()]
        result_mat[index] = values
    return result_mat


def _extract_state(tweet):
    """ Extracts a state code given a tweet object """
    key = ''
    if tweet['place']['place_type'] == 'city':
        state_code = tweet['place']['full_name'].split(",")[
            1].lstrip()
        if state_code in states:
            key = state_code

    elif tweet['place']['place_type'] == 'admin':

        state_name = tweet['place']['full_name'].split(",")[
            0].lstrip()

        if state_name in states_full:
            state_index = states_full.index(state_name)
            state_code = states[state_index]
            key = state_code

    return key


def build_data(gran, minsubset, maxsubset):
    """ Generates dataset dictionary given a state/city granularity, and given min and max number of tweets in any subset
    + generates lables for subsets for plotting
    + generates geographic distance matrix for plotting
    + generates a list of ids of the dataset tweets to be used for extracting tweets' data later """

    dataset = {}
    labels = []
    subset_coords = {}
    whitelist_coords = {}
    ids = {}
    whitelist_ids = []
    number_files = len(_get_files())

    if gran in {"states", "cities"}:
        for index, file in enumerate(_get_files()):
            start_time = time.time()
            opened_file = open(file, 'r', encoding="utf-8")
            for line in opened_file:
                tweet = json.loads(line)
                if tweet['place'] is not None:
                    if (tweet['source'] in whitelist or tweet['source'] in whitelist_full) and tweet['place']['country_code'] == 'US' and tweet['place']['place_type'] in ('city', 'admin'):
                        if gran == "cities":
                            if tweet['place']['place_type'] == 'city':
                                key = tweet['place']['full_name']
                                if key not in subset_coords:
                                    coords = tweet['place']['bounding_box']['coordinates'][0]
                                    subset_coords[key] = _get_center(coords)

                                if key not in dataset:
                                    dataset[key] = []
                                if key not in ids:
                                    ids[key] = []
                                if len(dataset[key]) <= maxsubset:
                                    clean_tweet = _clean_text(tweet)
                                    if clean_tweet:
                                        dataset[key].append(clean_tweet)
                                        ids[key].append(tweet['id'])

                        elif gran == "states":
                            key = _extract_state(tweet)
                            if key not in subset_coords:
                                coords = tweet['place']['bounding_box']['coordinates'][0]
                                subset_coords[key] = _get_center(
                                    coords)

                            if key not in dataset:
                                dataset[key] = []
                            if key not in ids:
                                ids[key] = []
                            if len(dataset[key]) <= maxsubset:
                                clean_tweet = _clean_text(tweet)
                                if clean_tweet:
                                    dataset[key].append(clean_tweet)
                                    ids[key].append(tweet['id'])

            time_elapsed = time.time() - start_time
            print("Estimated time left: ", int(
                time_elapsed * (number_files - (index + 1))), " sec.")

        blacklist = []

        for subset in dataset:
            if len(dataset[subset]) <= minsubset:
                blacklist.append(subset)
        for b in blacklist:
            del dataset[b]
            #del subsets_twit_infos[b]
            del ids[b]

        for subset in dataset:
            labels.append(subset)
            whitelist_coords[subset] = subset_coords[subset]

        geo_mat = _get_geo_mat(whitelist_coords)

        for subset in dataset:
            for i in ids[subset]:
                whitelist_ids.append(i)

        print("saving files....")

        data_path = "data/" + gran
        if not os.path.exists("data/" + gran):
            os.mkdir("data/" + gran)

        mat_path = data_path + "/dist_mats"
        if not os.path.exists(mat_path):
            os.mkdir(mat_path)

        save_geo_mat = open(
            data_path + "/dist_mats/geo_mat.pickle", "wb")
        pickle.dump(geo_mat, save_geo_mat, -1)

        save_dataset = open(
            data_path + "/dataset.pickle", "wb")
        pickle.dump(dataset, save_dataset, -1)

        save_labels = open(
            data_path + "/labels.pickle", "wb")
        pickle.dump(labels, save_labels, -1)

        save_tweet_ids = open(
            data_path + "/tweet_ids.pickle", "wb")
        pickle.dump(whitelist_ids, save_tweet_ids, -1)

    else:
        print("'" + gran + "'" +
              " is invalid. Possible values are ('states' , 'cities')")


# Accepted values for gran: 'states', 'cities'
# Recommneded maxsubset for gran='states' to be >1000000 for better representation
if __name__ == "__main__":
    build_data(gran="cities", minsubset=5000, maxsubset=200000)
