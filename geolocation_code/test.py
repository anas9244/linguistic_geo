import json
import os
import pickle
from nltk.stem import PorterStemmer
from collections import OrderedDict
import string
import re
import numpy as np
from geopy.distance import geodesic


def _prepend(list, str):

    # Using format()
    str += '{0}'
    list = [str.format(i) for i in list]
    return(list)


raw_data_path = "/media/data/twitter_geolocation/json/"


def _get_files():

    #fileDir = os.path.dirname(os.path.abspath(__file__))
    #path_dir = fileDir + "/" + dirr + "/"
    files = os.listdir(path=raw_data_path)
    files_paths = _prepend(files, raw_data_path)

    files_paths.sort(key=os.path.basename, reverse=True)

    return(files_paths)


for file in _get_files():
    print(file)
    opend_file = open(file, 'r', encoding="utf-8")
    for line in opend_file:
        tweet = json.loads(line)
        t_keys = set(tweet.keys())
        if 'Text' in t_keys:
            print(file, tweet)
