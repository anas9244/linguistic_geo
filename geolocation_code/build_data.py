import json
import os
import pickle
from nltk.stem import PorterStemmer
from collections import OrderedDict
import string
import re

states_full = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
               "Connecticut", "District of Columbia", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois",
               "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
               "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana",
               "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York",
               "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
               "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah",
               "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]

states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]


whitelist = [
    "iPhone",
    "android",
    "web",
    "iPad"]

whitelist_full = [
    "<a href=\"http://twitter.com/download/iphone\" rel=\"nofollow\">Twitter for iPhone</a>",
    "<a href=\"http://twitter.com/download/android\" rel=\"nofollow\">Twitter for Android</a>",
    "<a href=\"http://twitter.com\" rel=\"nofollow\">Twitter Web Client</a>",
    "<a href=\"http://twitter.com/#!/download/ipad\" rel=\"nofollow\">Twitter for iPad</a>"]

##########################
raw_data_path = "/media/data/twitter_geolocation/tweets_clean/"
##########################

if not os.path.exists('data'):
    os.mkdir('data')


ps = PorterStemmer()
punc = set(string.punctuation)


def prepend(list, str):

    # Using format()
    str += '{0}'
    list = [str.format(i) for i in list]
    return(list)


def get_files():

    #fileDir = os.path.dirname(os.path.abspath(__file__))
    #path_dir = fileDir + "/" + dirr + "/"
    files = os.listdir(path=raw_data_path)
    files_paths = prepend(files, raw_data_path)

    files_paths.sort(key=os.path.basename)

    return(files_paths)


def norm(txt):
    ps = PorterStemmer()

    punc = set(string.punctuation)

    no_punc = False

    for i in txt:
            # check for punctioans
        if i not in punc:
            # check for emojies
            if i.isalnum():
                no_punc = True
                break

    if no_punc:
        if txt.isnumeric():
            return("[num]")
        elif re.match("^https?:\/\/.*[\r\n]*", txt):
            return("[url]")
        else:
            if not txt.startswith('#'):
                txt = txt.lower()
                if txt.startswith('@'):
                    return("[mention]")
                else:
                    txt.lstrip(string.punctuation)
                    txt.rstrip(string.punctuation)
                    if txt.isalnum():
                        stemmed = ps.stem(txt)
                        return(stemmed)

                    else:
                        norm_txt = ""

                        for i in txt:
                            if i.isalnum() or i in ["-", "_"]:
                                norm_txt += i

                        stemmed = ps.stem(norm_txt)
                        return(stemmed)
            else:
                return(txt)

    else:
        # clean symbols/emojies repetitions
        cleaned = "".join(OrderedDict.fromkeys(txt))
        return(cleaned)


def norm_text(tweet):
    text = ""
    if 'truncated' in tweet:
        if tweet['truncated']:
            text = tweet['extended_tweet']['full_text']
        else:
            text = tweet['text']
    else:
        text = tweet['Text']
    normed_tokens = []
    for t in text.split():
        normed_tokens.append(norm(t))
    normed_text = " ".join(normed_tokens)
    if normed_tokens > 4:
        return normed_text
    else:
        return False


def build_data(gran, minsubset, maxsubset):

    dataset = {}
    labels = []

    for index, file in enumerate(get_files()):
        print(file)

        opened_file = open(file, 'r', encoding="utf-8")
        for line in opened_file:
            tweet = json.loads(line)
            if tweet['place'] != None:

                if (tweet['source'] in whitelist or tweet['source'] in whitelist_full) and tweet['place']['country_code'] == 'US' and tweet['place']['place_type'] in ('city', 'admin'):

                    if gran == "city":
                        if tweet['place']['place_type'] == 'city':
                            key = tweet['place']['full_name']

                            if key not in dataset:
                                dataset[key] = []
                            if len(dataset[key]) <= maxsubset:
                                clean_tweet = norm_text(tweet)
                                if clean_tweet:
                                    dataset[key].append(clean_tweet)

                    elif gran == "state":

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

                        if key not in dataset:
                            dataset[key] = []
                        if len(dataset[key]) <= maxsubset:
                            clean_tweet = norm_text(tweet)
                            if clean_tweet:
                                dataset[key].append(clean_tweet)
                        # else:
                        #     print(key + " has " + str(maxsubset) + " tweets")

    blacklist = []
    for key in dataset:
        if len(dataset[key]) <= minsubset:
            blacklist.append(key)
    for b in blacklist:
        del dataset[b]

    for key in dataset:
        labels.append(key)

    print("saving files....")

    data_path = "data/" + gran
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    save_city_tweets_dict = open(
        data_path + "/dataset.pickle", "wb")
    pickle.dump(dataset, save_city_tweets_dict, -1)

    save_city_tweets_dict = open(
        data_path + "/lables.pickle", "wb")
    pickle.dump(labels, save_city_tweets_dict, -1)


build_data(gran="states", minsubset=5000, maxsubset=2000000)
