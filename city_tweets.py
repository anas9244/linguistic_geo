import json
from operator import itemgetter
import twokenize
import os
from nltk.stem import PorterStemmer
import string
import re
from collections import OrderedDict
from collections import Counter
from statistics import mean

import pickle
ps = PorterStemmer()

punc = set(string.punctuation)


def prepend(list, str):
    # Using format()
    str += '{0}'
    list = [str.format(i) for i in list]
    return(list)


def get_files(path):
    #fileDir = os.path.dirname(os.path.abspath(__file__))
    path_dir = path
    files = os.listdir(path=path_dir)
    files_paths = prepend(files, path_dir)
    files_paths.sort(key=os.path.getmtime)
    return(files_paths)


new_set = get_files("/home/yeje2865/Desktop/Hiwi-Webis/json_clean/")
new_set02 = get_files("/home/yeje2865/Desktop/Hiwi-Webis/json_clean02/")
old_set = get_files("/home/yeje2865/Desktop/Geo_linguistics/json_resampled/")

data_set = new_set + new_set02 + old_set

print(len(data_set))


def norm(txt):

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


def city_corpus():
    # fileDir = os.path.dirname(__file__)
    # path = fileDir + "/json/11.jsonl"
    # file = open(path, 'r', encoding="utf-8")
    city_tweets = {}

    # text = []

    for index, file in enumerate(data_set):
        # if index == 1:
        #     break
        opened_file = open(file, 'r', encoding="utf-8")
        print(file)
        for line in opened_file:
            #valid = False

            tweet = json.loads(line)

            if tweet['place'] != None:

                if (tweet['source'] in whitelist or tweet['source'] in whitelist_full) and tweet['place']['place_type'] == 'city' and tweet['place']['country_code'] == 'US':

                    city = tweet['place']['id']
                    # place_id = tweet['place']['full_name'].replace("/", "_")
                    if city not in city_tweets:

                        city_tweets[city] = {}
                        city_tweets[city]['tweets'] = []
                        city_tweets[city]['name'] = tweet['place']['full_name']
                        city_tweets[city]['BoW'] = {}
                        city_tweets[city]['tokens'] = 0
                        city_tweets[city]['types'] = 0
                        #city_tweets[city]['all_tweets_lengths'] = []
                        city_tweets[city]['reply_tweets_lengths'] = [
                        ]
                        city_tweets[city]['non_reply_tweets_lengths'] = [
                        ]

                    if 'Text' in tweet:
                        tweet_text = tweet['Text']
                    else:
                        tweet_text = tweet['full_text']

                    tokens = tweet_text.split()
                    bow = []
                    for t in tokens:
                        bow.append(norm(t))
                    if tweet_text.startswith('@'):
                        city_tweets[city]['reply_tweets_lengths'].append(
                            len(bow))
                    else:

                        city_tweets[city]['non_reply_tweets_lengths'].append(
                            len(bow))

                    for word in bow:

                        city_tweets[city]['tokens'] += 1

                        if word not in city_tweets[city]['BoW']:
                            city_tweets[city]['types'] += 1
                            city_tweets[city]['BoW'][word] = 1

                        else:
                            city_tweets[city]['BoW'][word] += 1

                    city_tweets[city]['tweets'].append(
                        " ".join(bow))

    return city_tweets


def set_file(file_name):
    fileDir = os.path.dirname(os.path.abspath(
        __file__)) + "/city_tweets/"
    path = fileDir + file_name + ".json"
    file = open(path, 'w', encoding="utf-8")
    return file


city_tweets = city_corpus()

city_tweets_dict = {}
for city in city_tweets:

    if len(city_tweets[city]['tweets']) > 1000:

        if city not in city_tweets_dict:
            city_tweets_dict[city] = city_tweets[city]['tweets']

        tweets = city_tweets[city]['tweets']
        tokens = city_tweets[city]['tokens']
        out_file = set_file(city)
        try:
            meta_dict = {'name': city_tweets[city]['name'], 'num_of_tweets': len(city_tweets[city]['tweets']), 'tokens': tokens, 'types': city_tweets[city]['types'], 'type-token-ratio': city_tweets[city]['types'] / tokens, 'avergae_tweet_length': mean(
                city_tweets[city]['reply_tweets_lengths'] + city_tweets[city]['non_reply_tweets_lengths']), 'average_non_reply_length': mean(city_tweets[city]['non_reply_tweets_lengths']), 'average_reply_length': mean(city_tweets[city]['reply_tweets_lengths'])}
        except:
            print(city_tweets[city])
        json.dump(meta_dict, out_file, ensure_ascii=False)
        out_file.write("\n")
        for tweet in tweets:

            json.dump(tweet, out_file, ensure_ascii=False)
            out_file.write("\n")
        out_file.close()


save_city_tweets_dict = open(
    "city_tweets_dict.pickle", "wb")

pickle.dump(city_tweets_dict, save_city_tweets_dict, -1)
save_city_tweets_dict.close()


# tweets_dict_file = open("city_tweets_dict.pickle", "rb")
# tweets_dict = pickle.load(tweets_dict_file)
# tweets_dict_file.close()


# print(len(tweets_dict))
