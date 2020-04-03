from langdistance import LangDistance
import pickle

tweets_dict_file = open("city_tweets_dict.pickle", "rb")
tweets_dict = pickle.load(tweets_dict_file)
tweets_dict_file.close()
print(len(tweets_dict))


tweets_dict_top = {}

for city in tweets_dict:
    if len(tweets_dict[city]) > 5000:
        tweets_dict_top[city] = tweets_dict[city]


dist_lang = LangDistance(tweets_dict_top)
dist_lang.resample()
