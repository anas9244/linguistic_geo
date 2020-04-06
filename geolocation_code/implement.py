from langdistance import LangDistance
import pickle

tweets_dict_file = open("dataset_state.pickle", "rb")
tweets_dict = pickle.load(tweets_dict_file)
tweets_dict_file.close()
print(len(tweets_dict))


#tweets_dict_top = {}

# for city in tweets_dict:
#     if len(tweets_dict[city]) > 5000:
#         tweets_dict_top[city] = tweets_dict[city]
#         #print (tweets_dict_top[city][0])


dist_lang = LangDistance(tweets_dict)
dist_lang.Resample()
# dist_lang.Burrows_delta()
#dist_lang.JSD()
#dist_lang.TF_IDF()


##(50, 583490)
#(50, 50)

# inished 1/5 iteration
# Estimated time left:  129  sec.

# Finished 2/5 iteration
# Estimated time left:  97  sec.

# Finished 3/5 iteration
# Estimated time left:  63  sec

##########################


# 2393
# Finished 1/63 iteration
# Estimated time left:  1603  sec.

# Finished 2/63 iteration
# Estimated time left:  1580  sec.

# Finished 3/63 iteration
# Estimated time left:  1564  sec.
