import numpy as np
import matplotlib.pyplot as plt
import pickle
from operator import itemgetter
import math
import tweepy
# from shapely.geometry import MultiPoint, Point, Polygon
# import shapefile
# return a polygon for each state in a dictionary

from geopy.distance import geodesic


tweets_dict_file = open("city_tweets_dict.pickle", "rb")
tweets_dict = pickle.load(tweets_dict_file)
tweets_dict_file.close()

tweets_dict_top = {}

for city in tweets_dict:
    if len(tweets_dict[city]) > 5000:
        tweets_dict_top[city] = tweets_dict[city]
places = [i for i in tweets_dict_top]

access_token = "588509788-1z07dMNOMhOCw4OyLJwxSfse31TR7Aywj6h2uZgd"
access_token_secret = "WPlTT3cpVXREXWuGvjdbxD8ie92e61vadWFlzxcoesjHe"
consumer_key = "Em7YjncUOkyjxzZhP3hWWUDJL"
consumer_secret = "IXkMkxVh1eFq9FJpo5vjI1NsTlzAscsEezVRjxhHZIBAnJiaEO"

auth_key = [access_token, access_token_secret, consumer_key, consumer_secret]


access_token2 = "1118134186274062339-MGNkDPMhhQn0uwZlVY73T1tFvJvJ8w"
access_token_secret2 = "D6eXlgVJiJ5q3RQBAO03GhizanMVItD8dIZOhsBvujhUO"
consumer_key2 = "WC98mcJLSHMNna64T0EyzhGhV"
consumer_secret2 = "UKp18iWGxIZxcriW0nl1MD9b12R8fP7TIEUNb83eX1M91Vbzq9"


auth_key2 = [access_token2, access_token_secret2,
             consumer_key2, consumer_secret2]


ACCESS_TOKEN3 = "962405164924710912-rBkxVbzZmOhVUfJH4pcIWTeWNGoI38P"
ACCES_TOKEN_SECRET3 = "dAMqdN8ZX4LZG1G55DtDTq6mOfcmKqfL8NrUz9DstSCwJ"
CONSUMER_KEY3 = "zupgS7bx1lCj7Wv4YqxRIyCIV"
CONSUMER_SECRET3 = "c1RLamc68W6Izwdz6Cx81ZLNBnUqsuWLWwmXccregNggXAuo25"

auth_key3 = [ACCESS_TOKEN3, ACCES_TOKEN_SECRET3,
             CONSUMER_KEY3, CONSUMER_SECRET3]

auth = tweepy.OAuthHandler(auth_key3[2], auth_key3[3])
auth.set_access_token(auth_key3[0], auth_key3[1])

API = tweepy.API(auth)

print (len(places))
# for i in places:
#   #print (i)
#   #place=API.geo_id(i)
#   print (API.geo_id(i).centroid)
#   #print(place.bounding_box.coordinates)


places_coor={}
active_api = 0

#error=False

while len(places_coor) != len(places):

    for index, p in enumerate(places):

        if p not in places_coor:
            print (index)
            print (len(places_coor),len(places))

            try:
                places_coor[p]=API.geo_id(p).centroid
                error = False
            except tweepy.error.RateLimitError as e:
                error = True
                print(e)
                if active_api == 0:
                    active_api = 1
                    print(active_api)

                    auth = tweepy.OAuthHandler(
                        auth_key[2], auth_key[3])
                    auth.set_access_token(
                        auth_key[0], auth_key[1])

                    API = tweepy.API(auth)
                    #places_coor.append(API.geo_id(p).centroid)
                elif active_api == 1:
                    active_api = 2
                    print(active_api)
                    auth = tweepy.OAuthHandler(
                        auth_key2[2], auth_key2[3])
                    auth.set_access_token(
                        auth_key2[0], auth_key2[1])

                    API = tweepy.API(auth)
                    #places_coor.append(API.geo_id(p).centroid)
                elif active_api == 2:
                    active_api = 0
                    print(active_api)
                    auth = tweepy.OAuthHandler(
                        auth_key3[2], auth_key3[3])
                    auth.set_access_token(
                        auth_key3[0], auth_key3[1])

                    API = tweepy.API(auth)
                    #places_coor.append(API.geo_id(p).centroid)



# def get_dist(place01,place02):
#     coor1=API.geo_id(place01).centroid
#     coor2=API.geo_id(place02).centroid




#     return geodesic(coor1, coor2).kilometers



# def get_geo_delta(target):
#     coor1=API.geo_id(target).centroid
#     coor1=[coor1[1],coor1[0]]
#     deltas = {}
#     for p in places:
#         delta = 0
#         coor2=API.geo_id(p).centroid
#         coor2=[coor2[1],coor2[0]]



#         deltas[p] = geodesic(coor1, coor2).kilometers

#     return deltas



# result_mat = np.zeros((len(places), len(places)))

# for index, p in enumerate(places):

#     delats = get_geo_delta(p)
#     values = [value for value in delats.values()]
#     result_mat[index] = values



save_get_mat = open("places_coor.pickle", "wb")
pickle.dump(places_coor, save_get_mat, -1)
save_get_mat.close()