import os


# import json


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


def rchop(s, suffix):
    if suffix and s.endswith(suffix):
        return s[:-len(suffix)]
    return s


file_num = 352
for file in get_files("json_clean02"):
    file_name = os.path.basename(file)
    #print(rchop(file, file_name))
    # os.path.basename(file)
    # print(os.path.basename(file))
    #print(file)
    #print(rchop(file, file_name) + str(file_num) + ".jsonl")
    os.rename(file,rchop(file, file_name)+str(file_num)+".jsonl")

    file_num += 1
    # print(file)


# def set_file(file_index):
#     fileDir = os.path.dirname(os.path.abspath(
#         __file__)) + "/json_clean_old/"
#     path = fileDir + str(file_index) + ".jsonl"
#     file = open(path, 'a', encoding="utf-8")
#     return file


# count = 0
# tweet_dict = {}
# out_file = set_file(0)
# file_index = 0


# skip = False
# finished = False

# whitelist = [
#     "<a href=\"http://twitter.com/download/iphone\" rel=\"nofollow\">Twitter for iPhone</a>",
#     "<a href=\"http://twitter.com/download/android\" rel=\"nofollow\">Twitter for Android</a>",
#     "<a href=\"http://twitter.com\" rel=\"nofollow\">Twitter Web Client</a>",
#     "<a href=\"http://twitter.com/#!/download/ipad\" rel=\"nofollow\">Twitter for iPad</a>"]


# for index, file in enumerate(get_files("json")):
#     # if index >= 220:
#         # if index == :
#         #     break
#     opened_file = open(file, 'r', encoding="utf-8")
#     print(index, file)

#     for line in opened_file:
#         try:
#             tweet = json.loads(line)
#         except:
#             print(tweet)

#         if tweet['id']:

#             if tweet['truncated']:
#                 text = tweet['extended_tweet']['full_text']
#             else:
#                 text = tweet['text']

#             count += 1
#             if tweet['geo'] is not None:
#                 geo = tweet['geo']
#             else:
#                 geo = None

#             # if tweet['source'] == whitelist[0]:
#             #     source = "instagram"
#             if tweet['source'] == whitelist[0]:
#                 source = "iPhone"
#             elif tweet['source'] == whitelist[1]:
#                 source = "android"
#             elif tweet['source'] == whitelist[2]:
#                 source = "web"
#             elif tweet['source'] == whitelist[3]:
#                 source = "iPad"

#             tweet_dict = {'id': tweet['id'],
#                           'Text': text,
#                           'geo': geo,
#                           'place': tweet['place'],
#                           'source': source,
#                           'created_at': tweet['created_at'],
#                           'user': tweet['user']}

#             if count % 100000 == 0:
#                 json.dump(tweet_dict, out_file, ensure_ascii=False)
#                 out_file.write("\n")
#                 file_index += 1
#                 out_file = set_file(file_index)
#                 print(str(count))
#             else:
#                 json.dump(tweet_dict, out_file, ensure_ascii=False)
#                 out_file.write("\n")
