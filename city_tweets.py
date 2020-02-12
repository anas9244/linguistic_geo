import os
import json
import pickle


def prepend(list, str):
    # Using format()
    str += '{0}'
    list = [str.format(i) for i in list]
    return(list)


def get_files(folder):
    fileDir = os.path.dirname(os.path.abspath(__file__))
    path_dir = fileDir + "/" + folder + "/"
    files = os.listdir(path=path_dir)
    files_paths = prepend(files, path_dir)
    files_paths.sort(key=os.path.basename)
    return(files_paths)


new_states_tweets = {}
for file in get_files("new_states_tweets_merged"):
    state = os.path.basename(file).replace(".json", "")
    new_states_tweets[state] = []
    opened_file = open(file, 'r', encoding="utf-8")
    for index, line in enumerate(opened_file):
        if index > 0:
            new_states_tweets[state].append(line)


save_new_states_tweets = open("new_states_tweets.pickle", "wb")
pickle.dump(new_states_tweets, save_new_states_tweets, -1)
save_new_states_tweets.close()