import pickle
from langdistance import Resample, Burrows_delta, JSD, TF_IDF


def get_dataset(gran):
    dataset_file = open("data/" + gran + "/dataset.pickle", "rb")
    dataset = pickle.load(dataset_file)
    dataset_file.close()

    return dataset


dataset = get_dataset('states')
Resample(dataset)
Burrows_delta()
JSD()
TF_IDF()
