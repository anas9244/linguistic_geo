import pickle
from langdistance import Resample, Burrows_delta, JSD, TF_IDF, Norm_mat


def get_dataset(gran):
    dataset_file = open("data/" + gran + "/dataset.pickle", "rb")
    dataset = pickle.load(dataset_file)
    dataset_file.close()

    return dataset


def create_mats(gran):

    dataset = get_dataset(gran)
    print(len(dataset))
    Resample(gran, dataset)
    # Burrows_delta(gran)
    # JSD(gran)
    # TF_IDF(gran)
    # Norm_mat(gran)


create_mats('states')
