import pickle
from langdistance import Resample, Burrows_delta, JSD, TF_IDF, Norm_mat
import os


def get_dataset(gran):
    dist_path = "data/" + gran
    if not os.path.exists(dist_path):
        print(
            "Missing dataset data! Please run build_data.py first.")
        exit()
    elif len(os.listdir(dist_path)) == 0:
        print(
            "Missing dataset data! Please run build_data.py first.")
        exit()

    else:
        dataset_file = open(dist_path + "/dataset.pickle", "rb")
        dataset = pickle.load(dataset_file)
        dataset_file.close()

        return dataset


def create_mats(gran):

    dataset = get_dataset(gran)
    print(len(dataset))
    Resample(gran, dataset)
    Burrows_delta(gran)
    JSD(gran)
    TF_IDF(gran)
    Norm_mat(gran)


create_mats('cities')
