import os
import numpy as np


def delta_perm(params, skip=10):
    """Compute and print out permittivity interval

    Parameters
    ----------
    path : string
        pyEIT-mesh

    skip : int
        skipping interval

    """
    perms = []
    path = params["path"]
    fnames = os.listdir(path)
    for file in fnames[::skip]:
        tmp = np.load(path + file, allow_pickle=True)["mesh_obj"].tolist()
        perms.append(tmp["perm"])
    perms = np.array(perms)
    print("Max Perm:\t", np.max(perms), "\n Min Perm:\t", np.min(perms[perms > 1]))


# Runtime routines to print and read text, faster


def check_train_data(params):
    print("Path to data:\t\t", params["path"])
    print("Number of samples:\t", len(os.listdir(params["path"])))


def read_integer(msg):
    in_ = input("\t", msg)
    return int(in_)
