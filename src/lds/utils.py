
import numpy as np
import torch


def buildQfromQt_np(Qt, sigma_ax, sigma_ay):
    Q = np.zeros_like(Qt)
    lower_slice = slice(0, int(Q.shape[0]/2))
    upper_slice = slice(int(Q.shape[0]/2), Q.shape[0])
    Q[lower_slice, lower_slice] = sigma_ax**2*Qt[lower_slice, lower_slice]
    Q[upper_slice, upper_slice] = sigma_ay**2*Qt[upper_slice, upper_slice]
    return Q


def buildQfromQt_torch(Qt, sigma_ax, sigma_ay):
    Q = torch.zeros_like(Qt)
    lower_slice = slice(0, int(Q.shape[0]/2))
    upper_slice = slice(int(Q.shape[0]/2), Q.shape[0])
    Q[lower_slice, lower_slice] = sigma_ax**2*Qt[lower_slice, lower_slice]
    Q[upper_slice, upper_slice] = sigma_ay**2*Qt[upper_slice, upper_slice]
    return Q
