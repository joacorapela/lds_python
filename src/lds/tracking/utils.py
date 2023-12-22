
import numpy as np
import torch


def getLDSmatricesForTracking(dt, sigma_a, sigma_x, sigma_y):
    # Taken from the book
    # barShalomEtAl01-estimationWithApplicationToTrackingAndNavigation.pdf
    # section 6.3.3

    # Eq. 6.3.3-2
    B = np.array([[1, dt, .5*dt**2, 0, 0, 0],
                   [0, 1, dt, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1, dt, .5*dt**2],
                   [0, 0, 0, 0, 1, dt],
                   [0, 0, 0, 0, 0, 1]],
                  dtype=np.double)
    Z = np.array([[1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0]],
                  dtype=np.double)
    # Eq. 6.3.3-4
    Qt = np.array([[dt**4/4, dt**3/2, dt**2/2, 0, 0, 0],
                   [dt**3/2, dt**2,   dt,      0, 0, 0],
                   [dt**2/2, dt,      1,       0, 0, 0],
                   [0, 0, 0, dt**4/4, dt**3/2, dt**2/2],
                   [0, 0, 0, dt**3/2, dt**2,   dt],
                   [0, 0, 0, dt**2/2, dt,      1]],
                  dtype=np.double)
    R = np.diag([sigma_x**2, sigma_y**2])
    Q = buildQfromQt_np(Qt=Qt, sigma_ax=sigma_a, sigma_ay=sigma_a)

    return B, Q, Z, R, Qt


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
