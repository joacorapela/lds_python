import numpy as np

def simulateLDS(N, B, Q, Z, R, m0, V0):
    M = B.shape[0]
    P = Z.shape[0]
    # state noise
    w = np.random.multivariate_normal(np.zeros(M), Q, N).T
    # measurement noise
    v = np.random.multivariate_normal(np.zeros(P), R, N).T
    # initial state noise
    x = np.empty(shape=(M, N))
    y = np.empty(shape=(P, N))
    x0 = np.random.multivariate_normal(m0, V0, 1).flatten()
    x[:, 0] = B @ x0 + w[:, 0]
    for n in range(1, N):
        x[:, n] = B @ x[:, n-1] + w[:, n]
    y = Z @ x + v
    return x0, x, y
