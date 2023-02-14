
import math
import numpy as np
import torch


class OnlineKalmanFilter:
    """
    Class implementing the online Kalman filter algorithm for the following
    linear dynamical system:

    .. math::
       x_n &= B\ x_{n-1} + w_n,\ \\textrm{where}\ w_n\sim\mathcal{N}(w_n|0, Q)\, \\textrm{and}\ x_n\in\Re^M

       y_n &= Z\ x_{n-1} + v_n,\ \\textrm{where}\ v_n\sim\mathcal{N}(v_n|0, R)\, \\textrm{and}\ y_n\in\Re^N

       x_0&\in\mathcal{N}(x_0|m_0, V_0)

    Example use:

    .. code-block:: python

        online_kf = OnlineKalmanFilter(B, Q, m0, V0, Z, R)
        x_pred, P_pred = online_kf.predict()

        for y in ys:
            x_filt, P_filt = online_kf.update(y)
            x_pred, P_pred = online_kf.predict()

    A script using `OnlineKalmanFilter` for tracking the position of a mouse
    can be found `here
    <https://github.com/joacorapela/lds_python/blob/master/code/scripts/doOnlineFilterFWGMouseTrajectory.py>`_

    Note 1:
        invocation so `predict()` and `update(y)` should alternate. That is,
        each invocation to `update(y)` should be preceded by an invocation to
        `predict()`, and each invocation to `predict()` (except the first one)
        should be preceded by an invoation to `update(y)`.

    Note 2:
        observations :math:`y_n` should be sampled uniformly.
    """
    def __init__(self, B, Q, m0, V0, Z, R):
        self._B = B
        self._Q = Q
        self._m0 = m0
        self._V0 = V0
        self._Z = Z
        self._R = R

        self._x = m0
        self._P = V0

        M = len(m0)
        self.I = np.eye(M)

    def predict(self):
        """Predicts the next state.

        :return: (state, covariance): tuple containing the predicted state vector and covariance matrix.

        """
        self.x = self.B @ self.x
        self.P = self.B @ self.P @ self.B.T + self.Q
        return self.x, self.P

    def update(self, y):
        """Updates the current state and covariance.

        :param y: observation :math:`\in\Re^M`
        :return: (state, covariance): tuple containing the updated state vector and covariance matrix.

        """
        if not np.isnan(y).any():
            pred_obs = self.Z @ self.x
            residual = y - pred_obs
            Stmp = self.Z @ self.P @ self.Z.T + self.R
            S = (Stmp + Stmp.T) / 2
            Sinv = np.linalg.inv(S)
            K = self.P @ self.Z.T @ Sinv
            self.x = self.x + K @ residual
            self.P = (self.I - K @ self.Z) @ self.P
        return self.x, self.P

    @property
    def B(self):
        return self._B

    @B.setter
    def B(self, B):
        self._B = B

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, Q):
        self._Q = Q

    @property
    def m0(self):
        return self._m0

    @m0.setter
    def m0(self, m0):
        self._m0 = m0

    @property
    def V0(self):
        return self._V0

    @V0.setter
    def V0(self, V0):
        self._V0 = V0

    @property
    def Z(self):
        return self._Z

    @Z.setter
    def Z(self, Z):
        self._Z = Z

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, R):
        self._R = R

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = x

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, P):
        self._P = P


def filterLDS_SS_withMissingValues_torch(y, B, Q, m0, V0, Z, R):
    """ Kalman filter implementation of the algorithm described in Shumway and
    Stoffer 2006.

    :param: y: time series to be smoothed
    :type: y: numpy array (NxT)

    :param: B: state transition matrix
    :type: B: numpy matrix (MxM)

    :param: Q: state noise covariance matrix
    :type: Q: numpy matrix (MxM)

    :param: m0: initial state mean
    :type: m0: one-dimensional numpy array (M)

    :param: V0: initial state covariance
    :type: V0: numpy matrix (MxM)

    :param: Z: state to observation matrix
    :type: Z: numpy matrix (NxM)

    :param: R: observations covariance matrix
    :type: R: numpy matrix (NxN)

    :return:  {xnn1, Vnn1, xnn, Vnn, innov, K, Sn, logLike}: xnn1 and Vnn1 (predicted means, MxT, and covariances, MxMxT), xnn and Vnn (filtered means, MxT, and covariances, MxMxT), innov (innovations, NxT), K (Kalman gain matrices, MxNxT), Sn (innovations covariance matrices, NxNxT), logLike (data loglikelihood, float).
    :rtype: dictionary

    """

    # N: number of observations
    # M: dim state space
    # P: dim observations
    M = B.shape[0]
    N = y.shape[1]
    P = y.shape[0]
    xnn1 = torch.empty(size=[M, 1], dtype=torch.double)
    xnn1_h = torch.empty(size=[M, 1, N], dtype=torch.double)
    Vnn1 = torch.empty(size=[M, M], dtype=torch.double)
    Vnn1_h = torch.empty(size=[M, M, N], dtype=torch.double)
    xnn = torch.empty(size=[M, 1], dtype=torch.double)
    xnn_h = torch.empty(size=[M, 1, N], dtype=torch.double)
    Vnn = torch.empty(size=[M, M], dtype=torch.double)
    Vnn_h = torch.empty(size=[M, M, N], dtype=torch.double)
    innov = torch.empty(size=[P, 1], dtype=torch.double)
    innov_h = torch.empty(size=[P, 1, N], dtype=torch.double)
    Sn = torch.empty(size=[P, P], dtype=torch.double)
    Sn_h = torch.empty(size=[P, P, N], dtype=torch.double)

    # k==0
    xnn1 = (B @ m0).squeeze()
    Vnn1 = B @ V0 @ B.T + Q
    Stmp = Z @ Vnn1 @ Z.T + R
    Sn = (Stmp + torch.transpose(Stmp, 0, 1)) / 2
    Sinv = torch.linalg.inv(Sn)
    K = Vnn1 @ Z.T @ Sinv
    innov = y[:, 0] - (Z @  xnn1).squeeze()
    xnn = xnn1 + K @ innov
    Vnn = Vnn1 - K @ Z @ Vnn1
    logLike = -N*P*math.log(2*math.pi) - torch.logdet(Sn) - \
        innov.T @ Sinv @ innov

    xnn1_h[:, :, 0] = torch.unsqueeze(xnn1, 1)
    Vnn1_h[:, :, 0] = Vnn1
    xnn_h[:, :, 0] = torch.unsqueeze(xnn, 1)
    Vnn_h[:, :, 0] = Vnn
    innov_h[:, :, 0] = torch.unsqueeze(innov, 1)
    Sn_h[:, :, 0] = Sn

    # k>1
    for k in range(1, N):
        xnn1 = B @ xnn
        Vnn1 = B @ Vnn @ B.T + Q
        if(torch.any(torch.isnan(y[:, k]))):
            xnn = xnn1
            Vnn = Vnn1
        else:
            Stmp = Z @ Vnn1 @ Z.T + R
            Sn = (Stmp + Stmp.T)/2
            Sinv = torch.linalg.inv(Sn)
            K = Vnn1 @ Z.T @ Sinv
            innov = y[:, k] - (Z @ xnn1).squeeze()
            xnn = xnn1 + K @ innov
            Vnn = Vnn1 - K @ Z @ Vnn1
        logLike = logLike-torch.logdet(Sn) -\
            innov.T @ Sinv @ innov
        xnn1_h[:, :, k] = torch.unsqueeze(xnn1, 1)
        Vnn1_h[:, :, k] = Vnn1
        xnn_h[:, :, k] = torch.unsqueeze(xnn, 1)
        Vnn_h[:, :, k] = Vnn
        innov_h[:, :, k] = torch.unsqueeze(innov, 1)
        Sn_h[:, :, k] = Sn
    logLike = 0.5 * logLike
    answer = {"xnn1": xnn1_h, "Vnn1": Vnn1_h, "xnn": xnn_h, "Vnn": Vnn_h,
              "innov": innov_h, "KN": K, "Sn": Sn_h, "logLike": logLike}
    return answer


def filterLDS_SS_withMissingValues_np(y, B, Q, m0, V0, Z, R):
    """ Kalman filter implementation of the algorithm described in Shumway and
    Stoffer 2006.

    :param: y: time series to be smoothed
    :type: y: numpy array (NxT)

    :param: B: state transition matrix
    :type: B: numpy matrix (MxM)

    :param: Q: state noise covariance matrix
    :type: Q: numpy matrix (MxM)

    :param: m0: initial state mean
    :type: m0: one-dimensional numpy array (M)

    :param: V0: initial state covariance
    :type: V0: numpy matrix (MxM)

    :param: Z: state to observation matrix
    :type: Z: numpy matrix (NxM)

    :param: R: observations covariance matrix
    :type: R: numpy matrix (NxN)

    :return:  {xnn1, Vnn1, xnn, Vnn, innov, K, Sn, logLike}: xnn1 and Vnn1 (predicted means, MxT, and covariances, MxMxT), xnn and Vnn (filtered means, MxT, and covariances, MxMxT), innov (innovations, NxT), K (Kalman gain matrices, MxNxT), Sn (innovations covariance matrices, NxNxT), logLike (data loglikelihood, float).
    :rtype: dictionary

    """

    # N: number of observations
    # M: dim state space
    # P: dim observations
    M = B.shape[0]
    N = y.shape[1]
    P = y.shape[0]
    xnn1 = np.empty(shape=[M, 1, N])
    Vnn1 = np.empty(shape=[M, M, N])
    xnn = np.empty(shape=[M, 1, N])
    Vnn = np.empty(shape=[M, M, N])
    innov = np.empty(shape=[P, 1, N])
    Sn = np.empty(shape=[P, P, N])

    # k==0
    xnn1[:, 0, 0] = (B @ m0).squeeze()
    Vnn1[:, :, 0] = B @ V0 @ B.T + Q
    Stmp = Z @ Vnn1[:, :, 0] @ Z.T + R
    Sn[:, :, 0] = (Stmp + Stmp.T) / 2
    Sinv = np.linalg.inv(Sn[:, :, 0])
    K = Vnn1[:, :, 0] @ Z.T @ Sinv
    innov[:, 0, 0] = y[:, 0] - (Z @  xnn1[:, :, 0]).squeeze()
    xnn[:, :, 0] = xnn1[:, :, 0] + K @ innov[:, :, 0]
    Vnn[:, :, 0] = Vnn1[:, :, 0] - K @ Z @ Vnn1[:, :, 0]
    logLike = -N*P*np.log(2*np.pi) - np.linalg.slogdet(Sn[:, :, 0])[1] - \
        innov[:, :, 0].T @ Sinv @ innov[:, :, 0]

    # k>1
    for k in range(1, N):
        xnn1[:, :, k] = B @ xnn[:, :, k-1]
        Vnn1[:, :, k] = B @ Vnn[:, :, k-1] @ B.T + Q
        if(np.any(np.isnan(y[:, k]))):
            xnn[:, :, k] = xnn1[:, :, k]
            Vnn[:, :, k] = Vnn1[:, :, k]
        else:
            Stmp = Z @ Vnn1[:, :, k] @ Z.T + R
            Sn[:, :, k] = (Stmp + Stmp.T)/2
            Sinv = np.linalg.inv(Sn[:, :, k])
            K = Vnn1[:, :, k] @ Z.T @ Sinv
            innov[:, 0, k] = y[:, k] - (Z @ xnn1[:, :, k]).squeeze()
            xnn[:, :, k] = xnn1[:, :, k] + K @ innov[:, :, k]
            Vnn[:, :, k] = Vnn1[:, :, k] - K @ Z @ Vnn1[:, :, k]
        logLike = logLike-np.linalg.slogdet(Sn[:, :, k])[1] -\
            innov[:, :, k].T @ Sinv @ innov[:, :, k]
    logLike = 0.5 * logLike
    answer = {"xnn1": xnn1, "Vnn1": Vnn1, "xnn": xnn, "Vnn": Vnn,
              "innov": innov, "KN": K, "Sn": Sn, "logLike": logLike}
    return answer


def smoothLDS_SS(B, xnn, Vnn, xnn1, Vnn1, m0, V0):
    """ Kalman smoother implementation

    :param: B: state transition matrix
    :type: B: numpy matrix (MxM)

    :param: xnn: filtered means (from Kalman filter)
    :type: xnn: numpy array (MxT)

    :param: Vnn: filtered covariances (from Kalman filter)
    :type: Vnn: numpy array (MxMXT)

    :param: xnn1: predicted means (from Kalman filter)
    :type: xnn1: numpy array (MxT)

    :param: Vnn1: predicted covariances (from Kalman filter)
    :type: Vnn1: numpy array (MxMXT)

    :param: m0: initial state mean
    :type: m0: one-dimensional numpy array (M)

    :param: V0: initial state covariance
    :type: V0: numpy matrix (MxM)

    :return:  {xnN, VnN, Jn, x0N, V0N, J0}: xnn1 and Vnn1 (smoothed means, MxT, and covariances, MxMxT), Jn (smoothing gain matrix, MxMxT), x0N and V0N (smoothed initial state mean, M, and covariance, MxM), J0 (initial smoothing gain matrix, MxN).

    """
    N = xnn.shape[2]
    M = B.shape[0]
    xnN = np.empty(shape=[M, 1, N])
    VnN = np.empty(shape=[M, M, N])
    Jn = np.empty(shape=[M, M, N])

    xnN[:, :, -1] = xnn[:, :, -1]
    VnN[:, :, -1] = Vnn[:, :, -1]
    for n in reversed(range(1, N)):
        Jn[:, :, n-1] = Vnn[:, :, n-1] @ B.T @ np.linalg.inv(Vnn1[:, :, n])
        xnN[:, :, n-1] = xnn[:, :, n-1] + \
            Jn[:, :, n-1] @ (xnN[:, :, n]-xnn1[:, :, n])
        VnN[:, :, n-1] = Vnn[:, :, n-1] + \
            Jn[:, :, n-1] @ (VnN[:, :, n]-Vnn1[:, :, n]) @ Jn[:, :, n-1].T
    # initial state x00 and V00
    # return the smooth estimates of the state at time 0: x0N and V0N
    J0 = V0 @ B.T @ np.linalg.inv(Vnn1[:, :, 0])
    x0N = m0 + J0 @ (xnN[:, :, 0] - xnn1[:, :, 0])
    V0N = V0 + J0 @ (VnN[:, :, 0] - Vnn1[:, :, 0]) @ J0.T
    answer = {"xnN": xnN, "VnN": VnN, "Jn": Jn, "x0N": x0N, "V0N": V0N,
              "J0": J0}
    return answer
