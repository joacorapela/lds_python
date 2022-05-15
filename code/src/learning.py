
import math
import numpy as np
import scipy.optimize
import torch

import utils
import inference

iteration = 0


def scipy_optimize_SS_tracking_DWPA_fullV0(y, B, sigma_a0, Qt, Z, diag_R_0,
                                           m0_0, V0_0, max_iter=50, disp=True):
    iL_V0 = np.tril_indices(V0_0.shape[0])

    def get_coefs_from_params(sigma_a, diag_R, m0, V0, iL_V0=iL_V0):
        V0_chol = np.linalg.cholesky(V0)
        L_coefs = V0_chol[iL_V0]
        x = np.insert(np.concatenate([diag_R, m0, L_coefs]), 0, sigma_a)
        return x

    def get_params_from_coefs(x, iL_V0=iL_V0, sigma_a0=sigma_a0,
                              diag_R_0=diag_R_0, m0_0=m0_0, V0_0=V0_0):
        cur = 0
        sigma_a = x[slice(cur, cur+1)]
        cur += len(sigma_a)
        diag_R = x[slice(cur, cur+len(diag_R_0))]
        cur += len(diag_R)
        m0 = x[slice(cur, cur+len(m0_0))]
        cur += len(m0)
        M = V0_0.shape[0]
        L_coefs = x[slice(cur, cur+int(M*(M+1)/2))]
        L_V0 = np.zeros(shape=V0_0.shape)
        L_V0[iL_V0] = L_coefs
        V0 = L_V0 @ L_V0.T

        return sigma_a, diag_R, m0, V0

    def optim_criterion(x):
        sigma_a, diag_R, m0, V0 = get_params_from_coefs(x)
        R = np.diag(diag_R)
        kf = inference.filterLDS_SS_withMissingValues(y=y, B=B, Q=sigma_a*Qt,
                                                      m0=m0, V0=V0, Z=Z, R=R)
        answer = 0
        N = kf["Sn"].shape[2]
        for n in range(N):
            innov = kf["innov"][:, :, n]
            Sn = kf["Sn"][:,:,n] 

            Sn_inv = np.linalg.inv(Sn)
            answer += np.linalg.slogdet(Sn)[1]
            answer += innov.T @ Sn_inv @ innov
        return answer

    def callback(x):
        global iteration
        iteration += 1

        sigma_a, diag_R, m0, V0 = get_params_from_coefs(x)
        optim_value = optim_criterion(x=x)
        print("Iteration: {:d}".format(iteration))
        print("optim criterion: {:f}".format(optim_value.item()))
        print("sigma_a={:f}".format(sigma_a.item()))
        print("diag_R:")
        print(diag_R)
        print("m0:")
        print(m0)
        print("V0:")
        print(V0)

    x0 = get_coefs_from_params(sigma_a=sigma_a0, diag_R=diag_R_0,
                               m0=m0_0, V0=V0_0)
    options={"disp": disp, "maxiter": max_iter}
    opt_res = scipy.optimize.minimize(optim_criterion, x0, method="Nelder-Mead",
                                      callback=callback, options=options)
    import pdb; pdb.set_trace()


def scipy_optim_SS_tracking_DWPA_diagV0(y, B, sigma_ax0, sigma_ay0, Qt, Z,
                                        sqrt_diag_R_0, m0_0, sqrt_diag_V0_0,
                                        max_iter=50, disp=True):

    def get_coefs_from_params(sigma_ax, sigma_ay, sqrt_diag_R, m0,
                              sqrt_diag_V0):
        x = np.concatenate([[sigma_ax, sigma_ay], sqrt_diag_R, m0,
                            sqrt_diag_V0])
        return x

    def get_params_from_coefs(x, sigma_ax0=sigma_ax0, sigma_ay0=sigma_ay0,
                              sqrt_diag_R_0=sqrt_diag_R_0, m0_0=m0_0,
                              sqrt_diag_V0_0=sqrt_diag_V0_0):
        cur = 0
        sigma_ax = x[slice(cur, cur+1)]
        cur += len(sigma_ax)
        sigma_ay = x[slice(cur, cur+1)]
        cur += len(sigma_ay)
        sqrt_diag_R = x[slice(cur, cur+len(sqrt_diag_R_0))]
        cur += len(sqrt_diag_R)
        m0 = x[slice(cur, cur+len(m0_0))]
        cur += len(m0)
        sqrt_diag_V0 = x[slice(cur, cur+len(sqrt_diag_V0_0))]

        return sigma_ax, sigma_ay, sqrt_diag_R, m0, sqrt_diag_V0

    def optim_criterion(x):
        sigma_ax, sigma_ay, sqrt_diag_R, m0, sqrt_diag_V0 = \
            get_params_from_coefs(x)
        V0 = np.diag(sqrt_diag_V0**2)
        R = np.diag(sqrt_diag_R**2)
        # build Q from Qt, sigma_ax, sigma_ay
        Q = utils.buildQfromQt_np(Qt=Qt, sigma_ax=sigma_ax, sigma_ay=sigma_ay)

        kf = inference.filterLDS_SS_withMissingValues(y=y, B=B, Q=Q,
                                                      m0=m0, V0=V0, Z=Z, R=R)
        answer = 0
        N = kf["Sn"].shape[2]
        for n in range(N):
            innov = kf["innov"][:, :, n]
            Sn = kf["Sn"][:, :, n]

            Sn_inv = np.linalg.inv(Sn)
            answer += np.linalg.slogdet(Sn)[1]
            answer += innov.T @ Sn_inv @ innov
        return answer

    def callback(x):
        global iteration
        iteration += 1

        sigma_ax, sigma_ay, sqrt_diag_R, m0, sqrt_diag_V0 = \
            get_params_from_coefs(x)
        optim_value = optim_criterion(x=x)
        print("Iteration: {:d}".format(iteration))
        print("optim criterion: {:f}".format(optim_value.item()))
        print("sigma_ax={:f}".format(sigma_ax.item()))
        print("sigma_ay={:f}".format(sigma_ay.item()))
        print("sqrt_diag_R:")
        print(sqrt_diag_R)
        print("m0:")
        print(m0)
        print("sqrt_diag_V0:")
        print(sqrt_diag_V0)

    x0 = get_coefs_from_params(sigma_ax=sigma_ax0, sigma_ay=sigma_ay0,
                               sqrt_diag_R=sqrt_diag_R_0, m0=m0_0,
                               sqrt_diag_V0=sqrt_diag_V0_0)
    options = {"disp": disp, "maxiter": max_iter}
    opt_res = scipy.optimize.minimize(optim_criterion, x0,
                                      method="Nelder-Mead", callback=callback,
                                      options=options)
    sigma_ax, sigma_ay, sqrt_diag_R, m0, sqrt_diag_V0 = \
        get_params_from_coefs(opt_res["x"])
    x = {"sigma_ax": sigma_ax, "sigma_ay": sigma_ay,
         "sqrt_diag_R": sqrt_diag_R, "m0": m0,
         "sqrt_diag_V0": sqrt_diag_V0}
    answer = {"fun": opt_res["fun"], "message": opt_res["message"],
              "nfev": opt_res["nfev"], "nit": opt_res["nit"],
              "status": opt_res["status"], "success": opt_res["success"],
              "x": x}
    return answer


def torch_optimize_SS_tracking_DWPA_diagV0(y, B, sigma_ax0, sigma_ay0, Qt, Z,
                                           sqrt_diag_R_0, m0_0, sqrt_diag_V0_0,
                                           max_iter=50, lr=1.0,
                                           tolerance_grad=1e-7,
                                           tolerance_change=1e-5,
                                           line_search_fn="strong_wolfe",
                                           disp=True):

    def log_likelihood():
        V0 = torch.diag(sqrt_diag_V0**2)
        R = torch.diag(sqrt_diag_R**2)
        # build Q from Qt, sigma_ax, sigma_ay
        Q = utils.buildQfromQt_torch(Qt=Qt, sigma_ax=sigma_ax,
                                     sigma_ay=sigma_ay)

        kf = inference.filterLDS_SS_withMissingValues_torch(y=y, B=B, Q=Q,
                                                            m0=m0, V0=V0, Z=Z,
                                                            R=R)
        answer = 0
        N = y.shape[1]
        for n in range(N):
            innov = kf["innov"][:, :, n]
            Sn = kf["Sn"][:, :, n]

            Sn_inv = torch.inverse(Sn)
            answer = answer + torch.logdet(Sn)
            answer = answer + innov.T @ Sn_inv @ innov
        return answer

    optim_params = {"max_iter": max_iter, "lr": lr,
                    "tolerance_grad": tolerance_grad,
                    "tolerance_change": tolerance_change,
                    "line_search_fn": line_search_fn}
    sigma_ax = torch.Tensor([sigma_ax0])
    sigma_ay = torch.Tensor([sigma_ay0])
    sqrt_diag_R = sqrt_diag_R_0
    m0 = m0_0
    sqrt_diag_V0 = sqrt_diag_V0_0
    x = [sigma_ax, sigma_ay, sqrt_diag_R, m0, sqrt_diag_V0]
    # torch.autograd.set_detect_anomaly(True)
    optimizer = torch.optim.LBFGS(x, **optim_params)
    for i in range(len(x)):
        x[i].requires_grad = True

    def closure():
        optimizer.zero_grad()
        curEval = -log_likelihood()
        print("-logLikeL: ")
        print(curEval)
        curEval.backward(retain_graph=True)
        # x = [sigma_ax, sigma_ay, sqrt_diag_R, m0, sqrt_diag_V0]
        print("sigma_ax: ")
        print(sigma_ax)
        print("sigma_ay: ")
        print(sigma_ax)
        print("sqrt_diag_R: ")
        print(sqrt_diag_R)
        print("m0: ")
        print(m0)
        print("sqrt_diag_V0: ")
        print(sqrt_diag_V0)
        return curEval

    optimizer.step(closure)
    log_likelihood = log_likelihood()
    stateOneEpoch = optimizer.state[optimizer._params[0]]
    nfeval = stateOneEpoch["func_evals"]
    niter = stateOneEpoch["n_iter"]
    answer = {"log_likelihood": log_likelihood, "nfeval": nfeval,
              "niter": niter}
    for i in range(len(x)):
        x[i].requires_grad = False
    return answer


def em_SS_tracking_DWPA(y, B, sigma_a0, Qt, Z, R_0, m0_0, V0_0,
                        vars_to_estimate={"sigma_a": True, "R": True,
                                          "m0": True, "V0": True},
                        max_iter=50):
    sigma_a = sigma_a0
    R = R_0
    m0 = m0_0
    V0 = V0_0
    Qt_inv = np.linalg.inv(Qt)
    vec_Qt_inv = Qt_inv.flatten()

    M = B.shape[0]
    N = y.shape[1]
    log_like = np.empty(max_iter)
    for iter in range(max_iter):
        # E step
        kf = inference.filterLDS_SS_withMissingValues(y=y, B=B, Q=sigma_a**2*Qt,
                                                      m0=m0, V0=V0, Z=Z, R=R)
        print("LogLike[{:04d}]={:f}".format(iter, kf["logLike"].item()))
        print("sigma_a={:f}".format(sigma_a))
        print("R:")
        print(R)
        print("m0:")
        print(m0)
        print("V0:")
        print(V0)
        log_like[iter] = kf["logLike"]
        ks = inference.smoothLDS_SS(B=B, xnn=kf["xnn"], Vnn=kf["Vnn"],
                                    xnn1=kf["xnn1"], Vnn1=kf["Vnn1"],
                                    m0=m0, V0=V0)
        # M step
        if vars_to_estimate["sigma_a"]:
            Vnn1N = lag1CovSmootherLDS_SS(Z=Z, KN=kf["KN"], B=B, Vnn=kf["Vnn"],
                                          Jn=ks["Jn"], J0=ks["J0"])
            S11 = ks["xnN"][:,:,0] @ ks["xnN"][:,:,0].T + ks["VnN"][:,:,0]
            S10 = ks["xnN"][:,:,0] @ ks["x0N"].T + Vnn1N[:,:,0]
            S00 = ks["x0N"] @ ks["x0N"].T + ks["V0N"]
            for i in range(1, N):
                S11 = S11 + ks["xnN"][:, :, i] @ ks["xnN"][:, :, i].T + \
                      ks["VnN"][:, :, i]
                S10 = S10 + ks["xnN"][:, :, i] @ ks["xnN"][:, :, i-1].T + \
                      Vnn1N[:, :, i]
                S00 = S00 + ks["xnN"][:, :, i-1] @ ks["xnN"][:, :, i-1].T + \
                      ks["VnN"][:, :, i-1]

            # sigma_a
            aux1 = S11 - S10 @ B.T - B @ S10.T + B @ S00 @ B.T
            aux2 = Qt_inv @ aux1
            aux3 = np.trace(aux2)
            assert(aux3>0)
            aux4 = math.sqrt(aux3/(N*M))

            tmp = (S11 - S10 @ B.T - B @ S10.T + B @ S00 @ B.T).flatten()
            dot_prod = np.dot(vec_Qt_inv, tmp)
            assert(dot_prod>0)

            sigma_a = math.sqrt(dot_prod/(N*M))

            assert(math.abs(sigma_a-aux4)<1e-6)
            import pdb; pdb.set_trace()

        # R
        if vars_to_estimate["R"]:
            u = y[:, 0] - Z @ ks["xnN"][:, :, 0]
            R = u @ u.T + Z @ ks["VnN"][:, :, 0] @ Z.T
            for i in range(1, N):
                u = y[:, i] - Z @ ks["xnN"][:, :, i]
                R = R + u @ u.T + Z @ ks["VnN"][:, :, i] @ Z.T
            R = R/N

        # m0, V0
        if vars_to_estimate["m0"]:
            m0 = ks["x0N"]
        if vars_to_estimate["V0"]:
            V0 = ks["V0N"]

    return R, m0, V0, sigma_a

def lag1CovSmootherLDS_SS(Z, KN, B, Vnn, Jn, J0):
    M = KN.shape[0]
    N = Vnn.shape[2]
    Vnn1N = np.empty(shape=(M, M, N))
    eye = np.eye(M)
    Vnn1N[:, :, N-1] = (eye - KN @ Z) @ B @ Vnn[:, :, N-2]
    for k in range(N-1, 2):
        Vnn1N[:, :, k-1] = Vnn[:, :, k-1] @ Jn[:, :, k-2].T + \
                           Jn[:, :, k-1] @ \
                           (Vnn1N[:, :, k] - B @ Vnn[:, :, k-1]) @ Jn[:, :, k-2].T
    Vnn1N[:, :, 0] = Vnn[:, :, 0] @ J0.T + Jn[:, :, 0] @ \
                     (Vnn1N[:, :, 1] - B @ Vnn[:, :, 0]) @ J0.T
    return Vnn1N
