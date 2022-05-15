import sys
import pickle
import numpy as np

sys.path.append("../code/src")
import inference

def test_filterLDS_SS_withMissingValues():
    tol = 1e-4
    data_filename = "data/filterLDS_SS_withMissingValues.csv"
    with open(data_filename, "rb") as f:
        load_res = pickle.load(f)
    y = load_res["y"]
    B = load_res["B"]
    Q = load_res["Q"]
    m0 = load_res["m0"]
    V0 = load_res["V0"]
    Z = load_res["Z"]
    R = load_res["R"]
    xnn = load_res["xnn"]
    Vnn = load_res["Vnn"]
    xnn1 = load_res["xnn1"]
    Vnn1 = load_res["Vnn1"]

    filterRes = inference.filterLDS_SS_withMissingValues_np(y=y, B=B, Q=Q, m0=m0,
                                                            V0=V0, Z=Z, R=R)
    mse_xnn = np.mean((xnn - filterRes["xnn"])**2)
    assert(mse_xnn<=tol)

    mse_Vnn = np.mean((Vnn - filterRes["Vnn"])**2)
    assert(mse_Vnn<=tol)

    mse_xnn1 = np.mean((xnn1 - filterRes["xnn1"])**2)
    assert(mse_xnn1<=tol)

    mse_Vnn1 = np.mean((Vnn1 - filterRes["Vnn1"])**2)
    assert(mse_Vnn1<=tol)

if __name__=="__main__":
    test_filterLDS_SS_withMissingValues()
