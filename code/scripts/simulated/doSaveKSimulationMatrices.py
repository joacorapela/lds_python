
import sys
import numpy as np


def main(argv):
    simulation_filename = "../../results/00000000_simulation.npz"
    save_filename = "../../results/00000000_simulation_params.npz"
    loadRes = np.load(simulation_filename)
    params = dict(loadRes)
    params["Q"] = params["Qt"]*params["sigma_a"]**2
    params["A"] = params["B"]; del params["B"]
    params["C"] = params["Z"]; del params["Z"]
    del params["x"]
    del params["y"]
    if "dt" in params.keys():
        del params["dt"]
    del params["Qt"]
    del params["sigma_a"]
    np.savez(save_filename, **params)
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
