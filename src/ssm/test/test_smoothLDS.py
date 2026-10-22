import numpy as np
import pytest
from ssm.inference import filterLDS_SS_withMissingValues_np, smoothLDS_SS

@pytest.fixture
def lds_setup():
    """Sets up a standard linear dynamical system for testing."""
    B = np.array([[0.9]])
    Q = np.array([[0.1]])
    u = np.array([0.0])
    m0 = np.array([0.0])
    V0 = np.array([[1.0]])
    
    Z = np.array([[1.0]])
    R = np.array([[0.5]])
    a = np.array([0.0])
    
    N = 20
    np.random.seed(0)
    y = np.random.normal(0, 1, size=(1, N))
    
    return B, Q, u, m0, V0, Z, R, a, y

def test_smooth_shapes_and_keys(lds_setup):
    B, Q, u, m0, V0, Z, R, a, y = lds_setup
    N = y.shape[1]
    M = B.shape[0]
    
    # Run filter first to get required inputs
    filt_res = filterLDS_SS_withMissingValues_np(y, u, B, Q, m0, V0, a, Z, R)
    
    # Run smoother
    smooth_res = smoothLDS_SS(
        B, 
        filt_res["xnn"], 
        filt_res["Pnn"], 
        filt_res["xnn1"], 
        filt_res["Pnn1"], 
        m0, 
        V0
    )
    
    expected_keys = {"xnN", "PnN", "Jn", "x0N", "V0N", "J0"}
    assert expected_keys.issubset(smooth_res.keys())
    
    assert smooth_res["xnN"].shape == (M, 1, N)
    assert smooth_res["PnN"].shape == (M, M, N)
    assert smooth_res["x0N"].shape == (M, 1)
    assert smooth_res["V0N"].shape == (M, M)

def test_smoothing_reduces_uncertainty(lds_setup):
    """
    Core theoretical property test: 
    Smoothed state variance (PnN) should generally be less than or equal 
    to filtered state variance (Pnn) because it uses future observations.
    """
    B, Q, u, m0, V0, Z, R, a, y = lds_setup
    N = y.shape[1]
    
    filt_res = filterLDS_SS_withMissingValues_np(y, u, B, Q, m0, V0, a, Z, R)
    smooth_res = smoothLDS_SS(
        B, 
        filt_res["xnn"], 
        filt_res["Pnn"], 
        filt_res["xnn1"], 
        filt_res["Pnn1"], 
        m0, 
        V0
    )
    
    Pnn = filt_res["Pnn"]
    PnN = smooth_res["PnN"]
    
    # Exclude the very last time step where filter == smoother by definition
    for n in range(N - 1):
        filtered_var = Pnn[0, 0, n]
        smoothed_var = PnN[0, 0, n]
        assert smoothed_var <= filtered_var + 1e-7, \
            f"Smoothed variance ({smoothed_var}) exceeded filtered variance ({filtered_var}) at step {n}."
        # if smoothed_var > filtered_var + 1e-7:
        #     breakpoint()

def test_smoother_initial_state(lds_setup):
    """Tests that initial state smoothing outputs valid formats and shapes."""
    B, Q, u, m0, V0, Z, R, a, y = lds_setup
    filt_res = filterLDS_SS_withMissingValues_np(y, u, B, Q, m0, V0, a, Z, R)
    smooth_res = smoothLDS_SS(
        B, 
        filt_res["xnn"], 
        filt_res["Pnn"], 
        filt_res["xnn1"], 
        filt_res["Pnn1"], 
        m0, 
        V0
    )
    
    assert np.isfinite(smooth_res["x0N"]).all()
    assert np.isfinite(smooth_res["V0N"]).all()

if __name__ == "__main__":
    # 1. Manually instantiate the fixture data
    B = np.array([[0.9]])
    Q = np.array([[0.1]])
    u = np.array([0.0])
    m0 = np.array([0.0])
    V0 = np.array([[1.0]])
    Z = np.array([[1.0]])
    R = np.array([[0.5]])
    a = np.array([0.0])
    
    N = 20
    np.random.seed(0)
    y = np.random.normal(0, 1, size=(1, N))
    
    lds_data = (B, Q, u, m0, V0, Z, R, a, y)
    
    # 2. Run the test function passing the unpacked data
    test_smoothing_reduces_uncertainty(lds_data)
    print("Test executed successfully!")

