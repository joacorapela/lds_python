import numpy as np
import pytest
from ssm.inference import filterLDS_SS_withMissingValues_np

@pytest.fixture
def simple_lds_params():
    """Simple 1D state, 1D observation linear dynamical system."""
    B = np.array([[0.9]])
    Q = np.array([[0.1]])
    u = np.array([0.0])
    m0 = np.array([0.0])
    V0 = np.array([[1.0]])
    
    Z = np.array([[1.0]])
    R = np.array([[0.5]])
    a = np.array([0.0])
    
    return B, Q, u, m0, V0, Z, R, a

def test_filter_shapes_and_keys(simple_lds_params):
    B, Q, u, m0, V0, Z, R, a = simple_lds_params
    N = 10
    y = np.random.normal(0, 1, size=(1, N))
    
    res = filterLDS_SS_withMissingValues_np(y, u, B, Q, m0, V0, a, Z, R)
    
    expected_keys = {"xnn1", "Pnn1", "xnn", "Pnn", "innov", "K", "Sn", "log_like"}
    assert expected_keys.issubset(res.keys())
    
    assert res["xnn"].shape == (1, 1, N)
    assert res["Pnn"].shape == (1, 1, N)
    assert isinstance(res["log_like"], float)

def test_missing_values_handling(simple_lds_params):
    B, Q, u, m0, V0, Z, R, a = simple_lds_params
    N = 5
    y = np.array([[1.0, np.nan, 1.2, np.nan, 0.9]])
    
    # Should run successfully without throwing linalg errors or NaNs in states
    res = filterLDS_SS_withMissingValues_np(y, u, B, Q, m0, V0, a, Z, R)
    
    # When missing, filtered state should equal predicted state
    assert np.allclose(res["xnn"][:, :, 1], res["xnn1"][:, :, 1])
    assert np.allclose(res["Pnn"][:, :, 1], res["Pnn1"][:, :, 1])

def test_m0_dimension_validation(simple_lds_params):
    B, Q, u, m0, V0, Z, R, a = simple_lds_params
    y = np.zeros((1, 5))
    
    # Pass 2D m0 instead of 1D
    m0_invalid = np.array([[0.0]])
    
    with pytest.raises(ValueError, match="mean must be 1 dimensional"):
        filterLDS_SS_withMissingValues_np(y, u, B, Q, m0_invalid, V0, a, Z, R)

@pytest.fixture
def stationary_scalar_lds():
    """Defines a simple 1D random walk / AR(1) linear dynamical system with known parameters."""
    B = np.array([[0.95]])      # Stable transition
    Q = np.array([[0.05]])      # Process noise variance
    u = np.array([0.0])         # Zero offset
    m0 = np.array([0.0])        # Initial state mean
    V0 = np.array([[1.0]])      # Initial state variance
    
    Z = np.array([[1.0]])       # Direct observation of the state
    R = np.array([[0.1]])       # Observation noise variance
    a = np.array([0.0])         # Zero observation offset
    
    return B, Q, u, m0, V0, Z, R, a

def test_kalman_filter_accuracy_and_tracking(stationary_scalar_lds):
    """
    Tests that the Kalman filter accurately tracks a generated trajectory 
    and outperforms or mirrors expected statistical bounds.
    """
    B, Q, u, m0, V0, Z, R, a = stationary_scalar_lds
    np.random.seed(42)
    N = 100
    
    # 1. Generate synthetic ground truth states and noisy observations
    true_x = np.zeros((1, N))
    y = np.zeros((1, N))
    
    curr_x = 0.0
    for t in range(N):
        curr_x = (B[0, 0] * curr_x) + np.random.normal(0, np.sqrt(Q[0, 0]))
        true_x[0, t] = curr_x
        y[0, t] = curr_x + np.random.normal(0, np.sqrt(R[0, 0]))

    # 2. Run the Kalman filter
    res = filterLDS_SS_withMissingValues_np(y, u, B, Q, m0, V0, a, Z, R)
    
    xnn = res["xnn"]  # Shape: (1, 1, N)
    
    # 3. Accuracy Check: Filtered states should closely track the true hidden states
    # We compute the root mean squared error (RMSE) over the sequence
    filtered_trajectory = xnn[:, 0, :]
    rmse = np.sqrt(np.mean((filtered_trajectory - true_x) ** 2))
    
    # Given observation noise R=0.1 and process noise Q=0.05, 
    # a well-tuned Kalman filter should achieve low tracking error (e.g., RMSE < 0.5)
    assert rmse < 0.5, f"Filter RMSE ({rmse:.4f}) is too high, indicating inaccurate state estimation."

def test_kalman_filter_missing_data_accuracy(stationary_scalar_lds):
    """
    Tests that introducing substantial missing data (NaNs) forces the filter 
    to correctly propagate uncertainty (covariance increases) and use model predictions.
    """
    B, Q, u, m0, V0, Z, R, a = stationary_scalar_lds
    N = 50
    y = np.ones((1, N)) * 2.0
    
    # Inject consecutive missing values from t=10 to t=25
    y[0, 10:25] = np.nan
    
    res = filterLDS_SS_withMissingValues_np(y, u, B, Q, m0, V0, a, Z, R)
    Pnn = res["Pnn"]
    
    # During missing periods, uncertainty (variance Pnn) should strictly increase 
    # because no measurements arrive to correct the process noise accumulation.
    variances_during_missing = [Pnn[0, 0, t] for t in range(10, 25)]
    
    # Check that variance at the end of the missing block is higher than at the start of it
    assert variances_during_missing[-1] > variances_during_missing[0], \
        "Uncertainty did not grow during consecutive missing data steps, violating dynamical propagation."

def test_log_likelihood_finite_and_sensible(stationary_scalar_lds):
    """
    Ensures that the computed log-likelihood returns a valid finite float 
    and behaves consistently with data alignment.
    """
    B, Q, u, m0, V0, Z, R, a = stationary_scalar_lds
    N = 30
    y = np.random.normal(0, 1, size=(1, N))
    
    res = filterLDS_SS_withMissingValues_np(y, u, B, Q, m0, V0, a, Z, R)
    log_like = res["log_like"]
    
    assert isinstance(log_like, float)
    assert np.isfinite(log_like), "Log-likelihood evaluated to NaN or Infinity."
