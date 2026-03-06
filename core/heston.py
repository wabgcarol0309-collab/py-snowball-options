import numpy as np


def generate_heston_paths(
    S0: float,
    v0: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = None,
) -> np.ndarray:
    """
    Generate asset price paths under the Heston stochastic volatility model
    using the risk-neutral measure.

    Dynamics:
        dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW_1
        dv_t = kappa (theta - v_t) dt + sigma_v sqrt(v_t) dW_2
        corr(dW_1, dW_2) = rho

    Numerical scheme:
        - Variance: Full Truncation Euler
        - Asset price: Log-Euler

    Parameters
    ----------
    S0 : float
        Initial spot price.
    v0 : float
        Initial variance.
    r : float
        Risk-free rate.
    q : float
        Dividend yield / carry rate.
    kappa : float
        Mean reversion speed of variance.
    theta : float
        Long-run variance level.
    sigma_v : float
        Volatility of variance ("vol of vol").
    rho : float
        Correlation between spot and variance shocks.
    T : float
        Maturity in years.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of Monte Carlo paths.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Simulated price paths with shape (n_paths, n_steps + 1).
    """
    _validate_heston_inputs(
        S0=S0,
        v0=v0,
        kappa=kappa,
        theta=theta,
        sigma_v=sigma_v,
        rho=rho,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
    )

    rng = np.random.default_rng(seed)

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    rho_scale = np.sqrt(1.0 - rho * rho)

    # log-price and variance
    logS = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    v = np.empty((n_paths, n_steps + 1), dtype=np.float64)

    logS[:, 0] = np.log(S0)
    v[:, 0] = v0

    # Pre-generate random numbers for vectorized simulation
    z1 = rng.standard_normal((n_paths, n_steps))
    z2 = rng.standard_normal((n_paths, n_steps))

    for t in range(n_steps):
        w1 = z1[:, t]
        w2 = rho * z1[:, t] + rho_scale * z2[:, t]

        v_pos = np.maximum(v[:, t], 0.0)

        # Full truncation Euler for variance
        v_next = (
            v[:, t]
            + kappa * (theta - v_pos) * dt
            + sigma_v * np.sqrt(v_pos) * sqrt_dt * w2
        )
        v[:, t + 1] = np.maximum(v_next, 0.0)

        # Log-Euler for asset price under risk-neutral measure
        logS[:, t + 1] = (
            logS[:, t]
            + (r - q - 0.5 * v_pos) * dt
            + np.sqrt(v_pos) * sqrt_dt * w1
        )

    return np.exp(logS)


def generate_heston_paths_with_variance(
    S0: float,
    v0: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate both price paths and variance paths.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        prices: shape (n_paths, n_steps + 1)
        variances: shape (n_paths, n_steps + 1)
    """
    _validate_heston_inputs(
        S0=S0,
        v0=v0,
        kappa=kappa,
        theta=theta,
        sigma_v=sigma_v,
        rho=rho,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
    )

    rng = np.random.default_rng(seed)

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    rho_scale = np.sqrt(1.0 - rho * rho)

    logS = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    v = np.empty((n_paths, n_steps + 1), dtype=np.float64)

    logS[:, 0] = np.log(S0)
    v[:, 0] = v0

    z1 = rng.standard_normal((n_paths, n_steps))
    z2 = rng.standard_normal((n_paths, n_steps))

    for t in range(n_steps):
        w1 = z1[:, t]
        w2 = rho * z1[:, t] + rho_scale * z2[:, t]

        v_pos = np.maximum(v[:, t], 0.0)

        v_next = (
            v[:, t]
            + kappa * (theta - v_pos) * dt
            + sigma_v * np.sqrt(v_pos) * sqrt_dt * w2
        )
        v[:, t + 1] = np.maximum(v_next, 0.0)

        logS[:, t + 1] = (
            logS[:, t]
            + (r - q - 0.5 * v_pos) * dt
            + np.sqrt(v_pos) * sqrt_dt * w1
        )

    return np.exp(logS), v


def _validate_heston_inputs(
    S0: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    T: float,
    n_steps: int,
    n_paths: int,
) -> None:
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if v0 < 0:
        raise ValueError("v0 must be non-negative.")
    if kappa <= 0:
        raise ValueError("kappa must be positive.")
    if theta < 0:
        raise ValueError("theta must be non-negative.")
    if sigma_v < 0:
        raise ValueError("sigma_v must be non-negative.")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho must be between -1 and 1.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")