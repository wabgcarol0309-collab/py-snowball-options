from abc import ABC, abstractmethod
from typing import Callable
import numpy as np

from core.sde import generate_gbm_paths
from core.heston import generate_heston_paths


class BaseModel(ABC):
    """
    市场动力学模型的抽象基类
    所有模型都必须实现统一的路径模拟接口。
    """

    @abstractmethod
    def simulate(
        self,
        T: float,
        n_steps: int,
        n_paths: int,
        seed: int = None
    ) -> np.ndarray:
       
        raise NotImplementedError

    @staticmethod
    def _validate_common_inputs(T: float, n_steps: int, n_paths: int) -> None:
        if T <= 0:
            raise ValueError("T must be positive.")
        if n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if n_paths <= 0:
            raise ValueError("n_paths must be positive.")


class GBM(BaseModel):
    """几何布朗运动 (GBM) 模型"""

    def __init__(self, S0: float, r: float, q: float, vol: float):
        if S0 <= 0:
            raise ValueError("S0 must be positive.")
        if vol < 0:
            raise ValueError("vol must be non-negative.")

        self.S0 = float(S0)
        self.r = float(r)
        self.q = float(q)
        self.vol = float(vol)

    def simulate(
        self,
        T: float,
        n_steps: int,
        n_paths: int,
        seed: int = None
    ) -> np.ndarray:
        self._validate_common_inputs(T, n_steps, n_paths)
        return generate_gbm_paths(
            self.S0, self.r, self.q, self.vol,
            T, n_steps, n_paths, seed
        )

    def __repr__(self) -> str:
        return (
            f"GBM(S0={self.S0}, r={self.r}, q={self.q}, vol={self.vol})"
        )


class Heston(BaseModel):
    """Heston 随机波动率模型"""

    def __init__(
        self,
        S0: float,
        v0: float,
        r: float,
        q: float,
        kappa: float,
        theta: float,
        sigma_v: float,
        rho: float
    ):
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

        self.S0 = float(S0)
        self.v0 = float(v0)
        self.r = float(r)
        self.q = float(q)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma_v = float(sigma_v)
        self.rho = float(rho)

    @property
    def feller_value(self) -> float:
        """
        Feller condition: 2*kappa*theta - sigma_v^2
        大于等于 0 时更不容易触碰 0 方差。
        """
        return 2.0 * self.kappa * self.theta - self.sigma_v ** 2

    def simulate(
        self,
        T: float,
        n_steps: int,
        n_paths: int,
        seed: int = None
    ) -> np.ndarray:
        self._validate_common_inputs(T, n_steps, n_paths)
        return generate_heston_paths(
            self.S0,
            self.v0,
            self.r,
            self.q,
            self.kappa,
            self.theta,
            self.sigma_v,
            self.rho,
            T,
            n_steps,
            n_paths,
            seed
        )

    def __repr__(self) -> str:
        return (
            f"Heston(S0={self.S0}, v0={self.v0}, r={self.r}, q={self.q}, "
            f"kappa={self.kappa}, theta={self.theta}, "
            f"sigma_v={self.sigma_v}, rho={self.rho})"
        )


class LocalVol(BaseModel):
    """
    局部波动率模型 (Local Volatility Model)
    """

    def __init__(
        self,
        S0: float,
        r: float,
        q: float,
        local_vol_func: Callable[[np.ndarray, float], np.ndarray]
    ):
        if S0 <= 0:
            raise ValueError("S0 must be positive.")
        if not callable(local_vol_func):
            raise TypeError("local_vol_func must be callable.")

        self.S0 = float(S0)
        self.r = float(r)
        self.q = float(q)
        self.local_vol_func = local_vol_func

    def simulate(
        self,
        T: float,
        n_steps: int,
        n_paths: int,
        seed: int = None
    ) -> np.ndarray:
        self._validate_common_inputs(T, n_steps, n_paths)

        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        
        paths = np.empty((n_steps + 1, n_paths), dtype=np.float64)
        paths[0] = self.S0

        z = rng.standard_normal((n_steps, n_paths))

        for i in range(n_steps):
            t = i * dt
            s_t = paths[i]

            sigma_t = np.asarray(self.local_vol_func(s_t, t), dtype=np.float64)

            if sigma_t.shape != s_t.shape:
                raise ValueError(
                    "local_vol_func must return an array with shape (n_paths,)"
                )

            sigma_t = np.clip(sigma_t, 1e-8, None)

            log_s_next = (
                np.log(s_t)
                + (self.r - self.q - 0.5 * sigma_t ** 2) * dt
                + sigma_t * sqrt_dt * z[i]
            )

            paths[i + 1] = np.exp(log_s_next)

        return paths

    def __repr__(self) -> str:
        return (
            f"LocalVol(S0={self.S0}, r={self.r}, q={self.q}, "
            f"local_vol_func={self.local_vol_func.__name__ if hasattr(self.local_vol_func, '__name__') else 'callable'})"
        )