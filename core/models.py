from abc import ABC, abstractmethod
import numpy as np
from core.sde import generate_gbm_paths
from core.heston import generate_heston_paths

class BaseModel(ABC):
    """
    市场动力学模型的抽象基类
    """
    @abstractmethod
    def simulate(self, T: float, n_steps: int, n_paths: int, seed: int = None) -> np.ndarray:
        """所有子类必须实现此方法以生成价格路径"""
        pass

class GBM(BaseModel):
    """几何布朗运动 (GBM) 模型"""
    def __init__(self, S0: float, r: float, q: float, vol: float):
        self.S0 = S0
        self.r = r
        self.q = q
        self.vol = vol

    def simulate(self, T: float, n_steps: int, n_paths: int, seed: int = None) -> np.ndarray:
        return generate_gbm_paths(self.S0, self.r, self.q, self.vol, T, n_steps, n_paths, seed)

class Heston(BaseModel):
    """Heston 随机波动率模型"""
    def __init__(self, S0: float, v0: float, r: float, q: float, 
                 kappa: float, theta: float, sigma_v: float, rho: float):
        self.S0 = S0
        self.v0 = v0
        self.r = r
        self.q = q
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho

    def simulate(self, T: float, n_steps: int, n_paths: int, seed: int = None) -> np.ndarray:
        return generate_heston_paths(
            self.S0, self.v0, self.r, self.q, self.kappa, self.theta, 
            self.sigma_v, self.rho, T, n_steps, n_paths, seed
        )