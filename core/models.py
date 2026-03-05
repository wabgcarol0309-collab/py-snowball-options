from abc import ABC, abstractmethod
import numpy as np
from core.sde import generate_gbm_paths
from core.heston import generate_heston_paths
from typing import Callable

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
    
class LocalVol(BaseModel):
    """
    局部波动率模型 (Local Volatility Model)
    """
    def __init__(self, S0: float, r: float, q: float, local_vol_func: Callable[[np.ndarray, float], np.ndarray]):
        """
        参数:
        local_vol_func : 一个可调用函数 (通常是插值器)，输入为当前的资产价格数组 S_t 和当前时间 t，
                         返回对应的局部波动率数组 sigma_t。
        """
        self.S0 = S0
        self.r = r
        self.q = q
        self.local_vol_func = local_vol_func

    def simulate(self, T: float, n_steps: int, n_paths: int, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
            
        dt = T / n_steps
        # 生成标准正态随机数矩阵
        Z = np.random.standard_normal((n_steps, n_paths))
        
        # 预分配路径内存
        S_paths = np.zeros((n_steps + 1, n_paths))
        S_paths[0] = self.S0
        
        # 时间步迭代 (空间维度 n_paths 保持全向量化)
        for t in range(n_steps):
            current_time = t * dt
            current_S = S_paths[t]
            
            # 【核心差异】：动态获取当前状态下的局部波动率
            # 此时 sigma_t 是一个 shape 为 (n_paths,) 的数组
            sigma_t = self.local_vol_func(current_S, current_time)
            
            # 使用对数 Euler-Maruyama 离散化，保证数值稳定性 (资产价格不会变负)
            log_S_next = np.log(current_S) + (self.r - self.q - 0.5 * sigma_t**2) * dt + \
                         sigma_t * np.sqrt(dt) * Z[t]
                         
            S_paths[t+1] = np.exp(log_S_next)
            
        return S_paths