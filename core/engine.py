import numpy as np
from core.payoff import SnowballOption
from core.models import BaseModel

class MonteCarloEngine:
    """
    重构后的蒙特卡洛定价引擎 (模型不可知 Model-Agnostic)
    """
    def __init__(self, n_paths: int = 100000, n_steps: int = 252, seed: int = None):
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed

    def calculate_pv(self, option: SnowballOption, model: BaseModel, r: float) -> dict:
        """
        计算期权现值
        
        参数:
        option : 雪球期权合约对象
        model  : 实例化的市场动力学模型 (GBM 或 Heston)
        r      : 无风险利率 (用于最终的期望值贴现)
        """
        T = option.T
        dt = T / self.n_steps
        
        # 1. 多态调用：引擎不关心底层是常数波动率还是随机波动率
        paths = model.simulate(
            T=T, n_steps=self.n_steps, n_paths=self.n_paths, seed=self.seed
        )
        
        # 2. 计算所有路径的到期收益
        payoffs = option.calculate_payoff(paths=paths, dt=dt)
        
        # 3. 风险中性贴现 (Discounting)
        discount_factor = np.exp(-r * T)
        discounted_payoffs = payoffs * discount_factor
        
        # 4. 计算统计量
        price = np.mean(discounted_payoffs)
        standard_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(self.n_paths)
        
        return {
            "price": price,
            "standard_error": standard_error
        }