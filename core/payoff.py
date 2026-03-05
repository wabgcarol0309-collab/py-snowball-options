import numpy as np
from dataclasses import dataclass

@dataclass
class SnowballOption:
    """
    雪球期权合约参数数据类 (Data Class)
    """
    initial_price: float       # 期初标的价格 (S0)
    ko_barrier: float          # 敲出边界 (Knock-out Barrier)，例如 103% * S0
    ki_barrier: float          # 敲入边界 (Knock-in Barrier)，例如 80% * S0
    coupon_rate: float         # 年化票息率 (Coupon Rate)，例如 20% (0.20)
    ko_obs_indices: np.ndarray # 敲出观察日的索引数组 (例如每个月对应的交易日索引)
    T: float                   # 产品期限 (年化)
    notional: float = 1.0      # 本金 (Notional Principal)

    def calculate_payoff(self, paths: np.ndarray, dt: float) -> np.ndarray:
        """
        全向量化计算所有路径的雪球期权到期收益 (Payoff)。
        
        参数:
        paths : 形状为 (n_steps + 1, n_paths) 的价格路径矩阵
        dt    : 每次步长 (年化)
        """
        n_paths = paths.shape[1]
        payoffs = np.zeros(n_paths)

        # ==========================================
        # 1. 敲出 (Knock-out) 判断
        # ==========================================
        # 提取敲出观察日的价格矩阵: shape (len(ko_obs_indices), n_paths)
        ko_obs_prices = paths[self.ko_obs_indices, :]
        
        # 判断在观察日是否大于等于敲出价 (布尔矩阵)
        is_ko_matrix = ko_obs_prices >= self.ko_barrier
        
        # 对每条路径，判断该路径是否发生过敲出 (只要有一个观察日满足即为敲出)
        is_ko_path = np.any(is_ko_matrix, axis=0)
        
        # 找到敲出发生的具体月份/时间点，以计算实际获得的票息
        # np.argmax 会返回第一个为 True 的索引
        ko_idx_in_matrix = np.argmax(is_ko_matrix, axis=0)
        ko_steps = self.ko_obs_indices[ko_idx_in_matrix] 
        ko_time = ko_steps * dt  # 敲出时经过的年化时间

        # ==========================================
        # 2. 敲入 (Knock-in) 判断
        # ==========================================
        # 通常敲入是每日观察，即判断整个路径的最低点是否低于敲入边界
        min_prices = np.min(paths, axis=0)
        is_ki_path = min_prices < self.ki_barrier

        # ==========================================
        # 3. 向量化计算三种收益情形 (核心亮点)
        # ==========================================
        final_prices = paths[-1, :]

        # 情形 A：发生敲出 (Knock-out)
        # 收益 = 本金 * (1 + 年化票息 * 实际存续期)
        payoffs = np.where(is_ko_path, 
                           self.notional * (1.0 + self.coupon_rate * ko_time), 
                           payoffs)
        
        # 情形 B：未敲出，也未敲入 (No KO, No KI) -> 拿到高额红利
        # 收益 = 本金 * (1 + 年化票息 * 完整期限)
        condition_no_ko_no_ki = (~is_ko_path) & (~is_ki_path)
        payoffs = np.where(condition_no_ko_no_ki, 
                           self.notional * (1.0 + self.coupon_rate * self.T), 
                           payoffs)
        
        # 情形 C：未敲出，但发生敲入 (No KO, but KI) -> 承担标的下跌亏损
        # 收益 = 本金 * (期末价格 / 期初价格)
        condition_no_ko_but_ki = (~is_ko_path) & is_ki_path
        payoffs = np.where(condition_no_ko_but_ki, 
                           self.notional * (final_prices / self.initial_price), 
                           payoffs)

        return payoffs
    