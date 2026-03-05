import numpy as np
from core.payoff import SnowballOption
from core.engine import MonteCarloEngine
from core.models import GBM, Heston  # 导入我们的模型库

def main():
    print("=== 雪球期权定价系统 (多模型支持) ===\n")
    
    # 1. 通用参数
    spot_price = 100.0
    r = 0.03
    q = 0.0
    T = 1.0
    n_steps = 252
    
    # 2. 设定雪球产品条款
    ko_obs_indices = np.arange(21, 253, 21)
    snowball = SnowballOption(
        initial_price=spot_price, ko_barrier=103.0, ki_barrier=80.0,
        coupon_rate=0.20, ko_obs_indices=ko_obs_indices, T=T, notional=1000000.0
    )
    
    engine = MonteCarloEngine(n_paths=100000, n_steps=n_steps, seed=42)
    
    # ==========================================
    # 对比测试：分别使用 GBM 和 Heston 模型进行定价
    # ==========================================
    
    print("正在使用 GBM 模型计算...")
    gbm_model = GBM(S0=spot_price, r=r, q=q, vol=0.15)
    gbm_result = engine.calculate_pv(option=snowball, model=gbm_model, r=r)
    print(f"GBM 理论价格: {gbm_result['price']:,.2f} (SE: {gbm_result['standard_error']:.2f})\n")
    
    print("正在使用 Heston 模型计算 (考虑波动率偏斜)...")
    # Heston 参数：方差均值回归至 0.0225 (即 vol=15%)，引入负相关性 rho=-0.7 产生左偏斜
    heston_model = Heston(
        S0=spot_price, v0=0.0225, r=r, q=q, 
        kappa=2.0, theta=0.0225, sigma_v=0.1, rho=-0.7
    )
    heston_result = engine.calculate_pv(option=snowball, model=heston_model, r=r)
    print(f"Heston 理论价格: {heston_result['price']:,.2f} (SE: {heston_result['standard_error']:.2f})")

if __name__ == "__main__":
    main()