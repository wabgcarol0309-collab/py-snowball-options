import numpy as np
from core.payoff import SnowballOption
from core.engine import MonteCarloEngine
from core.models import GBM, Heston  # 导入我们的模型库
from utils.data_loader import fetch_cn_index_data, get_cn_risk_free_rate


def main():
    print("=== 雪球期权定价系统 (多模型支持) ===\n")
    
    index_symbol = "000905"  # 中证 500 指数代码
    
    try:
        spot_price, hist_vol = fetch_cn_index_data(symbol=index_symbol)
    except Exception as e:
        print(f"系统错误: {e}")
        return
        
    r = get_cn_risk_free_rate()
    q = 0.0
    
    
    # 2. 设定经典的中证 500 雪球条款
    # ==========================================
    T = 1.0         # 1年期产品
    n_steps = 252
    ko_obs_indices = np.arange(21, 253, 21) # 每月观察一次敲出
    
    # 国内经典结构：100% 平价敲出，80% 深度敲入
    ko_barrier = spot_price * 1.00  
    ki_barrier = spot_price * 0.80  
    
    print(f"\n根据 [中证500] 现价生成挂钩合约:")
    print(f"期初点位: {spot_price:.2f}")
    print(f"敲出线 (100%): {ko_barrier:.2f} (每月观察)")
    print(f"敲入线 (80%):  {ki_barrier:.2f} (每日观察)\n")
    
    snowball = SnowballOption(
        initial_price=spot_price, ko_barrier=ko_barrier, ki_barrier=ki_barrier,
        coupon_rate=0.20, # 极具吸引力的 20% 年化票息
        ko_obs_indices=ko_obs_indices, T=T, notional=1000000.0
    )
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

    from core.models import LocalVol # 记得在文件顶部导入



    print("\n正在使用 Local Volatility 模型计算 (模拟波动率微笑)...")
    
    # 构造一个模拟的 Local Vol 面函数 (Dummy Vol Surface)
    # 这里我们设定一个典型的 Equity Smile：平值(ATM)波动率最低，两端(OTM/ITM)波动率升高
    # 公式：sigma(S, t) = 0.15 + 0.005 * ((S - 100) / 10)^2 
    # 注意：真实交易中，这个函数会是由真实期权市场报价插值计算得出的 Dupire 曲面
    def dummy_smile_surface(S: np.ndarray, t: float) -> np.ndarray:
        atm_vol = 0.15
        skew_convexity = 0.005 * np.square((S - spot_price) / 10.0)
        # 为了防止极端路径下波动率爆炸，我们设置一个波动率上限 80%
        return np.clip(atm_vol + skew_convexity, a_min=0.01, a_max=0.80)

    # 实例化 Local Vol 模型
    lv_model = LocalVol(S0=spot_price, r=r, q=q, local_vol_func=dummy_smile_surface)
    
    # 直接塞进我们之前写好的通用引擎里！
    lv_result = engine.calculate_pv(option=snowball, model=lv_model, r=r)
    
    print(f"Local Vol 理论价格: {lv_result['price']:,.2f} (SE: {lv_result['standard_error']:.2f})")


if __name__ == "__main__":
    main()