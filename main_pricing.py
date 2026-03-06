import numpy as np
from core.payoff import SnowballOption
from core.engine import MonteCarloEngine
from core.models import GBM, Heston, LocalVol
from utils.data_loader_us import fetch_realtime_market_snapshot


def build_local_vol_from_smile(smile_df, spot_price):
    moneyness_grid = smile_df["moneyness"].values
    iv_grid = smile_df["impliedVolatility"].values

    def local_vol_func(S: np.ndarray, t: float) -> np.ndarray:
        m = S / spot_price
        sigma = np.interp(
            m,
            moneyness_grid,
            iv_grid,
            left=iv_grid[0],
            right=iv_grid[-1]
        )
        return np.clip(sigma, 0.05, 1.00)

    return local_vol_func


def main():
    print("=" * 70)
    print(" ❄️  US Snowball Pricing System (SPY, smile-driven) ❄️ ")
    print("=" * 70)

    snapshot = fetch_realtime_market_snapshot(symbol="SPY")

    spot_price = snapshot["spot"]
    hist_vol = snapshot["hist_vol"]
    q = snapshot["q"]
    r = snapshot["r"]
    smile_df = snapshot["smile"]
    heston_cfg = snapshot["heston_params"]

    print(f"标的: SPY")
    print(f"Spot: {spot_price:.4f}")
    print(f"Hist Vol: {hist_vol:.4%}")
    print(f"r: {r:.4%}")
    print(f"q: {q:.4%}")
    print(f"Smile Expiry: {snapshot['expiry']}")
    print(f"ATM IV: {heston_cfg['atm_iv']:.4%}")
    print(f"Put 90 IV: {heston_cfg['put90_iv']:.4%}")
    print()

    T = 1.0
    n_steps = 252
    n_paths = 20000

    ko_obs_indices = np.arange(21, 253, 21)
    ko_barrier = spot_price * 1.03
    ki_barrier = spot_price * 0.80

    snowball = SnowballOption(
        initial_price=spot_price,
        ko_barrier=ko_barrier,
        ki_barrier=ki_barrier,
        coupon_rate=0.18,
        ko_obs_indices=ko_obs_indices,
        T=T,
        notional=1_000_000.0
    )

    engine = MonteCarloEngine(n_paths=n_paths, n_steps=n_steps, seed=42)

    gbm_model = GBM(
        S0=spot_price,
        r=r,
        q=q,
        vol=hist_vol
    )

    heston_model = Heston(
        S0=spot_price,
        v0=heston_cfg["v0"],
        r=r,
        q=q,
        kappa=heston_cfg["kappa"],
        theta=heston_cfg["theta"],
        sigma_v=heston_cfg["sigma_v"],
        rho=heston_cfg["rho"]
    )

    lv_model = LocalVol(
        S0=spot_price,
        r=r,
        q=q,
        local_vol_func=build_local_vol_from_smile(smile_df, spot_price)
    )

    print("开始 GBM 定价...")
    gbm_result = engine.calculate_pv(option=snowball, model=gbm_model, r=r)
    print("GBM 完成")

    print("开始 Heston 定价...")
    heston_result = engine.calculate_pv(option=snowball, model=heston_model, r=r)
    print("Heston 完成")

    print("开始 Local Vol 定价...")
    lv_result = engine.calculate_pv(option=snowball, model=lv_model, r=r)
    print("Local Vol 完成")

    print("=" * 70)
    print(f"{'定价模型 (Model)':<28} | {'期权现值 (PV)':<18} | {'标准误 (SE)':<12}")
    print("-" * 70)
    print(f"{'GBM (Hist Vol)':<28} | ${gbm_result['price']:<16,.2f} | {gbm_result['standard_error']:<12,.2f}")
    print(f"{'Heston (Smile-based)':<28} | ${heston_result['price']:<16,.2f} | {heston_result['standard_error']:<12,.2f}")
    print(f"{'Local Vol (Smile interp)':<28} | ${lv_result['price']:<16,.2f} | {lv_result['standard_error']:<12,.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()