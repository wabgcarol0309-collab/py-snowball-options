import numpy as np
import pandas as pd
import yfinance as yf


def fetch_us_spot_and_history(symbol: str = "SPY", hist_window: str = "6mo"):
    """
    获取现价、历史波动率、股息率
    """
    ticker = yf.Ticker(symbol)

    hist = ticker.history(period=hist_window, auto_adjust=False)
    if hist.empty:
        raise ValueError(f"无法获取 {symbol} 的历史数据")

    close = hist["Close"].dropna()
    if len(close) < 30:
        raise ValueError(f"{symbol} 历史数据不足")

    spot = float(close.iloc[-1])

    log_ret = np.log(close / close.shift(1)).dropna()
    hist_vol = float(log_ret.std() * np.sqrt(252))

    try:
        info = ticker.info
    except Exception:
        info = {}

    q = float(info.get("dividendYield", 0.0) or 0.0)

    return {
        "spot": spot,
        "hist_vol": hist_vol,
        "q": q,
        "history": hist,
    }


def fetch_us_risk_free_rate():
    """
    用 ^IRX 近似美国短端无风险利率
    """
    ticker = yf.Ticker("^IRX")
    hist = ticker.history(period="5d")

    if hist.empty:
        return 0.045

    latest = float(hist["Close"].dropna().iloc[-1])
    return latest / 100.0


def fetch_iv_smile(symbol: str = "SPY", expiry: str = None, option_type: str = "put"):
    """
    从 yfinance 拉取某一到期日的 IV smile
    自动跳过过近到期日，并对过滤条件做更稳健处理
    """
    ticker = yf.Ticker(symbol)
    expiries = ticker.options

    if not expiries:
        raise ValueError(f"{symbol} 没有可用期权到期日")

    spot_hist = ticker.history(period="5d")
    if spot_hist.empty:
        raise ValueError(f"无法获取 {symbol} 的现价")

    spot = float(spot_hist["Close"].dropna().iloc[-1])

    # 如果没有指定 expiry，就自动选择一个更合适的到期日
    if expiry is None:
        expiry = _select_reasonable_expiry(expiries)

    chain = ticker.option_chain(expiry)

    if option_type.lower() == "put":
        df = chain.puts.copy()
    elif option_type.lower() == "call":
        df = chain.calls.copy()
    else:
        raise ValueError("option_type must be 'put' or 'call'")

    if df.empty:
        raise ValueError(f"{symbol} {expiry} 的 {option_type} 期权链为空")

    df = df.copy()

    # 基础过滤：先不要太严格
    df = df[
        (df["impliedVolatility"].notna()) &
        (df["impliedVolatility"] > 0)
    ].copy()

    if df.empty:
        raise ValueError(f"{symbol} {expiry} 没有可用的 impliedVolatility 数据")

    df["moneyness"] = df["strike"] / spot
    df["mid"] = (df["bid"].fillna(0) + df["ask"].fillna(0)) / 2.0
    df["spread"] = df["ask"].fillna(0) - df["bid"].fillna(0)

    # 先保留更宽一点的区间
    df = df[(df["moneyness"] >= 0.75) & (df["moneyness"] <= 1.25)].copy()

    if df.empty:
        raise ValueError(f"{symbol} {expiry} 在有效 moneyness 区间内没有 smile 数据")

    # 优先用较稳的流动性过滤，但别太狠
    if "openInterest" in df.columns:
        df = df[df["openInterest"].fillna(0) >= 1].copy()

    # 如果太少，就退一步，不按 openInterest 过滤
    if len(df) < 5:
        if option_type.lower() == "put":
            df = chain.puts.copy()
        else:
            df = chain.calls.copy()

        df = df[
            (df["impliedVolatility"].notna()) &
            (df["impliedVolatility"] > 0)
        ].copy()

        df["moneyness"] = df["strike"] / spot
        df["mid"] = (df["bid"].fillna(0) + df["ask"].fillna(0)) / 2.0
        df["spread"] = df["ask"].fillna(0) - df["bid"].fillna(0)
        df = df[(df["moneyness"] >= 0.75) & (df["moneyness"] <= 1.25)].copy()

    df = df.sort_values("strike").reset_index(drop=True)

    if len(df) < 5:
        raise ValueError(
            f"{symbol} {expiry} 过滤后没有足够的 smile 数据，当前仅有 {len(df)} 个点"
        )

    return {
        "symbol": symbol,
        "expiry": expiry,
        "spot": spot,
        "smile": df[
            [
                "strike",
                "moneyness",
                "impliedVolatility",
                "bid",
                "ask",
                "mid",
                "spread",
                "openInterest",
                "volume",
            ]
        ].copy()
    }


def _select_reasonable_expiry(expiries):
    """
    优先选择不是当天、且距离当前有一定天数的到期日。
    一般选第一个 >= 7 天的到期日；若没有，则退化为最远的一个。
    """
    today = pd.Timestamp.today().normalize()

    parsed = []
    for e in expiries:
        dt = pd.Timestamp(e)
        dte = (dt - today).days
        parsed.append((e, dte))

    # 优先选 7~45 天
    candidates = [e for e, dte in parsed if 7 <= dte <= 45]
    if candidates:
        return candidates[0]

    # 再选 >= 3 天
    candidates = [e for e, dte in parsed if dte >= 3]
    if candidates:
        return candidates[0]

    # 最后兜底
    return expiries[min(1, len(expiries) - 1)]



def _nearest_iv(smile_df: pd.DataFrame, target_moneyness: float) -> float:
    """
    从 smile 中找到最接近目标 moneyness 的 IV
    """
    tmp = smile_df.copy()
    tmp["dist"] = np.abs(tmp["moneyness"] - target_moneyness)
    row = tmp.sort_values("dist").iloc[0]
    return float(row["impliedVolatility"])


def build_heston_params_from_smile(
    smile_df: pd.DataFrame,
    hist_vol: float,
):
    """
    用 smile 做一个经验型 Heston 参数映射
    不是严格优化校准，但比只用 ATM 一点更稳。
    """
    atm_iv = _nearest_iv(smile_df, 1.00)
    put95_iv = _nearest_iv(smile_df, 0.95)
    put90_iv = _nearest_iv(smile_df, 0.90)
    call105_iv = _nearest_iv(smile_df, 1.05)

    # 左偏斜：put wing 比 ATM 高多少
    downside_skew = max(put90_iv - atm_iv, 0.0)

    # 左右不对称程度
    asymmetry = max(put95_iv - call105_iv, 0.0)

    # 当前状态：用 ATM IV
    v0 = atm_iv ** 2

    # 长期均值：不要直接等于 ATM IV，用更稳的历史波动率作为锚
    theta = hist_vol ** 2

    # 均值回复速度：偏高一点，避免高 IV 长时间滞留
    kappa = 2.0 + min(downside_skew * 10.0, 1.5)

    # vol of vol：跟 skew 正相关
    sigma_v = 0.12 + min(downside_skew * 2.5, 0.25)

    # 相关系数：偏斜越大，rho 越负
    rho = -0.30 - min(asymmetry * 3.0, 0.35)

    # 做边界裁剪
    sigma_v = float(np.clip(sigma_v, 0.10, 0.40))
    rho = float(np.clip(rho, -0.85, -0.20))
    kappa = float(np.clip(kappa, 1.5, 3.5))

    return {
        "atm_iv": float(atm_iv),
        "put95_iv": float(put95_iv),
        "put90_iv": float(put90_iv),
        "call105_iv": float(call105_iv),
        "v0": float(v0),
        "theta": float(theta),
        "kappa": float(kappa),
        "sigma_v": float(sigma_v),
        "rho": float(rho),
    }


def fetch_realtime_market_snapshot(symbol: str = "SPY"):
    base = fetch_us_spot_and_history(symbol=symbol)
    r = fetch_us_risk_free_rate()

    try:
        smile_pack = fetch_iv_smile(symbol=symbol, option_type="put")
    except Exception:
        smile_pack = fetch_iv_smile(symbol=symbol, option_type="call")

    heston_params = build_heston_params_from_smile(
        smile_df=smile_pack["smile"],
        hist_vol=base["hist_vol"],
    )

    return {
        "symbol": symbol,
        "spot": base["spot"],
        "hist_vol": base["hist_vol"],
        "q": base["q"],
        "r": r,
        "expiry": smile_pack["expiry"],
        "smile": smile_pack["smile"],
        "heston_params": heston_params,
    }