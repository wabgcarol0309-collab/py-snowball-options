import akshare as ak
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple

def fetch_cn_index_data(symbol: str = "000905", lookback_days: int = 365) -> Tuple[float, float]:
    """
    使用 AKShare 获取中国 A 股指数的最新收盘价和历史年化波动率。
    
    参数:
    symbol        : 指数代码，默认为 "000905" (中证500)
                    其他常用: "000852" (中证1000), "000300" (沪深300)
    lookback_days : 回溯的自然日数 (默认 365 天，用于覆盖过去一年的交易日)
    
    返回:
    (spot_price, annualized_vol)
    """
    index_name_map = {"000905": "中证500", "000852": "中证1000", "000300": "沪深300"}
    name = index_name_map.get(symbol, symbol)
    print(f"正在通过 AKShare 拉取 [{name}] 的市场数据...")
    
    # 动态计算开始和结束日期 (格式：YYYYMMDD)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    str_end = end_date.strftime("%Y%m%d")
    str_start = start_date.strftime("%Y%m%d")
    
    try:
        # 调用 AKShare 的 A 股指数历史行情接口
        # 返回的 DataFrame 包含: 日期, 开盘, 收盘, 最高, 最低, 成交量等
        hist_data = ak.index_zh_a_hist(symbol=symbol, period="daily", start_date=str_start, end_date=str_end)
    except Exception as e:
        raise ConnectionError(f"从 AKShare 获取数据失败，请检查网络或代码: {e}")
        
    if hist_data.empty:
        raise ValueError(f"未能获取到 {symbol} 的历史数据。")
        
    # 1. 获取最新收盘价
    # AKShare 返回的列名通常是中文 '收盘'
    spot_price = float(hist_data['收盘'].iloc[-1])
    
    # 2. 计算历史年化波动率
    # 利用对数收益率公式: ln(S_t / S_{t-1})
    close_prices = hist_data['收盘'].astype(float)
    log_returns = np.log(close_prices / close_prices.shift(1))
    
    daily_vol = log_returns.std()
    annualized_vol = daily_vol * np.sqrt(252) # A 股一年约 252 个交易日
    
    print(f"[{name}] 最新指数点位: {spot_price:.2f}, 过去一年历史波动率: {annualized_vol:.2%}")
    
    return spot_price, annualized_vol

def get_cn_risk_free_rate() -> float:
    """
    动态获取中国市场无风险利率。
    使用 AKShare 获取最新“中国10年期国债收益率”作为代理指标。
    """
    print("正在通过 AKShare 拉取最新 [中国10年期国债收益率] 作为无风险利率...")
    try:
        # 为了防止周末或法定长假（如春节/国庆）当天没有数据，我们往前推 30 天拉取，然后取最后一天
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
        bond_data = ak.bond_zh_us_rate(start_date=start_date)
        
        if bond_data.empty:
            raise ValueError("未能获取到国债收益率数据。")
            
        # 提取最后一行（即最新交易日）的 10 年期国债收益率
        # 注意：AKShare 返回的数值是百分比（例如 2.65 表示 2.65%），所以我们需要除以 100
        latest_rate = float(bond_data['中国国债收益率10年'].iloc[-1]) / 100.0
        latest_date = bond_data['日期'].iloc[-1]
        
        # 格式化日期用于打印
        if hasattr(latest_date, "strftime"):
            date_str = latest_date.strftime("%Y-%m-%d")
        else:
            date_str = str(latest_date)[:10]
            
        print(f"[国债市场] 最新观测日: {date_str}, 10年期国债收益率: {latest_rate:.4%}")
        return latest_rate
        
    except Exception as e:
        print(f"获取真实无风险利率失败: {e}。系统将退回使用默认值 2.0%。")
        return 0.02